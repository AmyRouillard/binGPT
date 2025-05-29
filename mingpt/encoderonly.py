import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently used in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class SelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    No causal mask is applied, making it suitable for encoders.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, attention_mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attention_mask is not None:
            # attention_mask is (B, 1, 1, T) where 0 indicates a padding token (mask it)
            # or (B, 1, T, T) for more complex masks
            att = att.masked_fill(attention_mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class EncoderBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)  # Non-causal self-attention
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                act=NewGELU(),  # or nn.GELU() if preferred
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            )
        )
        self.dropout = nn.Dropout(config.resid_pdrop)  # Dropout for the MLP path output

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        # MLP part
        m = self.mlp.c_fc(self.ln_2(x))
        m = self.mlp.act(m)
        m = self.mlp.c_proj(m)
        m = self.dropout(m)
        x = x + m
        return x


class EncoderOnlyTransformer(nn.Module):
    """Encoder-Only Transformer for N-to-N sequence tasks"""

    @staticmethod
    def get_default_config():
        C = CN()
        # model_type or (n_layer, n_head, n_embd) must be given
        C.model_type = "encoder-base"  # Default to a new type
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None  # Size of the input vocabulary
        C.output_vocab_size = None  # Size of the output vocabulary (if different, e.g. for classification)
        # Defaults to vocab_size if not set
        C.block_size = None  # Maximum sequence length
        C.pad_token_id = None  # Optional: ID of the padding token for auto-masking
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1

        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        self.config = config  # Save config

        if config.to_dict().get("output_vocab_size", None) is None:
            config.output_vocab_size = (
                config.vocab_size
            )  # Default output vocab to input vocab

        type_given = config.model_type is not None
        params_given = all(
            [
                config.n_layer is not None,
                config.n_head is not None,
                config.n_embd is not None,
            ]
        )
        assert type_given ^ params_given  # exactly one of these (XOR)

        if type_given:
            config.merge_from_dict(
                {
                    # Add presets for encoder models (e.g., inspired by BERT sizes)
                    "encoder-tiny": dict(n_layer=2, n_head=2, n_embd=128),
                    "encoder-mini": dict(n_layer=6, n_head=6, n_embd=192),
                    "encoder-small": dict(n_layer=6, n_head=8, n_embd=512),
                    "encoder-base": dict(n_layer=12, n_head=12, n_embd=768),
                    "encoder-large": dict(n_layer=24, n_head=16, n_embd=1024),
                }.get(
                    config.model_type, {}
                )  # Use .get for safety, though XOR assert helps
            )
            # After merge, ensure n_layer, n_head, n_embd are set if model_type was valid
            assert all(
                [
                    config.n_layer is not None,
                    config.n_head is not None,
                    config.n_embd is not None,
                ]
            ), f"Model type {config.model_type} not found in presets or presets are incomplete."

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # For N-to-N prediction, a head is applied to each output token's representation
        self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith(
                "c_proj.weight"
            ):  # Targets SelfAttention.c_proj and EncoderBlock.mlp.c_proj
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        n_params_transformer = sum(p.numel() for p in self.transformer.parameters())
        n_params_head = sum(p.numel() for p in self.lm_head.parameters())
        total_params = n_params_transformer + n_params_head
        print(f"number of transformer parameters: {n_params_transformer:.3e}")
        print(f"number of prediction head parameters: {n_params_head:.3e}")
        print(f"total number of parameters: {total_params:.3e}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        Separates out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:  # Skip parameters that don't require gradients
                    continue
                fpn = "%s.%s" % (mn, pn) if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Sanity check
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"

        # Check if all learnable parameters are covered
        uncovered_params = set(param_dict.keys()) - union_params
        if (
            uncovered_params
        ):  # Allow if only LayerNorm scale/bias are uncovered (if they were not in named_modules for some reason)
            # Typically, LayerNorm weights are 'weight' and 'bias', handled by blacklist and bias rules
            print(
                f"Warning: parameters {str(uncovered_params)} were not separated into either decay/no_decay set!"
            )
        # assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(self, idx, targets=None, attention_mask=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        # Create attention mask if not provided and pad_token_id is configured
        if attention_mask is None and self.config.pad_token_id is not None:
            # (B, T) -> (B, 1, 1, T). Mask should be 1 for non-pad, 0 for pad.
            # masked_fill expects True for positions to MASK (fill with -inf)
            # So if mask is 0 for pad, (mask == 0) will be True for pad positions.
            pad_mask = idx == self.config.pad_token_id  # True for pad tokens
            attention_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            # In SelfAttention, we use: att = att.masked_fill(attention_mask == True, float('-inf'))
            # So, if attention_mask here means "True where it's a pad", it's correct.
            # Let's rename to pad_attention_mask to be clear this is for padding.
            # The SelfAttention module expects 0 where it should be masked.
            # Let's adjust: 1 for non-pad, 0 for pad.
            attention_mask = (
                (idx != self.config.pad_token_id).unsqueeze(1).unsqueeze(2).to(x.dtype)
            )  # (B,1,1,T)
            # SelfAttention uses: att.masked_fill(attention_mask == 0, float('-inf'))

        # Forward the Transformer
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)  # Pass mask to each block

        x = self.transformer.ln_f(x)  # (b, t, n_embd)
        logits = self.lm_head(x)  # (b, t, output_vocab_size)

        loss = None
        if targets is not None:
            # For N-to-N, targets are (b, t)
            ignore_index = getattr(self.config, "pad_token_id", -100)
            if ignore_index is None:
                ignore_index = -100

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )
            # Use pad_token_id for ignore_index if available, otherwise common -100

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, temperature=1.0, do_sample=False, top_k=None):

        assert (
            idx.size(1) == self.block_size
        ), f"Cannot generate sequence of length {idx.size(1)}, block size is {self.block_size}"

        # forward the model to get the logits for the index in the sequence
        logits, _ = self(idx)

        # scale the logits by temperature
        logits = logits / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            # get the top k logits and their indices
            _, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
            # create a mask for the rest of the logits
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(dim=-1, index=top_k_indices, value=False)
            # set the rest of the logits to -inf
            logits.masked_fill_(mask, float("-inf"))
            # now logits only contain the top k options

        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        return idx_next


class EncoderOnlyTransformerForProbing(EncoderOnlyTransformer):
    """GPT Language Model with probing head"""

    def __init__(self, config, probe_layer=0):
        super().__init__(config)
        self.probe_layer = probe_layer

        # assert that probe_layer  is a valid layer index
        assert (
            probe_layer <= config.n_layer
        ), f"probe_layer  {probe_layer } is out of range"
        assert probe_layer >= 0, f"probe_layer  {probe_layer } is out of range"

    @torch.no_grad()
    def forward_1of2(self, idx, attention_mask=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        # Create attention mask if not provided and pad_token_id is configured
        if attention_mask is None and self.config.pad_token_id is not None:
            # (B, T) -> (B, 1, 1, T). Mask should be 1 for non-pad, 0 for pad.
            # masked_fill expects True for positions to MASK (fill with -inf)
            # So if mask is 0 for pad, (mask == 0) will be True for pad positions.
            pad_mask = idx == self.config.pad_token_id  # True for pad tokens
            attention_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            # In SelfAttention, we use: att = att.masked_fill(attention_mask == True, float('-inf'))
            # So, if attention_mask here means "True where it's a pad", it's correct.
            # Let's rename to pad_attention_mask to be clear this is for padding.
            # The SelfAttention module expects 0 where it should be masked.
            # Let's adjust: 1 for non-pad, 0 for pad.
            attention_mask = (
                (idx != self.config.pad_token_id).unsqueeze(1).unsqueeze(2).to(x.dtype)
            )  # (B,1,1,T)
            # SelfAttention uses: att.masked_fill(attention_mask == 0, float('-inf'))

        # Forward the Transformer
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h[: self.probe_layer]:
            x = block(x, attention_mask=attention_mask)  # TODO deeper into the block?

        return x

    @torch.no_grad()
    def forward_2of2(self, x, attention_mask=None):

        for block in self.transformer.h[self.probe_layer :]:
            x = block(x, attention_mask=attention_mask)  # TODO deeper into the block?

        x = self.transformer.ln_f(x)  # (b, t, n_embd)
        logits = self.lm_head(x)  # (b, t, output_vocab_size)

        return logits
