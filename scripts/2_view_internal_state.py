# %%

from utils.tentmapdataset import ProbeDataset

from mingpt.model import Probe
from mingpt.encoderonly import EncoderOnlyTransformerForProbing, EncoderOnlyTransformer
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN

from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import json
import time
import torch
import csv
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns

# %%

wdir = "/home/amyrouillard/project-files/"  # "C:/Users/Amy/Desktop/Green_Git/binGPT/" #"/mnt/lustre/users/arouillard/project-files/"  #
# wdir = "D:/home/amyrouillard/project-files/"
model_dir = wdir + f"models/2025_06_02_14_44"  # 2025_05_29_09_29
transformer_load_epoch = 55
# model_dir = wdir + f"models/2025_06_02_15_24"
# transformer_load_epoch = 83
# model_dir = wdir + f"models/2025_06_02_15_40"
# transformer_load_epoch = 81
num_workers = 8

if os.path.exists(os.path.join(model_dir, "config.json")):
    # read json file
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        configs = json.load(f)
else:
    raise ValueError("No config.json found in model_dir, using default configs.")


# check if config.json exist in model_dir, if not create it
if os.path.exists(os.path.join(model_dir, "model_config.json")):
    # read json file
    with open(os.path.join(model_dir, "model_config.json"), "r") as f:
        model_config_dict = json.load(f)
else:
    raise ValueError("No model_config.json found in model_dir, using default configs.")

if os.path.exists(os.path.join(model_dir, "trainer_config.json")):
    with open(os.path.join(model_dir, "trainer_config.json"), "r") as f:
        train_config_dict = json.load(f)

    train_config = Trainer.get_default_config()
    train_config.merge_from_dict(train_config_dict)

else:
    raise ValueError(
        "No trainer_config.json found in model_dir, using default configs."
    )

# %%
model_config = CN(**model_config_dict)
model = EncoderOnlyTransformer(model_config)

model.load_state_dict(
    torch.load(os.path.join(model_dir, f"model_{transformer_load_epoch}.pt"))
)
model.eval()

# %%

show_all = True


# %%

idx = [[0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
idx = torch.tensor(
    idx, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
)
targets = [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
targets = torch.tensor(
    targets, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
)

# idx = [[0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# idx = torch.tensor(
#     idx, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
# )
# targets = [[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# targets = torch.tensor(
#     targets, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
# )

# idx = [[0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
# idx = torch.tensor(
#     idx, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
# )
# targets = [[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# targets = torch.tensor(
#     targets, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
# )

# idx = [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
# idx = torch.tensor(
#     idx, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
# )
# targets = [[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# targets = torch.tensor(
#     targets, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
# )

# %%

# idx = [[0, 1, 1, 1, 0, 1, 0,1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
# idx = torch.tensor(
#     idx, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
# )
# targets = [[1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# targets = torch.tensor(
#     targets, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
# )

# %%

print(f"idx: {idx} {idx.shape}")
print(f"targets: {targets} {targets.shape}")

# %%

device = idx.device
b, t = idx.size()
attention_mask = None

# print(f"b: {b}, t: {t}, device: {device}")
# print(
#     f"attention_mask: {attention_mask} {attention_mask.shape if attention_mask is not None else None}"
# )
# %%

pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

# print(f"pos: {pos} {pos.shape}")

# token embeddings of shape (b, t, n_embd)
tok_emb = model.transformer.wte(idx)
# position embeddings of shape (1, t, n_embd)
pos_emb = model.transformer.wpe(pos)

x = model.transformer.drop(tok_emb + pos_emb)  # shape (b, t, n_embd)

z_min = min(x.min().item(), pos_emb.min().item(), tok_emb.min().item())
z_max = max(x.max().item(), pos_emb.max().item(), tok_emb.max().item())

if show_all:
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    sns.heatmap(
        tok_emb[0].detach().cpu().numpy(),
        ax=ax[0],
        cmap="RdYlGn",
        vmin=z_min,
        vmax=z_max,
        cbar=False,
        # cbar_kws={"label": "Embedding Value"},
        # annot=True,
        # fmt=".2f",
    )
    ax[0].set_title("Token Embeddings")
    ax[0].set_xlabel("Embedding Dimension")
    ax[0].set_ylabel("Token Position")

    sns.heatmap(
        pos_emb[0].detach().cpu().numpy(),
        ax=ax[1],
        cmap="RdYlGn",
        vmin=z_min,
        vmax=z_max,
        cbar=False,
        # cbar_kws={"label": "Embedding Value"},
    )
    ax[1].set_title("Position Embeddings")
    ax[1].set_xlabel("Embedding Dimension")
    ax[1].set_ylabel("Token Position")

    sns.heatmap(
        x[0].detach().cpu().numpy(),
        ax=ax[2],
        cmap="RdYlGn",
        vmin=z_min,
        vmax=z_max,
        cbar_kws={"label": "Embedding Value"},
    )
    ax[2].set_title("Combined Embeddings")
    ax[2].set_xlabel("Embedding Dimension")
    ax[2].set_ylabel("Token Position")

    fig.tight_layout()

# %%

print(f"Number of layers: {len(model.transformer.h)}")
# %%
#########################################################################

#########################################################################

for block in model.transformer.h:
    # Self-attention part
    a = block.ln_1(x)
    B, T, C = a.size()  # batch size, sequence length, embedding dimensionality (n_embd)
    # print(f"B: {B}, T: {T}, C: {C}")

    if show_all:
        fig, ax = plt.subplots(1, 4, figsize=(12, 6))
        sns.heatmap(
            x[0].detach().cpu().numpy(),
            ax=ax[0],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[0].set_title("Combined Embeddings")
        ax[0].set_xlabel("Embedding Dimension")
        ax[0].set_ylabel("Token Position")

        x_mean = x.mean(dim=-1)  # (b, t, 1)
        x_std = x.std(dim=-1)  # (b, t, 1)
        # join mean and std to get (b, t, 2)
        tmp = torch.cat((x_mean, x_std), dim=0).T  # (b, t, 2)
        sns.heatmap(
            tmp.detach().cpu().numpy(),
            ax=ax[1],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            annot=True,
            fmt=".2f",
        )

        sns.heatmap(
            a[0].detach().cpu().numpy(),
            ax=ax[2],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[2].set_title("Scaled Combined Embeddings")
        ax[2].set_xlabel("Embedding Dimension")
        ax[2].set_ylabel("Token Position")

        x_mean = a.mean(dim=-1)  # (b, t, 1)
        x_std = a.std(dim=-1)  # (b, t, 1)
        # join mean and std to get (b, t, 2)
        tmp = torch.cat((x_mean, x_std), dim=0).T  # (b, t, 2)
        sns.heatmap(
            tmp.detach().cpu().numpy(),
            ax=ax[3],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            annot=True,
            fmt=".2f",
        )

        fig.tight_layout()

    if show_all:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(
            a[0].detach().cpu().numpy(),
            ax=ax[0],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[0].set_title("Linear layer input")
        ax[0].set_xlabel("Embedding Dimension")
        ax[0].set_ylabel("Token Position")

    # first linear projection
    # (B, T, C) -> (B, T, 3 * C)
    a = block.attn.c_attn(a)

    if show_all:
        sns.heatmap(
            a[0].detach().cpu().numpy(),
            ax=ax[1],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[1].set_title("Linear layer output")
        ax[1].set_xlabel("Embedding Dimension")
        ax[1].set_ylabel("Token Position")

        fig.tight_layout()

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v = a.split(block.attn.n_embd, dim=2)

    if show_all:
        z_min = min(q.min().item(), k.min().item(), v.min().item())
        z_max = max(q.max().item(), k.max().item(), v.max().item())

        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        sns.heatmap(
            q[0].detach().cpu().numpy(),
            ax=ax[0],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            cbar=False,
            vmin=z_min,
            vmax=z_max,
        )
        ax[0].set_title("Query Embeddings")
        ax[0].set_xlabel("Embedding Dimension")
        ax[0].set_ylabel("Token Position")
        sns.heatmap(
            k[0].detach().cpu().numpy(),
            ax=ax[1],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            cbar=False,
            vmin=z_min,
            vmax=z_max,
        )
        ax[1].set_title("Key Embeddings")
        ax[1].set_xlabel("Embedding Dimension")
        ax[1].set_ylabel("Token Position")
        sns.heatmap(
            v[0].detach().cpu().numpy(),
            ax=ax[2],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            cbar=True,
            vmin=z_min,
            vmax=z_max,
        )
        ax[2].set_title("Value Embeddings")
        ax[2].set_xlabel("Embedding Dimension")
        ax[2].set_ylabel("Token Position")

        fig.tight_layout()

    k = k.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(
        1, 2
    )  # (B, nh, T, hs)
    q = q.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(
        1, 2
    )  # (B, nh, T, hs)
    v = v.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(
        1, 2
    )  # (B, nh, T, hs)

    # print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")

    if show_all:
        fig, ax = plt.subplots(2, 3, figsize=(12, 6))

        for i in range(2):
            sns.heatmap(
                q[0, i].detach().cpu().numpy(),
                ax=ax[i, 0],
                cmap="RdYlGn",
                cbar_kws={"label": "Embedding Value"},
                cbar=False,
                vmin=z_min,
                vmax=z_max,
            )
            ax[i, 0].set_title("Query Embeddings" + " (Head {})".format(i))
            ax[i, 0].set_xlabel("Embedding Dimension")
            ax[i, 0].set_ylabel("Token Position")
            sns.heatmap(
                k[0, i].detach().cpu().numpy(),
                ax=ax[i, 1],
                cmap="RdYlGn",
                cbar_kws={"label": "Embedding Value"},
                cbar=False,
                vmin=z_min,
                vmax=z_max,
            )
            ax[i, 1].set_title("Key Embeddings" + " (Head {})".format(i))
            ax[i, 1].set_xlabel("Embedding Dimension")
            ax[i, 1].set_ylabel("Token Position")
            sns.heatmap(
                v[0, i].detach().cpu().numpy(),
                ax=ax[i, 2],
                cmap="RdYlGn",
                cbar_kws={"label": "Embedding Value"},
                cbar=True,
                vmin=z_min,
                vmax=z_max,
            )
            ax[i, 2].set_title("Value Embeddings" + " (Head {})".format(i))
            ax[i, 2].set_xlabel("Embedding Dimension")
            ax[i, 2].set_ylabel("Token Position")

        fig.tight_layout()

    # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

    if attention_mask is not None:
        # attention_mask is (B, 1, 1, T) where 0 indicates a padding token (mask it)
        # or (B, 1, T, T) for more complex masks
        att = att.masked_fill(attention_mask == 0, float("-inf"))

    att = F.softmax(att, dim=-1)
    att = block.attn.attn_dropout(att)

    z_min = att.min().item()
    z_max = att.max().item()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(2):
        sns.heatmap(
            att[0, i].detach().cpu().numpy(),
            ax=ax[i],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            cbar=True if i == 1 else False,
            vmin=z_min,
            vmax=z_max,
        )
        ax[i].set_title("q @ k" + " (Head {})".format(i))
        ax[i].set_xlabel("Embedding Dimension")
        ax[i].set_ylabel("Token Position")

    fig.tight_layout()

    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

    z_min = y.min().item()
    z_max = y.max().item()

    if show_all:
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        for i in range(2):
            sns.heatmap(
                y[0, i].detach().cpu().numpy(),
                ax=ax[0, i],
                cmap="RdYlGn",
                cbar_kws={"label": "Embedding Value"},
                cbar=True if i == 1 else False,
                vmin=z_min,
                vmax=z_max,
                annot=True,
                fmt=".3f",
            )
            ax[0, i].set_title("q @ k @ v" + " (Head {})".format(i))
            ax[0, i].set_xlabel("Embedding Dimension")
            ax[0, i].set_ylabel("Token Position")

    y = (
        y.transpose(1, 2).contiguous().view(B, T, C)
    )  # re-assemble all head outputs side by side

    if show_all:
        sns.heatmap(
            y[0].detach().cpu().numpy(),
            ax=ax[1, 0],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            cbar=False,
            vmin=z_min,
            vmax=z_max,
        )
        ax[1, 0].set_title("Re-assembled q @ k @ v")
        ax[1, 0].set_xlabel("Embedding Dimension")
        ax[1, 0].set_ylabel("Token Position")

    # output projection
    y = block.attn.c_proj(y)
    y = block.attn.resid_dropout(y)

    if show_all:
        sns.heatmap(
            y[0].detach().cpu().numpy(),
            ax=ax[1, 1],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            cbar=True,
        )
        ax[1, 1].set_title("Output Linear Projection")
        ax[1, 1].set_xlabel("Embedding Dimension")
        ax[1, 1].set_ylabel("Token Position")

        fig.tight_layout()

    if show_all:
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))

        sns.heatmap(
            x[0].detach().cpu().numpy(),
            ax=ax[0],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            cbar=True,
        )
        ax[0].set_title("")
        ax[0].set_xlabel("Embedding Dimension")
        ax[0].set_ylabel("Token Position")

        sns.heatmap(
            y[0].detach().cpu().numpy(),
            ax=ax[1],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            cbar=True,
        )
        ax[1].set_title("")
        ax[1].set_xlabel("Embedding Dimension")
        ax[1].set_ylabel("Token Position")

    x = x + y

    if show_all:
        sns.heatmap(
            x[0].detach().cpu().numpy(),
            ax=ax[2],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            cbar=True,
        )
        ax[2].set_title("")
        ax[2].set_xlabel("Embedding Dimension")
        ax[2].set_ylabel("Token Position")

        fig.tight_layout()

    # MLP part
    m = block.ln_2(x)

    if show_all:
        fig, ax = plt.subplots(1, 4, figsize=(12, 6))
        sns.heatmap(
            x[0].detach().cpu().numpy(),
            ax=ax[0],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[0].set_title("Combined Embeddings")
        ax[0].set_xlabel("Embedding Dimension")
        ax[0].set_ylabel("Token Position")
        x_mean = x.mean(dim=-1)  # (b, t, 1)
        x_std = x.std(dim=-1)  # (b, t, 1)
        tmp = torch.cat((x_mean, x_std), dim=0).T  # (b, t, 2)
        sns.heatmap(
            tmp.detach().cpu().numpy(),
            ax=ax[1],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            annot=True,
            fmt=".2f",
        )
        sns.heatmap(
            m[0].detach().cpu().numpy(),
            ax=ax[2],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[2].set_title("Scaled Combined Embeddings")
        ax[2].set_xlabel("Embedding Dimension")
        ax[2].set_ylabel("Token Position")

        x_mean = m.mean(dim=-1)  # (b, t, 1)
        x_std = m.std(dim=-1)  # (b, t, 1)
        # join mean and std to get (b, t, 2)
        tmp = torch.cat((x_mean, x_std), dim=0).T  # (b, t, 2)
        sns.heatmap(
            tmp.detach().cpu().numpy(),
            ax=ax[3],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
            annot=True,
            fmt=".2f",
        )

        fig.tight_layout()

    if show_all:
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        sns.heatmap(
            m[0].detach().cpu().numpy(),
            ax=ax[0, 0],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[0, 0].set_title("Input to MLP")
        ax[0, 0].set_xlabel("Embedding Dimension")
        ax[0, 0].set_ylabel("Token Position")

    m = block.mlp.c_fc(m)

    if show_all:
        a2 = sns.heatmap(
            m[0].detach().cpu().numpy(),
            ax=ax[0, 1],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[0, 1].set_title("First Linear Projection")
        ax[0, 1].set_xlabel("Embedding Dimension")
        ax[0, 1].set_ylabel("Token Position")

    m = block.mlp.act(m)

    if show_all:
        a3 = sns.heatmap(
            m[0].detach().cpu().numpy(),
            ax=ax[1, 0],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[1, 0].set_title("Activation Function")
        ax[1, 0].set_xlabel("Embedding Dimension")
        ax[1, 0].set_ylabel("Token Position")

    m = block.mlp.c_proj(m)
    m = block.dropout(m)

    if show_all:
        sns.heatmap(
            m[0].detach().cpu().numpy(),
            ax=ax[1, 1],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[1, 1].set_title("Output Linear Projection")
        ax[1, 1].set_xlabel("Embedding Dimension")
        ax[1, 1].set_ylabel("Token Position")

        fig.tight_layout()

    if show_all:
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))

        sns.heatmap(
            x[0].detach().cpu().numpy(),
            ax=ax[0, 0],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[0, 0].set_title("Combined Embeddings")
        ax[0, 0].set_xlabel("Embedding Dimension")
        ax[0, 0].set_ylabel("Token Position")

    x = x + m

    if show_all:
        sns.heatmap(
            y[0].detach().cpu().numpy(),
            ax=ax[0, 1],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[0, 1].set_title("Attention Output")
        ax[0, 1].set_xlabel("Embedding Dimension")
        ax[0, 1].set_ylabel("Token Position")

        sns.heatmap(
            m[0].detach().cpu().numpy(),
            ax=ax[1, 0],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[1, 0].set_title("MLP")
        ax[1, 0].set_xlabel("Embedding Dimension")
        ax[1, 0].set_ylabel("Token Position")

        sns.heatmap(
            x[0].detach().cpu().numpy(),
            ax=ax[1, 1],
            cmap="RdYlGn",
            cbar_kws={"label": "Embedding Value"},
        )
        ax[1, 1].set_title("Combined Output")
        ax[1, 1].set_xlabel("Embedding Dimension")
        ax[1, 1].set_ylabel("Token Position")

        fig.tight_layout()

#########################################################################

#########################################################################
# %%

if show_all:
    fig, ax = plt.subplots(1, 4, figsize=(12, 6))
    sns.heatmap(
        x[0].detach().cpu().numpy(),
        ax=ax[0],
        cmap="RdYlGn",
        cbar_kws={"label": "Embedding Value"},
    )
    ax[0].set_title("Combined Output")
    ax[0].set_xlabel("Embedding Dimension")
    ax[0].set_ylabel("Token Position")
    x_mean = x.mean(dim=-1)  # (b, t, 1)
    x_std = x.std(dim=-1)  # (b, t, 1)
    tmp = torch.cat((x_mean, x_std), dim=0).T  # (b, t, 2)
    sns.heatmap(
        tmp.detach().cpu().numpy(),
        ax=ax[1],
        cmap="RdYlGn",
        cbar_kws={"label": "Embedding Value"},
        annot=True,
        fmt=".2f",
    )

x = model.transformer.ln_f(x)  # (b, t, n_embd)

if show_all:
    sns.heatmap(
        x[0].detach().cpu().numpy(),
        ax=ax[2],
        cmap="RdYlGn",
        cbar_kws={"label": "Embedding Value"},
    )
    ax[2].set_title("Scaled Combined Output")
    ax[2].set_xlabel("Embedding Dimension")
    ax[2].set_ylabel("Token Position")

    x_mean = x.mean(dim=-1)  # (b, t, 1)
    x_std = x.std(dim=-1)  # (b, t, 1)
    # join mean and std to get (b, t, 2)
    tmp = torch.cat((x_mean, x_std), dim=0).T  # (b, t, 2)
    sns.heatmap(
        tmp.detach().cpu().numpy(),
        ax=ax[3],
        cmap="RdYlGn",
        cbar_kws={"label": "Embedding Value"},
        annot=True,
        fmt=".2f",
    )

    fig.tight_layout()

# %%

logits = model.lm_head(x)  # (b, t, output_vocab_size)

if show_all:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(
        x[0].detach().cpu().numpy(),
        ax=ax[0],
        cmap="RdYlGn",
        cbar_kws={"label": "Logit Value"},
    )
    sns.heatmap(
        logits.softmax(dim=-1)[0].detach().cpu().numpy(),
        ax=ax[1],
        cmap="RdYlGn",
        cbar_kws={"label": "Logit Value"},
        annot=True,
        fmt=".2f",
    )

# %%

loss = None
if targets is not None:
    # For N-to-N, targets are (b, t)
    ignore_index = getattr(model.config, "pad_token_id", -100)
    if ignore_index is None:
        ignore_index = -100

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )
    # Use pad_token_id for ignore_index if available, otherwise common -100


# %%
model_logits, model_loss = model(idx, targets=targets, attention_mask=attention_mask)
model_pred = model_logits.softmax(dim=-1).argmax(dim=-1)  # (b, t)


print(f"q: {configs["n"]}")
print(f"Idx:          {idx} {idx.shape}")
# softmax the logits to get probabilities
pred = logits.softmax(dim=-1).argmax(dim=-1)  # (b, t)
print(f"Pred:         {pred} {pred.shape}")
print(f"Pred (model): {model_pred} {model_pred.shape}")
print(f"Targets:      {targets} {targets.shape}")

print(f"Loss:       {loss.item() if loss is not None else None}")
print(f"Model Loss: {model_loss.item() if model_loss is not None else None}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(
    model_logits.softmax(dim=-1)[0].detach().cpu().numpy(),
    ax=ax[0],
    cmap="RdYlGn",
    cbar_kws={"label": "Logit Value"},
    annot=True,
    fmt=".4f",
)
ax[0].set_title("Model Logits")
ax[0].set_xlabel("Token Position")
ax[0].set_ylabel("Output Vocabulary Size")

sns.heatmap(
    logits.softmax(dim=-1)[0].detach().cpu().numpy(),
    ax=ax[1],
    cmap="RdYlGn",
    cbar_kws={"label": "Logit Value"},
    annot=True,
    fmt=".4f",
)
ax[1].set_title("Logits")
ax[1].set_xlabel("Token Position")
ax[1].set_ylabel("Output Vocabulary Size")
fig.tight_layout()

# %%
from utils.tentmapdataset import TentDataset

train_dataset = TentDataset(
    "train",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)
test_dataset = TentDataset(
    "test",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)
validation_dataset = TentDataset(
    "validation",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)


loader = DataLoader(
    train_dataset + validation_dataset + test_dataset,
    shuffle=False,
    # pin_memory=True,
    batch_size=2**15,
    num_workers=num_workers,
)

mistakes = []
for batch in loader:
    idx, targets = batch
    idx = idx.to(device)
    targets = targets.to(device)

    model_logits, _ = model(idx)
    model_pred = model_logits.softmax(dim=-1).argmax(dim=-1)  # (b, t)

    mask_correct = (model_pred == targets).all(dim=-1)

    if mask_correct.sum() == idx.size(0):
        continue

    mistakes.extend(
        [
            idx[~mask_correct].cpu().numpy(),
        ]
    )

# %%

mistakes = np.concatenate(mistakes, axis=0)
print(f"Mistakes: {mistakes.shape}")
# %%
for m in mistakes:
    print(m)
# %%

# # count the number of mistakes that have 0 at index i
# for i in range(mistakes.shape[1]):
#     print(f"Index {i}: {np.sum(mistakes[:, i] == 0)} mistakes")

count = 0
for m in mistakes:
    if m[9] == 1:
        print(m)
        count += 1

print(f"Count: {count} mistakes with 1 at index 9")

# %%
