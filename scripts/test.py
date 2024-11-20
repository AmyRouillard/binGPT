# %%

from utils.tentmapdataset import TentMap2Dataset

dataset = TentMap2Dataset()
X, Y = dataset.generate_batch(10)

[print(x, y) for x, y in zip(X, Y)]
# %%
dataset2 = TentMap2Dataset(iterations=2)
X, Y = dataset2.generate_batch(10)

[print(x, y) for x, y in zip(X, Y)]

# %%

from utils.gol import ConwayGame
import matplotlib.pyplot as plt
from mingpt.bpe import get_encoder


# %%

# plot game.grid as a heatmap
fig, ax = plt.subplots(1, 2, figsize=(10, 5))


N = 8
game = ConwayGame(width=N, height=N)
game.randomize_grid_uniform()

text_0 = game.get_state_as_string(game.grid)
print(text_0)

ax[0].imshow(game.grid, cmap="binary")

game.update_grid()
text_1 = game.get_state_as_string(game.grid)
print(text_0)

ax[1].imshow(game.grid, cmap="binary")
# %%

bos = "@"
eos = "$"

prompt_template = bos + "{}>{}" + eos

prompt = prompt_template.format(text_0, text_1)
print(prompt)
# %%


# text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
text = text_0
e = get_encoder()
r = e.encode_and_show_work(text)
# %%

print("Original text is:")
print(text)
print("First the text gets pre-tokenized, broken up into chunks, the outcome is:")
print(r["tokens"])
print("Then we iterate over each chunk and process them in turn...")
for part in r["parts"]:
    print(part)
print("and the final outcome is concatenating and flattening all the token_ix:")
print(r["bpe_idx"])

# %%

from utils.tokenizer import Tokenizer, CustomTokenizer

# %%

# https://www.cogsci.ed.ac.uk/~richard/utf-8.cgi?input=0&mode=chars
tokenizer = Tokenizer(n_pad=8 * 8 * 2 + 3)

bpe = tokenizer.tokenize_str(prompt)

print(bpe)

# %%

tokenizer = Tokenizer(n_pad=dataset2.max_length * 2 - dataset2.iterations + 3)

print(dataset2.max_length, len(X[1]), len(Y[1]))
prompt = prompt_template.format(X[1], Y[1])
print(prompt)
bpe = tokenizer.tokenize_str(prompt)
print(bpe)
# %%

from mingpt.utils import CfgNode as CN

config = CN(
    n_layer=1,
    n_head=1,
    vocab_size=2**8,
    block_size=tokenizer.n_pad,
    model_type=None,
    n_embd=tokenizer.n_pad,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
)
# %%
from mingpt.model import GPT

model = GPT(config)


# %%

# reshape bpe to a batch of size 1
bpe = bpe.unsqueeze(0)

# %%
# pass the bpe through the model
out = model(bpe)

# %%
# take the argmax of the prediction
pred = out[0].argmax(dim=-1)
# decode the prediction
pred_texts = tokenizer.sequences_to_texts(pred)

print(pred_texts)
# %%


dataset2 = TentMap2Dataset(iterations=2)
X, Y = dataset2.generate_batch(10)

bos = None
eos = "$"
prompt_template = "{}>{}" + eos

tokenizer = CustomTokenizer(
    bos_token=bos,
    eos_token=eos,
    n_pad=dataset2.max_length * 2 - dataset2.iterations + 3,
)
prompts = [prompt_template.format(x, y) for x, y in zip(X, Y)]
print(prompts)
bpe = tokenizer.texts_to_sequences(prompts)
print(bpe)
pred_text = tokenizer.sequences_to_texts(pred)
print(pred_text)

# %%
config = CN(
    n_layer=1,
    n_head=1,
    vocab_size=tokenizer.vocab_size,
    block_size=tokenizer.n_pad,
    model_type=None,
    n_embd=tokenizer.n_pad,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
)

print(config)

model = GPT(config)
# %%

out = model(bpe)

# take the argmax of the prediction
pred = out[0].argmax(dim=-1)
# decode the prediction
pred_text = tokenizer.sequences_to_texts(pred)

print(pred_text)

# %%

from mingpt.utils import CfgNode as CN
from mingpt.model import GPT
from utils.tokenizer import CustomTokenizer
from utils.tentmapdataset import TentMap2Dataset

# %%

# create a dataset that is x,y pairs wher y = f^n(x)
n = 2
dataset = TentMap2Dataset(iterations=n)
# batch of 10 samples
X, Y = dataset.generate_batch(10)

# %%
bos = None  # no beginning of sentence token
eos = "$"  # end of sentence token
prompt_template = "{}>{}" + eos  # template for the prompt

# create a custom tokenizer
tokenizer = CustomTokenizer(
    bos_token=bos,
    eos_token=eos,
    n_pad=dataset.max_length * 2 - dataset.iterations + 3,
)
prompts = [prompt_template.format(x, y) for x, y in zip(X, Y)]
print(prompts)

# %%

tokenized_texts = tokenizer.texts_to_sequences(prompts)

print("Tokenized texts:")
print(tokenized_texts)
print("Decoded texts:")
print(tokenizer.sequences_to_texts(tokenized_texts))

# %%
config = CN(
    # model parameters
    n_layer=3,
    n_head=3,
    n_embd=tokenizer.n_pad,
    vocab_size=tokenizer.vocab_size,
    block_size=tokenizer.n_pad,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
    # training
    # device to train on
    device="auto",
    # dataloder parameters
    num_workers=4,
    # optimizer parameters
    max_iters=None,
    batch_size=64,
    learning_rate=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,  # only applied on matmul weights
    grad_norm_clip=1.0,
)

print(config)

model = GPT(config)
# %%

# out = model(tokenized_texts)

# # take the argmax of the prediction
# pred = out[0].argmax(dim=-1)
# # decode the prediction
# pred_text = tokenizer.sequences_to_texts(pred)

# print("Predicted text: (untrained model)")
# print(pred_text)

# %%

out = model.forward(tokenized_texts)

# take the argmax of the prediction
pred = out[0].argmax(dim=-1)
# decode the prediction
pred_text = tokenizer.sequences_to_texts(pred)

print("Predicted text: (untrained model)")
print(pred_text)
