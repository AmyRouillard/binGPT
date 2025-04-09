# %%


# %%

from utils.tentmapdataset import TentDataset, ProbeDataset

# print an example instance of the dataset
n = 4
length = 22
train_dataset = TentDataset("train", length=length, n_iterations=n)  # , type="decimal"
test_dataset = TentDataset("test", length=length, n_iterations=n)  # , type="decimal"

x, y = train_dataset[0]

print("x:", x)
print("y:", y)

x, y = test_dataset[0]

print("x:", x)
print("y:", y)


# %%

# create a GPT instance
from mingpt.model import GPTforProbing
from mingpt.utils import CfgNode as CN

model_config = CN(
    n_layer=3,
    n_head=3,
    n_embd=2**4 * 3,
    model_type=None,
    vocab_size=train_dataset.get_vocab_size(),
    block_size=train_dataset.get_block_size(),
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
)

probe_layer = 1
model = GPTforProbing(model_config, probe_layer)

print(f"Number of training samples: {len(train_dataset):.3e}")
print(f"Number of test samples: {len(test_dataset):.3e}")


# %%
import torch
import os

model_dir = "C:/Users/Amy/Desktop/Green_Git/binGPT/models/binary"

print("Loading model from disk...")
model.load_state_dict(
    torch.load(
        os.path.join(
            model_dir,
            "model.pt",
        ),
        map_location=torch.device("cpu"),
    )
)

# %%

# add a random perturbation to the weights of the model
# for k, v in model.state_dict().items():
#     if "weight" in k:
#         v += torch.randn_like(v) * 0.01
#         print(k, v.mean(), v.std())

# %%

# get the weights of the model
weights = model.state_dict()
print("Weights loaded from disk.")
print("Number of weights:", len(weights))
print("Weights:", weights.keys())
print("Weights shapes:")
for k, v in weights.items():
    print(f"{k}: {v.shape}")

# %%

# plot the distribution of weights in each layer
import matplotlib.pyplot as plt

n_plots = 0

for k, v in weights.items():
    if "weight" in k:
        n_plots += 1

fig, ax = plt.subplots(n_plots, 1, figsize=(10, n_plots * 2))
count = 0
for k, v in weights.items():
    if "weight" in k:
        ax_idx = count
        count += 1
        ax[ax_idx].hist(v.flatten(), bins=50, alpha=0.5)
        ax[ax_idx].set_title(k)
        ax[ax_idx].set_xlabel("Weight value")
        ax[ax_idx].set_ylabel("Frequency")

        mean = v.mean()
        std = v.std()
        # add mean and std to the plot in text
        ax[ax_idx].text(
            0.5,
            0.9,
            f"mean: {mean:.2f}\nstd: {std:.2f}",
            transform=ax[ax_idx].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
plt.tight_layout()


# %%

input_dim = weights["transformer.wpe.weight"].shape
print("Input dim:", input_dim)

# %%

# take a reandom sample of training data
import random

batch_size = 10
indices = random.sample(range(len(train_dataset)), batch_size)

# batch_old = []
# for i in indices:
#     x, y = train_dataset[i]
#     batch_old.append(x)
# batch_old = torch.stack(batch_old)
# print("Batch shape:", batch_old.shape)
# result_old = (batch_old[:, n] == 1).unsqueeze(1).long()
# print("Result shape:", result_old.shape)

# flip or no-flip: check if the nth bit of batch is 1 or 0
train_probe = ProbeDataset("train", length=length, n_iterations=n)
batch = []
result = []
for i in indices:
    x, y = train_probe[i]
    batch.append(x)
    result.append(y)
batch = torch.stack(batch)
print("Batch shape:", batch.shape)
result = torch.tensor(result).unsqueeze(1)
print("Result shape:", result.shape)
# %%

import torch.nn as nn
import torch.nn.functional as F


class Probe(nn.Module):
    def __init__(
        self,
        n_classes,
        input_dim,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.shape = input_dim
        # assume the input dim is torch.Size([batch_size,...])
        self.input_dim = int(torch.prod(torch.tensor(self.shape)))

        self.W = nn.Linear(
            self.input_dim,
            self.n_classes,
            bias=True,
        )

    def forward(self, x, targets=None):
        x = x.view(x.size(0), -1)  # flatten the input

        x = self.W(x)  # apply the linear layer

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)), targets.view(-1), ignore_index=-1
            )

        return x, loss


# %%

probe = Probe(
    n_classes=2,
    input_dim=input_dim,
)

print("Probe shape:", probe.shape)
print("Probe input dim:", probe.input_dim)
print("Probe weights shape:", probe.W.weight.shape)

# %%

out = model.forward_1of2(batch)
out, loss = probe.forward(out, result)

print(out, loss)

# %%


batch_size = 4000

train_probe = ProbeDataset("train", length=length, n_iterations=n)
test_probe = ProbeDataset("test", length=length, n_iterations=n)
print(f"Number of training samples: {len(train_probe):.3e}")
print(f"Number of test samples: {len(test_probe):.3e}")

indices = random.sample(range(len(train_probe)), batch_size)
X = []
Y = []
for i in indices:
    x, y = train_probe[i]
    X.append(x)
    Y.append(y)
X = torch.stack(X)
Y = torch.tensor(Y).unsqueeze(1)


indices = random.sample(range(len(train_probe)), 1000)
X_test = []
Y_test = []
for i in indices:
    x, y = train_probe[i]
    X_test.append(x)
    Y_test.append(y)
X_test = torch.stack(X_test)
Y_test = torch.tensor(Y_test).unsqueeze(1)

# %%

# train probe
import torch.optim as optim

for probe_layer in [3]:  # [0, 1, 2, 3]:
    model = GPTforProbing(model_config, probe_layer)

    # randomly initialize the weights of the model
    model.apply(model._init_weights)
    # model.load_state_dict(
    #     torch.load(
    #         os.path.join(
    #             model_dir,
    #             "model.pt",
    #         ),
    #         map_location=torch.device("cpu"),
    #     )
    # )

    probe = Probe(
        n_classes=2,
        input_dim=input_dim,
    )

    # set the model to training mode
    probe.train()

    # set the optimizer
    optimizer = optim.Adam(probe.parameters(), lr=1e-2)

    # set the loss to zero
    loss = 0
    # set the optimizer to zero grad
    optimizer.zero_grad()
    # set the loss function
    loss_fn = nn.CrossEntropyLoss()

    for epochs in range(500):

        # forward pass
        out, loss = probe.forward(model.forward_1of2(X), Y)
        _, val_loss = probe.forward(model.forward_1of2(X_test), Y_test)

        # backward pass
        loss.backward()

        # update the weights
        optimizer.step()

        if epochs % 100 == 0:
            print(f"Epoch {epochs}")
            print("Training loss:", loss.item())
            print("Validation loss:", val_loss.item())

    print(f"Epoch {epochs}")
    print("Training loss:", loss.item())
    print("Validation loss:", val_loss.item())
# %%

# 0
# Training loss: 0.05622125789523125
# Validation loss: 0.05804743990302086
# Random:
# Training loss: 0.05796302482485771
# Validation loss: 0.057671695947647095

# 1
# Training loss: 0.028845416381955147
# Validation loss: 0.029208114370703697
# Random:
# Training loss: 0.035739973187446594
# Validation loss: 0.03721821680665016

# 2
# Training loss: 0.0028942893259227276
# Validation loss: 0.0031556703615933657
# Random:
# Training loss: 0.04501847177743912
# Validation loss: 0.049120549112558365

# 3
# Training loss: 0.005391385406255722
# Validation loss: 0.005553032737225294
# Random:
# Training loss: 0.03059990331530571
# Validation loss: 0.028499478474259377

# %%


probe = Probe(
    n_classes=2,
    input_dim=X.shape[1:],
)

# set the model to training mode
probe.train()

# set the optimizer
optimizer = optim.Adam(probe.parameters(), lr=1e-2)

# set the loss to zero
loss = 0

# set the optimizer to zero grad
optimizer.zero_grad()

# set the loss function
loss_fn = nn.CrossEntropyLoss()

for epochs in range(500):

    # forward pass
    out, loss = probe.forward(X.float(), Y)
    _, val_loss = probe.forward(X_test.float(), Y_test)

    # backward pass
    loss.backward()

    # update the weights
    optimizer.step()

    if epochs % 100 == 0:
        print(f"Epoch {epochs}")
        print("Training loss:", loss.item())
        print("Validation loss:", val_loss.item())

print(f"Epoch {epochs}")
print("Training loss:", loss.item())
print("Validation loss:", val_loss.item())

# %%
