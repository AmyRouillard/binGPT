# %%

import time
import os

# datetime
dt = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

data_type = "binary"  # "decimal"

wdir = "C:/Users/Amy/Desktop/Green_Git/binGPT/"
model_dir = wdirx + f"models/{data_type}_{dt}/"  #
# model_dir = wdir + "models/binary_2025_04_23_13_02"

# %%

from utils.tentmapdataset import TentDataset

# print an example instance of the dataset
n = 4
length = 22
train_dataset = TentDataset("train", length=length, n_iterations=n, type=data_type)
test_dataset = TentDataset("test", length=length, n_iterations=n, type=data_type)

x, y = train_dataset[0]

print("x:", x)
print("y:", y)

x, y = test_dataset[0]

print("x:", x)
print("y:", y)

# %%

print(train_dataset.map_idx[:10])
print(test_dataset.map_idx[:10])
# %%

from mingpt.model import GPT
from mingpt.utils import CfgNode as CN

import json

# check if config.json exist in model_dir, if not create it
if os.path.exists(os.path.join(model_dir, "config.json")):
    # read json file
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        model_config_dict = json.load(f)

    print(model_config_dict)
else:
    # create model_config_dict
    model_config_dict = {
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 2**5 * 4,
        "model_type": None,
        "vocab_size": train_dataset.get_vocab_size(),
        "block_size": train_dataset.get_block_size(),
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "resid_pdrop": 0.1,
    }
    # OTHELLO
    # model_config_dict = {
    #     "n_layer": 8,
    #     "n_head": 8,
    #     "n_embd": 2**6 * 8,
    #     "model_type": None,
    #     "vocab_size": train_dataset.get_vocab_size(),
    #     "block_size": train_dataset.get_block_size(),
    #     "embd_pdrop": 0.1,
    #     "attn_pdrop": 0.1,
    #     "resid_pdrop": 0.1,
    # }


model_config = CN(**model_config_dict)


model = GPT(model_config)

print(f"Number of training samples: {len(train_dataset):.3e}")
print(f"Number of test samples: {len(test_dataset):.3e}")


# %%

import torch
import torch_directml

# Initialize the DirectML device
device = torch_directml.device(torch_directml.default_device())
print(f"Using DirectML device: {device}")

# %%

# create a Trainer object
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 3e-4
train_config.batch_size = 64 * 4
train_config.max_iters = (len(train_dataset) / train_config.batch_size) * 2  # 6000
train_config.num_workers = 0  # os.cpu_count()
train_config.device = device

print(train_config)

trainer = Trainer(train_config, model, train_dataset)
# %%


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(
            f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
        )


trainer.set_callback("on_batch_end", batch_end_callback)

# %%

print("Number of iterations", train_config.max_iters)
print("Number of iterations per batch:", len(train_dataset) / train_config.batch_size)
print(
    "Number of epochs:",
    train_config.max_iters / (len(train_dataset) / train_config.batch_size),
)


# %%

import torch

# if os.path.join(model_dir, "model.pt") load, else train
if os.path.exists(os.path.join(model_dir, "model.pt")):
    print("Loading model from disk...")
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
else:
    print("Training model...")
    trainer.run()

# %%

print(model_dir)
# check is dir exist if not create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# save model to model_dir
if os.path.exists(os.path.join(model_dir, "model.pt")):
    print("Model exist will not be overwritten.")
else:
    print("Model saving...")
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

if os.path.exists(os.path.join(model_dir, "config.json")):
    print("Configs exist will not be overwritten.")
else:
    print("Configs saving...")
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config_dict, f, indent=4)

# %%

model.eval()


# %%

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def eval_split(model, split, max_batches, device):
    dataset = {"train": train_dataset, "test": test_dataset}[split]
    n = train_dataset.length
    results = []
    mistakes = []
    incorrect_preds = []
    correct_preds = []
    mistakes_printed_already = 0
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        num_workers=0,
        drop_last=False,
    )

    print(f"Num iter: {len(dataset)/train_config.batch_size:.1f}")
    for b, (x, y) in tqdm(enumerate(loader)):
        x = x.to(device)
        y = y.to(device)
        # isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # let the model sample the rest of the sequence
        cat = model.generate(
            inp, n, do_sample=False
        )  # using greedy argmax, not sampling
        sol_candidate = cat[:, n:]  # isolate the filled in sequence
        # compare the predicted sequence to the true sequence
        correct = (sol == sol_candidate).all(1).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if (
                not correct[i] and "".join(map(str, inp[i].tolist())) not in mistakes
            ):  # and mistakes_printed_already < 3  # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                mistakes.append("".join(map(str, inp[i].tolist())))
                # print(
                #     "GPT claims that %s -> %s but g.t. is %s"
                #     % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist())
                # )
                incorrect_preds.append(
                    (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist())
                )
            else:
                correct_preds.append(
                    (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist())
                )
        if max_batches is not None and b + 1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print(
        "%s final score: %d/%d = %.2f%% correct"
        % (split, rt.sum(), len(results), 100 * rt.mean())
    )
    return correct_preds, incorrect_preds


# %%

# let's run a random given sequence through the model as well
n = train_dataset.length  # naugy direct access shrug
inp, sol = train_dataset[3]
inp = inp[:n]
sol = sol[-n:]

inp = inp.unsqueeze(0).to(trainer.device)
sol = sol.unsqueeze(0).to(trainer.device)

assert inp[0].nelement() == n
with torch.no_grad():
    cat = model.generate(inp, n, do_sample=False)

sol_candidate = cat[:, n:]
print("input sequence  :", inp.tolist())
print("output:         ", sol.tolist())
print("predicted:      ", sol_candidate.tolist())
# print('gt sort         :', sol.tolist())
print("matches         :", bool((sol == sol_candidate).all()))

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

weight_dist = {}
fig, ax = plt.subplots(n_plots, 1, figsize=(10, n_plots * 2))
count = 0
for k, v in weights.items():
    if "weight" in k:
        ax_idx = count
        count += 1
        ax[ax_idx].hist(v.flatten().cpu(), bins=50, alpha=0.5)
        ax[ax_idx].set_title(k)
        ax[ax_idx].set_xlabel("Weight value")
        ax[ax_idx].set_ylabel("Frequency")

        mean = v.mean()
        std = v.std()
        weight_dist[k] = (mean, std)
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

# print("Weight distribution:")
# for k, v in weight_dist.items():
#     print(f"{k}: mean: {v[0]:.2f}, std: {v[1]:.2f}")

# %%

import numpy as np

eval_type = "test"
for gamma in [0.0, 0.1, 0.5]:
    print(gamma)

    dir_out = os.path.join(
        model_dir, f"{eval_type} incorrect_preds_{str(gamma).replace(".","_")}.npy"
    )

    if os.path.exists(dir_out):

        incorrect_preds = np.load(dir_out, allow_pickle=True)

        print(
            "%s final score: %d/%d = %.2f%% correct"
            % (
                eval_type,
                len(test_dataset) - len(incorrect_preds),
                len(test_dataset),
                100 * (len(test_dataset) - len(incorrect_preds)) / len(test_dataset),
            )
        )

    else:

        weights_new = weights.copy()
        fig, ax = plt.subplots(n_plots, 1, figsize=(10, n_plots * 2))
        count = 0
        for k, v in weights_new.items():
            if "weight" in k:
                ax_idx = count
                count += 1
                ax[ax_idx].hist(v.flatten().cpu(), bins=50, alpha=0.5)

                # give a small perturbation to the weights of the selected components
                v += torch.randn_like(v) * gamma * weight_dist[k][1]

                ax[ax_idx].hist(v.flatten().cpu(), bins=50, alpha=0.5, color="red")
                ax[ax_idx].set_title(k)
                ax[ax_idx].set_xlabel("Weight value")
                ax[ax_idx].set_ylabel("Frequency")

        plt.tight_layout()

        # set model weights
        model.load_state_dict(weights_new)

        with torch.no_grad():
            results = eval_split(
                model, eval_type, max_batches=None, device=trainer.device
            )
        correct_preds, incorrect_preds = results

        # save correct_preds
        # np.save(os.path.join(model_dir, f"{eval_type} correct_preds.npy"), correct_preds)
        np.save(
            dir_out,
            incorrect_preds,
        )

# %%
from utils.tentmapdataset import ProbeDataset

input_dim = weights["transformer.wpe.weight"].shape
print("Input dim:", input_dim)

train_probe = ProbeDataset("train", length=length, n_iterations=n)
test_probe = ProbeDataset("test", length=length, n_iterations=n)

n_classes = train_probe.n_classes

print(f"Number of training samples: {len(train_probe):.3e}")
print(f"Number of test samples: {len(test_probe):.3e}")
print(f"Number of classes: {n_classes}")


# %%

from mingpt.model import GPTforProbing, Probe
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


batch_size = 10000
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


for probe_layer in range(model_config.n_layer + 1):
    for w in ["random", "trained"]:

        if w == "random":
            # randomly initialize the weights of the model
            model.apply(model._init_weights)
        else:
            model.load_state_dict(weights)

        print(f"Initialized: {w} Probe layer: {probe_layer}")
        model = GPTforProbing(model_config, probe_layer)

        X_tilde = model.forward_1of2(X)
        X_test_tilde = model.forward_1of2(X_test)

        probe = Probe(
            n_classes=n_classes,
            input_dim=input_dim,
        )

        # set the model to training mode
        probe.train()

        # set the optimizer
        optimizer = optim.Adam(probe.parameters(), lr=3e-4)

        # set the loss to zero
        loss = 0
        # set the optimizer to zero grad
        optimizer.zero_grad()
        # set the loss function
        loss_fn = nn.CrossEntropyLoss()

        for epochs in range(5000):

            # forward pass
            out, loss = probe.forward(X_tilde, Y)
            _, val_loss = probe.forward(X_test_tilde, Y_test)

            # backward pass
            loss.backward()

            # update the weights
            optimizer.step()

            if epochs % 500 == 0:
                print(
                    f"\tEpoch {epochs}, Training loss: {loss.item():.2e}, Validation loss: {val_loss.item():.2e}",
                )

        print(
            f"\tEpoch {epochs}, Training loss: {loss.item():.2e}, Validation loss: {val_loss.item():.2e}",
        )


# %%
