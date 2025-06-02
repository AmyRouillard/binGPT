# %%
# %%

from utils.tentmapdataset import ProbeDataset, ProbeDatasetMod
from mingpt.model import Probe
from mingpt.encoderonly import EncoderOnlyTransformerForProbing

import torch.optim as optim
import os
import json

import time
from mingpt.trainer import Trainer
from torch.utils.data import DataLoader
from mingpt.utils import CfgNode as CN

import torch
import csv

# %%

num_workers = 0

wdir = "/home/amyrouillard/project-files/"  # "C:/Users/Amy/Desktop/Green_Git/binGPT/" #"/mnt/lustre/users/arouillard/project-files/"  #
model_dir = wdir + f"models/2025_05_27_13_41/"


if os.path.exists(os.path.join(model_dir, "config.json")):
    # read json file
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        configs = json.load(f)
else:
    raise ValueError("No config.json found in model_dir, using default configs.")

print("Using configs:", configs)
# %%

# train_probe = ProbeDatasetMod(
#     "train",
#     length=configs["length"],
#     n_iterations=configs["n"],
#     type="decimal",
#     in_test=configs["in_test"],
# )

train_probe = ProbeDataset(
    "train",
    length=configs["length"],
    n_iterations=configs["n"],
    type="decimal",
    in_test=configs["in_test"],
)

n_classes = train_probe.n_classes

print(f"Number of training samples: {len(train_probe):.3e}")
print(f"Number of classes: {n_classes}")


# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

batch_size = 2**6  # train_config.batch_size


train_loader = DataLoader(
    train_probe,
    sampler=torch.utils.data.RandomSampler(train_probe, replacement=False),
    shuffle=False,
    # pin_memory=True,
    batch_size=batch_size,
    num_workers=num_workers,
)

data_iter = iter(train_loader)
batch = next(data_iter)

batch = [t.to(device) for t in batch]
x, y = batch


# %%
