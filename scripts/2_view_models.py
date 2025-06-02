# %%

from utils.tentmapdataset import ProbeDataset
from mingpt.model import Probe
from mingpt.encoderonly import EncoderOnlyTransformerForProbing, EncoderOnlyTransformer

from torch.nn import functional as F
import torch.optim as optim
import os
import json

import time
from mingpt.trainer import Trainer
from torch.utils.data import DataLoader
from mingpt.utils import CfgNode as CN

import torch
import csv
import numpy as np

# %%

wdir = "/home/amyrouillard/project-files/"  # "C:/Users/Amy/Desktop/Green_Git/binGPT/" #"/mnt/lustre/users/arouillard/project-files/"  #
model_dir = wdir + f"models/2025_05_29_09_29/"
transformer_load_epoch = 50
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
model.eval()
# %%
for k in model_config_dict.keys():
    print(f"{k}: {model_config_dict[k]}")
# %%
import matplotlib.pyplot as plt

n_figs = 1 + model_config_dict["n_layer"] * 6 + 2
fig, ax = plt.subplots(n_figs, 2, figsize=(10, 2 * n_figs), squeeze=False)
i = 1
for name, param in model.named_parameters():

    print(name, param.numel())
    if "wte.weight" in name:
        ax[0, 0].imshow(param.detach().cpu().numpy(), aspect="auto")
        ax[0, 0].set_title(
            f"wte.weight {param.shape}\n {param.detach().cpu().numpy().min():.2e} {param.detach().cpu().numpy().max():.2e}"
        )
    elif "wpe.weight" in name:
        ax[0, 1].imshow(param.detach().cpu().numpy(), aspect="auto")
        ax[0, 1].set_title(
            f"wpe.weight {param.shape}\n {param.detach().cpu().numpy().min():.2e} {param.detach().cpu().numpy().max():.2e}"
        )
    elif "weight" in name:
        tmp = param.detach().cpu().numpy()
        if len(tmp.shape) == 1:
            tmp = tmp.reshape(1, -1)
        ax[i, 0].imshow(tmp, aspect="auto")
        ax[i, 0].set_title(
            f"{name} {param.shape}\n {param.detach().cpu().numpy().min():.2e} {param.detach().cpu().numpy().max():.2e}"
        )
    elif "bias" in name:
        ax[i, 1].imshow(param.detach().cpu().numpy()[None, :], aspect="auto")
        ax[i, 1].set_title(
            f"{name} {param.shape}\n {param.detach().cpu().numpy().min():.2e} {param.detach().cpu().numpy().max():.2e}"
        )
        i += 1

for i in range(n_figs):
    ax[i, 0].set_xticks([])
    ax[i, 1].set_xticks([])
    ax[i, 0].set_yticks([])
    ax[i, 1].set_yticks([])

fig.tight_layout()

plt.show()


# %%
