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
gpt_load_epoch = 50
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

for name, param in model.named_parameters():

    print(name, param.shape)
    if "weight" in name:
        print("Weight mean:", param.mean().item())
        print("Weight std:", param.std().item())
    elif "bias" in name:
        print("Bias mean:", param.mean().item())
        print("Bias std:", param.std().item())
