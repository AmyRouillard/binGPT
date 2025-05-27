# %%

from utils.tentmapdataset import ProbeDataset
from mingpt.model import GPTforProbing, Probe

import torch.nn as nn
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

wdir = "C:/Users/Amy/Desktop/Green_Git/binGPT/"
model_dir = wdir + f"models/2025_05_26_16_28/"
gpt_load_epoch = 0


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

model_config = CN(**model_config_dict)

# %%

train_probe = ProbeDataset(
    "train",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)
test_probe = ProbeDataset(
    "test",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)
val_probe = ProbeDataset(
    "validation",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)

n_classes = train_probe.n_classes

print(f"Number of training samples: {len(train_probe):.3e}")
print(f"Number of test samples: {len(test_probe):.3e}")
print(f"Number of classes: {n_classes}")

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

batch_size = 2**15  # train_config.batch_size


train_loader = DataLoader(
    train_probe + val_probe,
    shuffle=False,
    pin_memory=True,
    batch_size=batch_size,
)

test_loader = DataLoader(
    test_probe,
    shuffle=False,  # No need to shuffle validation data
    pin_memory=True,
    batch_size=batch_size,
)

best_epoch = {
    "random": {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    },
    "trained": {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    },
}

target_step = 1
steps = [i for i in range(-target_step, target_step + 1) if i != 0]
steps = torch.tensor(steps, device=device)

for probe_layer in range(model_config.n_layer + 1):
    for w in ["random"]:  # , "trained"]:

        print(f"Initialized: {w} Probe layer: {probe_layer}")
        model = GPTforProbing(model_config, probe_layer)

        if w == "random":
            # randomly initialize the weights of the model
            # model.apply(model._init_weights)
            model.load_state_dict(torch.load(os.path.join(model_dir, f"model_-1.pt")))
        else:
            model.load_state_dict(
                torch.load(os.path.join(model_dir, f"model_{gpt_load_epoch}.pt"))
            )
        model.eval()

        input_dim = model.transformer.wpe.weight.shape
        # multiply elements of input_dim
        input_dim = input_dim[0] * input_dim[1]

        probe = Probe(
            n_classes=n_classes,
            input_dim=input_dim,
        )

        probe_path = os.path.join(
            model_dir, f"probe_{w}_{probe_layer}_model_{best_epoch[w][probe_layer]}.pt"
        )
        if os.path.exists(probe_path):
            print(f"Loading probe from {probe_path}")
            probe.load_state_dict(torch.load(probe_path))
        else:
            raise FileNotFoundError(
                f"Probe file {probe_path} does not exist. Please train the probe first."
            )

        probe.to(device)
        model.to(device)
        probe.eval()
        model.eval()
        out_dir = model_dir + f"modified_{gpt_load_epoch}_{w}_{probe_layer}/"

        for i, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            # target modification is a random choice from steps of shape targets
            target_modification = torch.randint(
                low=0, high=len(steps), size=targets.shape
            )
            # check if targets == 0 or n_classes - 1
            target_modification[targets == 0] = torch.randint(
                low=0,
                high=len(steps) // 2,
                size=targets[targets == 0].shape,
            )
            target_modification[targets == n_classes - 1] = torch.randint(
                low=len(steps) // 2,
                high=len(steps),
                size=targets[targets == n_classes - 1].shape,
            )
            # modify targets
            targets_modified = targets + steps[target_modification]
            # check that targets are in range [0, n_classes - 1] else raise ValueError
            if not torch.all((targets >= 0) & (targets < n_classes)):
                raise ValueError(
                    f"Modified targets out of range: {targets[~((targets >= 0) & (targets < n_classes))]}"
                )

            x = model.forward_1of2(inputs)
            y_pred, _ = model.forward_2of2(x)

            # modify x so that probe predicts modified targets
            x_tmp = x.view(x.size(0), -1).clone()
            optimizer = optim.Adam(x_tmp, lr=3e-4)
            loss_fn = nn.CrossEntropyLoss()
            if _ in range(100):
                # TODO: implement a better way to modify x
                # TODO: early stopping if loss does not decrease

                # set the optimizer to zero grad
                optimizer.zero_grad()
                outputs, _ = probe.forward(x_tmp)
                loss = loss_fn(outputs, targets_modified)
                loss.backward()
                optimizer.step()
                # update x with the modified x_tmp

            y_pred_mod, _ = model.forward_2of2(x_tmp.view(x.size(0), *x.size()[1:]))

            # save target, predictions, modified target, and modified predictions as numpy arrays
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            np.save(
                os.path.join(out_dir, f"batch_{i}_target.npy"),
                targets.cpu().numpy(),
            )
            np.save(
                os.path.join(out_dir, f"batch_{i}_pred.npy"),
                y_pred.cpu().numpy(),
            )
            np.save(
                os.path.join(out_dir, f"batch_{i}_target_mod.npy"),
                targets_modified.cpu().numpy(),
            )
            np.save(
                os.path.join(out_dir, f"batch_{i}_pred_mod.npy"),
                y_pred_mod.cpu().numpy(),
            )


# %%
