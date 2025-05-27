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


# helper function evaluate the prob on accuracy
def evaluate_probe(model, probe, data_loader, device, save_path=None):
    probe.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            x = model.forward_1of2(inputs)
            outputs, _ = probe.forward(x.view(x.size(0), -1))
            _, predicted = torch.max(outputs, 1)

            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            if save_path is not None:
                np.save(
                    os.path.join(save_path, f"batch_{i}_sol.npy"),
                    targets.cpu().numpy(),
                )
                np.save(
                    os.path.join(save_path, f"batch_{i}_pred.npy"),
                    predicted.cpu().numpy(),
                )

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy


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

        eval_dir = model_dir + f"eval_{gpt_load_epoch}_{w}_{probe_layer}/"

        # Evaluate the probe on the test set
        if not os.path.exists(eval_dir + "train_val/"):
            os.makedirs(eval_dir + "train_val/")
        test_accuracy = evaluate_probe(
            model, probe, test_loader, device, eval_dir + "train_val/"
        )
        print(f"Test accuracy for {w} probe layer {probe_layer}: {test_accuracy:.4f}")
        # Evaluate the probe on the training set

        if not os.path.exists(eval_dir + "train/"):
            os.makedirs(eval_dir + "train/")
        train_accuracy = evaluate_probe(
            model, probe, train_loader, device, eval_dir + "train/"
        )
        print(f"Train accuracy for {w} probe layer {probe_layer}: {train_accuracy:.4f}")

        results = {
            "test_accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
        }
        with open(os.path.join(eval_dir, f"results.json"), "w") as f:
            json.dump(results, f, indent=4)


# %%
