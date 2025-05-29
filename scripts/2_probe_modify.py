# %%

from utils.tentmapdataset import ProbeDatasetMod
from mingpt.model import Probe
from mingpt.encoderonly import EncoderOnlyTransformerForProbing

from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
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

model_config = CN(**model_config_dict)

# %%
target_step = 1
train_probe = ProbeDatasetMod(
    "train",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
    target_step=target_step,
)
test_probe = ProbeDatasetMod(
    "test",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
    target_step=target_step,
)
val_probe = ProbeDatasetMod(
    "validation",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
    target_step=target_step,
)

n_classes = train_probe.n_classes

print(f"Number of training samples: {len(train_probe):.3e}")
print(f"Number of test samples: {len(test_probe):.3e}")
print(f"Number of classes: {n_classes}")

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

batch_size = 2**17  # train_config.batch_size


train_loader = DataLoader(
    train_probe + val_probe,
    shuffle=True,
    pin_memory=True,
    batch_size=batch_size,
    num_workers=num_workers,
)

test_loader = DataLoader(
    test_probe,
    shuffle=True,  # No need to shuffle validation data
    pin_memory=True,
    batch_size=batch_size,
    num_workers=num_workers,
)

best_epoch = {
    "random": {
        0: 0,
        # 1: 0,
        # 2: 0,
        # 3: 0,
        # 4: 0,
    },
    "trained": {
        0: 0,
        # 1: 0,
        # 2: 0,
        # 3: 0,
        # 4: 0,
    },
}

for probe_layer in range(model_config.n_layer + 1):
    for w in ["random", "trained"]:

        print(f"Initialized: {w} Probe layer: {probe_layer}")
        model = EncoderOnlyTransformerForProbing(model_config, probe_layer)

        if w == "random":
            # randomly initialize the weights of the model
            # model.apply(model._init_weights)
            model.load_state_dict(torch.load(os.path.join(model_dir, f"model_-1.pt")))
        else:
            model.load_state_dict(
                torch.load(os.path.join(model_dir, f"model_{gpt_load_epoch}.pt"))
            )

        input_dim = model.transformer.wpe.weight.shape
        # multiply elements of input_dim
        input_dim = input_dim[0] * input_dim[1]

        probe = Probe(
            n_classes=n_classes,
            input_dim=input_dim,
        )

        probe_path = os.path.join(
            model_dir,
            f"model_{gpt_load_epoch}_probe_{w}_{probe_layer}/epoch_{best_epoch[w][probe_layer]}.pt",
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

        for i, batch in enumerate(test_loader):
            inputs, targets, targets_mod, true_out, true_out_mod = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            x = model.forward_1of2(inputs)
            logits = model.forward_2of2(x)
            probs = F.softmax(logits, dim=-1)
            y_pred = torch.argmax(probs, dim=-1)

            # modify x so that probe predicts modified targets
            x_tmp = x.view(x.size(0), -1).clone()
            # convert x_tmp to something that can be optimized
            x_tmp = torch.nn.Parameter(x_tmp, requires_grad=True)

            optimizer = optim.Adam([x_tmp], lr=3e-3)
            for _ in range(100):
                # TODO: implement a better way to modify x
                # TODO: early stopping if loss does not decrease

                # set the optimizer to zero grad
                optimizer.zero_grad()
                outputs, _ = probe.forward(x_tmp)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), targets_mod.view(-1)
                )
                loss.backward()
                optimizer.step()
                # update x with the modified x_tmp

            logits = model.forward_2of2(x_tmp.view(x.size(0), *x.size()[1:]))
            probs_mod = F.softmax(logits, dim=-1)
            y_pred_mod = torch.argmax(probs_mod, dim=-1)

            # save target, predictions, modified target, and modified predictions as numpy arrays
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            np.save(
                os.path.join(out_dir, f"batch_{i}_target_idx.npy"),
                targets.cpu().numpy(),
            )
            np.save(
                os.path.join(out_dir, f"batch_{i}_target_idx_mod.npy"),
                targets_mod.cpu().numpy(),
            )
            np.save(
                os.path.join(out_dir, f"batch_{i}_true_pred.npy"),
                true_out.cpu().numpy(),
            )
            np.save(
                os.path.join(out_dir, f"batch_{i}_pred.npy"),
                y_pred.cpu().numpy(),
            )
            np.save(
                os.path.join(out_dir, f"batch_{i}_pred_mod.npy"),
                y_pred_mod.cpu().numpy(),
            )
            np.save(
                os.path.join(out_dir, f"batch_{i}_true_pred_mod.npy"),
                true_out_mod.cpu().numpy(),
            )
            # save x
            np.save(
                os.path.join(out_dir, f"batch_{i}_intermediated.npy"),
                x.cpu().numpy(),
            )

            acc = (y_pred_mod.view(y_pred_mod.size(0), -1) == true_out_mod).all(
                1
            ).cpu().sum().item() / targets.size(0)
            print(f"Batch {i}: Accuracy of modified predictions: {acc:.4f}")


# %%
