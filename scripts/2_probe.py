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

# %%

wdir = "C:/Users/Amy/Desktop/Green_Git/binGPT/"
model_dir = wdir + f"models/2025_05_26_16_28/"
load_epoch = 0


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
    train_probe,
    sampler=torch.utils.data.RandomSampler(train_probe, replacement=False),
    shuffle=False,
    pin_memory=True,
    batch_size=batch_size,
)

val_loader = DataLoader(
    val_probe,
    shuffle=False,  # No need to shuffle validation data
    pin_memory=True,
    batch_size=batch_size,
)

best_val_loss = float("inf")
best_epoch = 0


for probe_layer in range(model_config.n_layer + 1):
    for w in ["random"]:  # , "trained"]:

        if not os.path.exists(
            os.path.join(model_dir, f"probe_{w}_{probe_layer}_training_log.csv")
        ):
            with open(
                os.path.join(model_dir, f"probe_{w}_{probe_layer}_training_log.csv"),
                "w",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "epoch_num",
                        "iter_dt (ms)",
                        "iter_num",
                        "train_loss",
                        "best_val_loss",
                    ]
                )

        print(f"Initialized: {w} Probe layer: {probe_layer}")
        model = GPTforProbing(model_config, probe_layer)

        if w == "random":
            # randomly initialize the weights of the model
            model.apply(model._init_weights)
        else:
            model.load_state_dict(
                torch.load(os.path.join(model_dir, f"model_{load_epoch}.pt"))
            )
        model.eval()

        input_dim = model.transformer.wpe.weight.shape
        # multiply elements of input_dim
        input_dim = input_dim[0] * input_dim[1]

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

        iter_num = 0
        iter_time = time.time()
        data_iter = iter(train_loader)
        epoch_num = 0
        while epoch_num < 10:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
                # print("loaded batch")
            except StopIteration:

                epoch_num += 1
                probe.eval()
                # --- Validation Check ---
                total_val_loss = 0.0
                total_val_samples = 0
                # Add other metrics if needed, e.g., correct_predictions = 0
                with torch.no_grad():  # Disable gradient calculations
                    for batch in val_loader:
                        batch = [t.to(device) for t in batch]
                        x, y = batch
                        x = model.forward_1of2(x)
                        _, loss = probe.forward(x.view(x.size(0), -1), y)

                        total_val_loss += loss.item() * x.size(
                            0
                        )  # Weighted by batch size
                        total_val_samples += x.size(0)
                        # Example for accuracy:
                        # _, predicted = torch.max(logits, 1)
                        # correct_predictions += (predicted == y).sum().item()

                probe.train()  # Set model back to training mode

                avg_val_loss = (
                    total_val_loss / total_val_samples
                    if total_val_samples > 0
                    else float("nan")
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch_num
                    # Save the model state
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            model_dir, f"probe_{w}_{probe_layer}_model_{epoch_num}.pt"
                        ),
                    )
                #     print(
                #         f"New best validation loss: {best_val_loss:.4f} at epoch {epoch_num}"
                #     )
                # else:
                #     print(f"Epoch {epoch_num} - Validation Loss: {avg_val_loss:.4f}")

                data_iter = iter(train_loader)
                batch = next(data_iter)

            # except Exception as e:
            #     print(f"Error fetching batch: {e}")
            #     # skip to next iteration
            #     continue

            batch = [t.to(device) for t in batch]
            x, y = batch

            x = model.forward_1of2(x)
            out, loss = probe.forward(x.view(x.size(0), -1), y)

            probe.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tnow = time.time()
            iter_dt = tnow - iter_time
            iter_time = tnow

            iter_num += 1

            if iter_num % 10 == 0:
                print(
                    f"Probe {w} Layer {probe_layer} - Epoch {epoch_num} - Iter {iter_num}/{len(train_loader)}: "
                    f"Loss: {loss.item():.4f}, Iter Time: {iter_dt * 1000:.2f} ms"
                )

                with open(
                    os.path.join(
                        model_dir, f"probe_{w}_{probe_layer}_training_log.csv"
                    ),
                    "a",
                    newline="",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch_num,
                            iter_dt * 1000,
                            iter_num,
                            loss.item(),
                            best_val_loss,
                        ]
                    )


# %%
