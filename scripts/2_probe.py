# %%

from utils.tentmapdataset import ProbeDataset
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

wdir = "/home/amyrouillard/project-files/"  # "C:/Users/Amy/Desktop/Green_Git/binGPT/" #"/mnt/lustre/users/arouillard/project-files/"  #
model_dir = wdir + f"models/2025_05_29_09_29/"
gpt_load_epoch = 50
num_workers = 8

# wdir = "/home/amyrouillard/project-files/"  # "C:/Users/Amy/Desktop/Green_Git/binGPT/" #"/mnt/lustre/users/arouillard/project-files/"  #
# model_dir = wdir + f"models/2025_05_27_13_41/"
# gpt_load_epoch = -1

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
# test_probe = ProbeDataset(
#     "test",
#     length=configs["length"],
#     n_iterations=configs["n"],
#     type=configs["data_type"],
#     in_test=configs["in_test"],
# )
val_probe = ProbeDataset(
    "validation",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)

n_classes = train_probe.n_classes

print(f"Number of training samples: {len(train_probe):.3e}")
print(f"Number of validation samples: {len(val_probe):.3e}")
print(f"Number of classes: {n_classes}")


# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

batch_size = 2**17  # train_config.batch_size


train_loader = DataLoader(
    train_probe,
    sampler=torch.utils.data.RandomSampler(train_probe, replacement=False),
    shuffle=False,
    pin_memory=True,
    batch_size=batch_size,
    num_workers=num_workers,
)

val_loader = DataLoader(
    val_probe,
    shuffle=False,  # No need to shuffle validation data
    pin_memory=True,
    batch_size=batch_size,
    num_workers=num_workers,
)


early_stopping_patience = 10


for probe_layer in range(model_config.n_layer + 1):
    for w in ["random", "trained"]:

        if not os.path.exists(
            os.path.join(model_dir, f"model_{gpt_load_epoch}_probe_{w}_{probe_layer}")
        ):
            os.makedirs(
                os.path.join(
                    model_dir, f"model_{gpt_load_epoch}_probe_{w}_{probe_layer}"
                )
            )

        if not os.path.exists(
            os.path.join(
                model_dir,
                f"model_{gpt_load_epoch}_probe_{w}_{probe_layer}/training_log.csv",
            )
        ):
            with open(
                os.path.join(
                    model_dir,
                    f"model_{gpt_load_epoch}_probe_{w}_{probe_layer}/training_log.csv",
                ),
                "w",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "epoch_num",
                        "iter_num",
                        "iter_dt (ms)",
                        "train_loss",
                        "current_val_loss",
                        "best_val_loss",
                        "avg_val_accuracy",
                    ]
                )

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

        model.to(device)
        probe.to(device)

        # set the optimizer
        optimizer = optim.Adam(probe.parameters(), lr=1e-2)

        # set the optimizer to zero grad
        optimizer.zero_grad()

        iter_num = 0
        iter_time = time.time()
        data_iter = iter(train_loader)
        epoch_num = 0
        stop_training_flag = False
        best_val_loss = float("inf")
        avg_val_loss = 0
        avg_val_accuracy = 0
        patience_counter = 0
        best_epoch = 0
        while epoch_num < 500:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
                # print("loaded batch")
            except StopIteration:

                probe.eval()
                # --- Validation Check ---
                total_val_loss = 0.0
                total_val_samples = 0
                accuracy = 0.0  # Initialize accuracy or other metrics
                # Add other metrics if needed, e.g., correct_predictions = 0
                with torch.no_grad():  # Disable gradient calculations
                    for batch in val_loader:
                        batch = [t.to(device) for t in batch]
                        x, y = batch
                        x = model.forward_1of2(x)
                        logits, vloss = probe.forward(x.view(x.size(0), -1), y)

                        total_val_loss += vloss.item() * x.size(
                            0
                        )  # Weighted by batch size
                        total_val_samples += x.size(0)
                        accuracy += (logits.argmax(dim=-1) == y.view(-1)).sum().item()
                        # Example for accuracy:
                        # _, predicted = torch.max(logits, 1)
                        # correct_predictions += (predicted == y).sum().item()

                probe.train()  # Set model back to training mode

                avg_val_loss = (
                    total_val_loss / total_val_samples
                    if total_val_samples > 0
                    else float("nan")
                )
                avg_val_accuracy = (
                    accuracy / total_val_samples
                    if total_val_samples > 0
                    else float("nan")
                )

                improved = False
                if avg_val_loss < best_val_loss:
                    improved = True
                    best_val_loss = avg_val_loss
                    best_epoch = epoch_num
                    # Save the model state
                    torch.save(
                        probe.state_dict(),
                        os.path.join(
                            model_dir,
                            f"model_{gpt_load_epoch}_probe_{w}_{probe_layer}/epoch_{epoch_num}.pt",
                        ),
                    )

                if improved and avg_val_accuracy < 1.0:
                    patience_counter = 0
                else:
                    patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        stop_training_flag = True

                print(
                    f"Probe {w} Layer {probe_layer}, Epoch {epoch_num}, "
                    f"Loss: {loss:.2e}, Best loss: {best_val_loss:.2e}, Val acc: {avg_val_accuracy:.6f}, "
                    f"Patience: {patience_counter}/{early_stopping_patience}"
                )

                with open(
                    os.path.join(
                        model_dir,
                        f"model_{gpt_load_epoch}_probe_{w}_{probe_layer}/training_log.csv",
                    ),
                    "a",
                    newline="",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch_num,
                            iter_num,
                            iter_dt * 1000,
                            loss.item(),
                            avg_val_loss,
                            best_val_loss,
                            avg_val_accuracy,
                        ]
                    )

                data_iter = iter(train_loader)
                batch = next(data_iter)
                iter_num = 0
                epoch_num += 1

            if stop_training_flag:
                print(
                    f"Stopping training for Probe {w} Layer {probe_layer} at epoch {epoch_num} due to early stopping."
                )
                break

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

            if iter_num % 10 == 0:
                with open(
                    os.path.join(
                        model_dir,
                        f"model_{gpt_load_epoch}_probe_{w}_{probe_layer}/training_log.csv",
                    ),
                    "a",
                    newline="",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch_num,
                            iter_num,
                            iter_dt * 1000,
                            loss.item(),
                            avg_val_loss,
                            best_val_loss,
                            avg_val_accuracy,
                        ]
                    )

            iter_num += 1


# %%
