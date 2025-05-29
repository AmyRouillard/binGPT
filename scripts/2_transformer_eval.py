# %%

import os
from utils.tentmapdataset import TentDataset
from mingpt.encoderonly import EncoderOnlyTransformer
from mingpt.utils import CfgNode as CN
import json
import torch
from torch.utils.data import DataLoader
import numpy as np

# %%

wdir = "/home/amyrouillard/project-files/"  # "C:/Users/Amy/Desktop/Green_Git/binGPT/" #"/mnt/lustre/users/arouillard/project-files/"  #
model_dir = wdir + f"models/2025_05_29_09_29"
gpt_load_epoch = 0


# model_dir = wdir + "models/binary_2025_04_23_13_02"

if os.path.exists(os.path.join(model_dir, "config.json")):
    # read json file
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        configs = json.load(f)
else:

    raise ValueError("No config.json found in model_dir, using default configs.")

train_dataset = TentDataset(
    "train",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)
test_dataset = TentDataset(
    "test",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)
validation_dataset = TentDataset(
    "validation",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)


# check if config.json exist in model_dir, if not create it
if os.path.exists(os.path.join(model_dir, "model_config.json")):
    # read json file
    with open(os.path.join(model_dir, "model_config.json"), "r") as f:
        model_config_dict = json.load(f)
else:
    raise ValueError("No model_config.json found in model_dir, using default configs.")

model_config = CN(**model_config_dict)
model = EncoderOnlyTransformer(model_config)


print(f"Number of training samples: {len(train_dataset):.3e}")
print(f"Number of test samples: {len(test_dataset):.3e}")
print(f"Number of validation samples: {len(validation_dataset):.3e}")


# %%

# Load the model state dict
model_path = os.path.join(model_dir, f"model_{gpt_load_epoch}.pt")
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
else:
    raise FileNotFoundError(f"Model file {model_path} does not exist.")

# %%

model.eval()  # Set the model to evaluation mode
batch_size = 2**15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)
model.to(device)  # Move model to the appropriate device


train_loader = DataLoader(
    train_dataset + validation_dataset,
    shuffle=True,
    pin_memory=True,
    batch_size=batch_size,
)

test_loader = DataLoader(
    test_dataset,
    shuffle=True,
    pin_memory=True,
    batch_size=batch_size,
)

# %%


# evaluate model on accuracy
def evaluate_model(model, data_loader, save_path=None):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            y_pred = model.generate(x)
            y_pred = y_pred.view(y_pred.size(0), -1)  # Flatten the predictions

            tmp = (y == y_pred).all(1).cpu()

            correct += tmp.sum().item()
            total += tmp.size(0)

            if save_path is not None:
                # save inp, sol, and sol_candidate to npy arrays
                np.save(
                    os.path.join(save_path, f"batch_{i}_inp.npy"),
                    x.cpu().numpy(),
                )
                np.save(
                    os.path.join(save_path, f"batch_{i}_sol.npy"),
                    y.cpu().numpy(),
                )
                np.save(
                    os.path.join(save_path, f"batch_{i}_pred.npy"),
                    y_pred.cpu().numpy(),
                )

    accuracy = correct / total if total > 0 else 0
    return accuracy


# %%

eval_dir = model_dir + f"eval_{gpt_load_epoch}/"
# Evaluate on training + validation dataset
if not os.path.exists(eval_dir + "train_val/"):
    os.makedirs(eval_dir + "train_val/")
train_val_accuracy = evaluate_model(
    model, train_loader, save_path=eval_dir + "train_val/"
)
print(f"Train + Validation Accuracy: {train_val_accuracy:.4f}")
# Evaluate on test dataset

if not os.path.exists(eval_dir + "test/"):
    os.makedirs(eval_dir + "test/")
test_accuracy = evaluate_model(model, test_loader, save_path=eval_dir + "test/")
print(f"Test Accuracy: {test_accuracy:.4f}")


results = {
    "train_val_accuracy": train_val_accuracy,
    "test_accuracy": test_accuracy,
}

results_path = os.path.join(eval_dir, "model_eval_sresults.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

# %%
