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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2**18  # train_config.batch_size

best_epoch = {
    "random": {
        0: 135,
        1: 130,
        2: 129,
    },
    "trained": {
        0: 100,
        1: 57,
        2: 23,
    },
}

# %%

# create log.csv file
log_file = os.path.join(model_dir, f"modified_model_log.csv")
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "target_step",
            "trained_model",
            "gpt_load_epoch",
            "probe_layer",
            "load_epoch",
            "set",  # train+validation, test,
            "batch",
            "accuracy",
            "accuracy_0",
            "accuracy_1",
            "num_false",
            "p_false",
        ]
    )


for target_step in [
    # -8,
    # -7,
    # -6,
    # -5,
    # -4,
    # -3,
    # -2,
    -1,
    1,
    # 2,
    # 3,
    # 4,
    # 5,
    # 6,
    # 7,
    # 8,
]:

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

    n_classes = train_probe.n_classes
    for probe_layer in range(model_config.n_layer + 1):
        for w in ["trained", "random"]:

            print(f"Initialized: {w} Probe layer: {probe_layer}")
            model = EncoderOnlyTransformerForProbing(model_config, probe_layer)

            if w == "random":
                # randomly initialize the weights of the model
                # model.apply(model._init_weights)
                model.load_state_dict(
                    torch.load(os.path.join(model_dir, f"model_-1.pt"))
                )
            else:
                model.load_state_dict(
                    torch.load(os.path.join(model_dir, f"model_{gpt_load_epoch}.pt"))
                )
            model.to(device)
            model.eval()

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
                # print(f"Loading probe from {probe_path}")
                probe.load_state_dict(torch.load(probe_path))
            else:
                raise FileNotFoundError(
                    f"Probe file {probe_path} does not exist. Please train the probe first."
                )

            probe.to(device)
            probe.eval()

            out_dir = (
                model_dir
                + f"modified_model_{gpt_load_epoch}_probe_{w}_{probe_layer}_{best_epoch[w][probe_layer]}/step_{int(target_step)}/"
            )
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            for loader_name, loader in zip(
                ["train_val", "test"], [train_loader, test_loader]
            ):
                for i, batch in enumerate(loader):
                    inputs, targets, targets_mod, true_out, true_out_mod = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    targets_mod = targets_mod.to(device)
                    true_out_mod = true_out_mod.to(device)

                    x = model.forward_1of2(inputs)
                    logits = model.forward_2of2(x)
                    probs = F.softmax(logits, dim=-1)
                    y_pred = torch.argmax(probs, dim=-1)

                    # modify x so that probe predicts modified targets
                    x_tmp = x.view(x.size(0), -1).clone()
                    # convert x_tmp to something that can be optimized
                    x_tmp = torch.nn.Parameter(x_tmp, requires_grad=True)

                    optimizer = optim.Adam([x_tmp], lr=3e-3)
                    patience = 0
                    max_patience = 1
                    flag = False
                    for itt in range(100):
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

                        probs = F.softmax(outputs, dim=-1)
                        _, predicted = torch.max(probs, dim=-1)

                        total_ = (predicted == targets_mod.view(-1)).sum().item()

                        # print(
                        #     f"Batch {i}, Iteration {itt}: Loss: {loss.item():.2e}, "
                        #     f"Accuracy: {total_ / targets.size(0):.2e} ({total_}/{targets.size(0)})"
                        # )
                        if total_ == targets.size(0):

                            patience += 1

                        if patience >= max_patience:
                            print(
                                f"Batch {i}: All targets modified successfully after {itt} iterations."
                            )
                            flag = True
                            break

                        # update x with the modified x_tmp

                    if not flag:
                        print(
                            f"Batch {i}: Failed to modify all targets after {itt} iterations."
                        )

                    logits = model.forward_2of2(x_tmp.view(x.size(0), *x.size()[1:]))
                    probs_mod = F.softmax(logits, dim=-1)
                    y_pred_mod = torch.argmax(probs_mod, dim=-1)

                    acc = (y_pred_mod.view(y_pred_mod.size(0), -1) == true_out_mod).all(
                        1
                    ).cpu().sum().item() / targets.size(0)

                    # find indices where inputs[:,configs["n"]]==0
                    mask = inputs[:, configs["n"] - 1] == 0
                    acc_0 = (
                        y_pred_mod.view(y_pred_mod.size(0), -1)[mask]
                        == true_out_mod[mask]
                    ).all(1).cpu().sum().item() / mask.sum().item()
                    acc_1 = (
                        y_pred_mod.view(y_pred_mod.size(0), -1)[~mask]
                        == true_out_mod[~mask]
                    ).all(1).cpu().sum().item() / (~mask).sum().item()

                    # acc = (y_pred_mod.view(y_pred_mod.size(0), -1) == true_out_mod)[:, :5].all(
                    #     1
                    # ).cpu().sum().item() / targets.size(0)
                    # print(f"Batch {i}: 5: {acc:.4f}")

                    # acc = (y_pred_mod.view(y_pred_mod.size(0), -1) == true_out_mod)[:, :10].all(
                    #     1
                    # ).cpu().sum().item() / targets.size(0)
                    # print(f"Batch {i}: 10: {acc:.4f}")

                    # acc = (y_pred_mod.view(y_pred_mod.size(0), -1) == true_out_mod)[:, :15].all(
                    #     1
                    # ).cpu().sum().item() / targets.size(0)
                    # print(f"Batch {i}: 15: {acc:.4f}")

                    # acc = (y_pred_mod.view(y_pred_mod.size(0), -1) == true_out_mod)[:, :20].all(
                    #     1
                    # ).cpu().sum().item() / targets.size(0)
                    # print(f"Batch {i}: 20: {acc:.4f}")

                    # find the number of indices where y_pred_mod.view(y_pred_mod.size(0), -1) == true_out_mod is false
                    num_false = (
                        (y_pred_mod.view(y_pred_mod.size(0), -1) != true_out_mod)
                        .sum()
                        .item()
                    )
                    p_false = num_false / targets.size(0) / true_out_mod.size(-1)

                    print(
                        f"Batch {i}: Accuracy: {acc:.4f} ({acc_0:.4f};{acc_1:.4f}) #false predictions: {num_false:.2e}/{targets.size(0)*true_out_mod.size(-1):.2e} ({p_false:.4f})"
                    )

                    print(
                        y_pred_mod.shape,
                        y_pred.shape,
                        y_pred_mod[mask].shape,
                        y_pred[mask].shape,
                    )
                    N_unchanged0 = (
                        (y_pred_mod[mask] == y_pred[mask]).all(1).cpu().sum().item()
                    )
                    N_unchanged1 = (
                        (y_pred_mod[~mask] == y_pred[~mask]).all(1).cpu().sum().item()
                    )
                    print(
                        f"Changed: {targets.size(0)-N_unchanged0}/{targets.size(0)} ({(targets.size(0)-N_unchanged0)/targets.size(0):.4f})"
                    )
                    print(
                        f"Changed: {targets.size(0)-N_unchanged1}/{targets.size(0)} ({(targets.size(0)-N_unchanged1)/targets.size(0):.4f})"
                    )

                    # write to log.csv
                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                target_step,
                                True if w == "trained" else False,
                                gpt_load_epoch if w == "trained" else -1,
                                probe_layer,
                                best_epoch[w][probe_layer],
                                loader_name,
                                i,
                                f"{acc:.4f}",
                                f"{acc_0:.4f}",
                                f"{acc_1:.4f}",
                                f"{num_false:.2e}",
                                f"{p_false:.4f}",
                            ]
                        )

                    # if w == "trained":
                    #     n_sample = 1
                    #     mask_incorrect = (
                    #         y_pred_mod.view(y_pred_mod.size(0), -1)[mask]
                    #         == true_out_mod[mask]
                    #     ).all(1) == False
                    #     N = y_pred_mod[mask][mask_incorrect].size(0)
                    #     if N < n_sample:
                    #         n_sample = N
                    #     idxs = torch.randperm(N)[:n_sample]
                    #     # print y_pred_mod[idx]
                    #     print(
                    #         f"No flip q={configs["n"]} mod={target_step} #incorrect={N}:"
                    #     )
                    #     for idx in idxs:
                    #         print(
                    #             f"{targets[mask][mask_incorrect][idx]}->{targets_mod[mask][mask_incorrect][idx]}"
                    #             f"\n{inputs[mask][mask_incorrect][idx]}"
                    #             f"\n{y_pred[mask][mask_incorrect][idx]}"
                    #             f"\n{y_pred_mod[mask][mask_incorrect][idx]}"
                    #             f"\n{true_out_mod[mask][mask_incorrect][idx]}\n"
                    #         )

                    #     n_sample = 2
                    #     mask_incorrect = (
                    #         y_pred_mod.view(y_pred_mod.size(0), -1)[~mask]
                    #         == true_out_mod[~mask]
                    #     ).all(1) == False
                    #     N = y_pred_mod[~mask][mask_incorrect].size(0)
                    #     if N < n_sample:
                    #         n_sample = N
                    #     idxs = torch.randperm(N)[:n_sample]
                    #     # print y_pred_mod[idx]
                    #     print(
                    #         f"Flip q={configs["n"]} mod={target_step} #incorrect={N}:"
                    #     )
                    #     for idx in idxs:
                    #         print(
                    #             f"{targets[~mask][mask_incorrect][idx]}->{targets_mod[~mask][mask_incorrect][idx]}"
                    #             f"\n{inputs[~mask][mask_incorrect][idx]}"
                    #             f"\n{y_pred[~mask][mask_incorrect][idx]}"
                    #             f"\n{y_pred_mod[~mask][mask_incorrect][idx]}"
                    #             f"\n{true_out_mod[~mask][mask_incorrect][idx]}\n"
                    #         )

                    # save target, predictions, modified target, and modified predictions as numpy arrays

                    # np.save(
                    #     os.path.join(
                    #         out_dir, f"{loader_name}_batch_{i}_target_idx.npy"
                    #     ),
                    #     targets.cpu().numpy(),
                    # )
                    # np.save(
                    #     os.path.join(
                    #         out_dir, f"{loader_name}_batch_{i}_target_idx_mod.npy"
                    #     ),
                    #     targets_mod.cpu().numpy(),
                    # )
                    # np.save(
                    #     os.path.join(out_dir, f"{loader_name}_batch_{i}_true_pred.npy"),
                    #     true_out.cpu().numpy(),
                    # )
                    # np.save(
                    #     os.path.join(out_dir, f"{loader_name}_batch_{i}_pred.npy"),
                    #     y_pred.cpu().numpy(),
                    # )
                    # np.save(
                    #     os.path.join(out_dir, f"{loader_name}_batch_{i}_pred_mod.npy"),
                    #     y_pred_mod.cpu().numpy(),
                    # )
                    # np.save(
                    #     os.path.join(
                    #         out_dir, f"{loader_name}_batch_{i}_true_pred_mod.npy"
                    #     ),
                    #     true_out_mod.cpu().numpy(),
                    # )
                    # np.save(
                    #     os.path.join(out_dir, f"batch_{i}_intermediated.npy"),
                    #     x.cpu().numpy(),
                    # )
                    # np.save(
                    #     os.path.join(out_dir, f"batch_{i}_intermediated_mod.npy"),
                    #     x_tmp.view(x.size(0), *x.size()[1:]).cpu().detach().numpy(),
                    # )
                    break

# %%
