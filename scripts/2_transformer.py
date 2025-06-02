# %%

import time
import os
import numpy as np
from utils.tentmapdataset import TentDataset
from mingpt.encoderonly import EncoderOnlyTransformer
from mingpt.utils import CfgNode as CN
import json
from mingpt.trainer import Trainer
import torch
import csv

# %%


# datetime
dt = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

wdir = "/home/amyrouillard/project-files/"  # "C:/Users/Amy/Desktop/Green_Git/binGPT/" #"/mnt/lustre/users/arouillard/project-files/"  #
model_dir = wdir + f"models/{dt}/"  #
# model_dir = wdir + "models/binary_2025_04_23_13_02"

if os.path.exists(os.path.join(model_dir, "config.json")):
    # read json file
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        configs = json.load(f)
else:

    configs = {
        # "data_type": "binary",
        "data_type": "decimal",
        "n": 3,
        "length": 23,
    }

    n = 4
    # 0-train, 1-test, 2-validation
    in_test = (
        list("1" * (2 ** (configs["length"] - n - 1)))
        + list("2" * (2 ** (configs["length"] - n - 1)))
        + list("0" * (2 ** (configs["length"] - n) * (2**n - 1)))
    )

    # shuffle the in_test list with a fixed seed
    rng = np.random.default_rng(42)
    in_test = rng.permutation(in_test).tolist()

    configs["in_test"] = in_test

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(configs, f, indent=4)

train_dataset = TentDataset(
    "train",
    length=configs["length"],
    n_iterations=configs["n"],
    type=configs["data_type"],
    in_test=configs["in_test"],
)
# test_dataset = TentDataset(
#     "test",
#     length=configs["length"],
#     n_iterations=configs["n"],
#     type=configs["data_type"],
#     in_test=configs["in_test"],
# )
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

    # create model_config_dict
    model_config_dict = {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 2**4 * 2,
        "model_type": None,
        "vocab_size": train_dataset.get_vocab_size(),
        "block_size": train_dataset.get_block_size(),
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "output_vocab_size": None,
        "pad_token_id": None,
    }
    # OTHELLO FOR COMPARISON
    # model_config_dict = {
    #     "n_layer": 8,
    #     "n_head": 8,
    #     "n_embd": 2**6 * 8,
    #     "model_type": None,
    #     "vocab_size": train_dataset.get_vocab_size(),
    #     "block_size": train_dataset.get_block_size(),
    #     "embd_pdrop": 0.1,
    #     "attn_pdrop": 0.1,
    #     "resid_pdrop": 0.1,
    # }

    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config_dict, f, indent=4)


model_config = CN(**model_config_dict)
model = EncoderOnlyTransformer(model_config)


print(f"Number of training samples: {len(train_dataset):.3e}")
# print(f"Number of test samples: {len(test_dataset):.3e}")
print(f"Number of validation samples: {len(validation_dataset):.3e}")


# %%

if os.path.exists(os.path.join(model_dir, "trainer_config.json")):
    with open(os.path.join(model_dir, "trainer_config.json"), "r") as f:
        train_config_dict = json.load(f)

    train_config = Trainer.get_default_config()
    train_config.merge_from_dict(train_config_dict)

else:

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 3e-4
    train_config.batch_size = 2**14  # 2**15
    train_config.max_iters = (
        len(train_dataset) / train_config.batch_size
    ) * 500  # 6000
    train_config.num_workers = 8  # 0  # os.cpu_count()
    train_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config.model_dir = model_dir
    train_config.early_stopping_patience = 10

    train_config_dict = train_config.to_dict()

    # save the config to model_dir
    with open(os.path.join(model_dir, "trainer_config.json"), "w") as f:
        json.dump(train_config_dict, f, indent=4)


print(train_config)
trainer = Trainer(train_config, model, train_dataset, validation_dataset)

print("Number of iterations", train_config.max_iters)
print("Number of iterations per epoch:", len(train_dataset) / train_config.batch_size)
print(
    "Number of epochs:",
    train_config.max_iters / (len(train_dataset) / train_config.batch_size),
)

# %%

# create .csv file with the iter_dt (ms), iter_num, loss, current_metric_val, best_metric_val, patience_counter


if not os.path.exists(os.path.join(model_dir, "training_log.csv")):
    with open(os.path.join(model_dir, "training_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch_num",
                "iter_num",
                "iter_dt (ms)",
                "train_loss",
                "current_metric_val",
                "best_metric_val",
            ]
        )


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        # print(
        #     f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.4e}"
        # )
        with open(os.path.join(model_dir, "training_log.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    trainer.epoch_num,
                    trainer.iter_num,
                    trainer.iter_dt * 1000,
                    trainer.loss.item(),
                    trainer.current_metric_val,
                    trainer.best_metric_val,
                ]
            )


trainer.set_callback("on_batch_end", batch_end_callback)


# def epoch_end_callback(trainer):
#     torch.save(
#         model.state_dict(), os.path.join(model_dir, f"model_{trainer.epoch_num}.pt")
#     )


# trainer.set_callback("on_epoch_end", epoch_end_callback)

# %%

# find models matching trainer_config.model_dir + f"model_*.pt"
import glob

model_files = glob.glob(
    os.path.join(model_dir, "model_*.pt"),
)
if model_files:
    # load the latest model
    latest_model_file = max(model_files, key=os.path.getctime)
    print(f"Loading model from {latest_model_file}")
    trainer.model.load_state_dict(
        torch.load(latest_model_file, map_location=train_config.device)
    )

    trainer.epoch_num = int(
        int(latest_model_file.split("_")[-1].split(".")[0]) + 1
    )  # extract epoch number from filename
    print(f"Resuming training from epoch {trainer.epoch_num}")
else:
    print("No previous model found, starting training from scratch.")
    trainer.model.apply(trainer.model._init_weights)
    torch.save(
        trainer.model.state_dict(),
        os.path.join(model_dir, f"model_-1.pt"),
    )
# %%

trainer.run()

# %%
