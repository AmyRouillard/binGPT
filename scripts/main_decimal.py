# %%

from utils.tentmapdataset import TentDataset

# print an example instance of the dataset
n = 1
length = 16
train_dataset = TentDataset("train", length=length, n_iterations=n, type="decimal")
test_dataset = TentDataset("test", length=length, n_iterations=n, type="decimal")

x, y = train_dataset[5]

print("x:", x)
print("y:", y)

x, y = test_dataset[5]

print("x:", x)
print("y:", y)

# %%
# X, Y = [], []
# for i in range(train_dataset.__len__()):
#     x, y = train_dataset[i]
#     x = x.tolist()[:length]
#     y = y.tolist()[-length:]
#     # append the input and target sequences to the lists if they are not already in the lists
#     if x not in X:
#         X.append(x)
#     if y not in Y:
#         Y.append(y)

#     # if "".join(map(str, x)) == "0" * length:
#     #     print("x:", x)
#     #     print("y:", y)

#     # if "".join(map(str, x)) == "1" + "0" * (length - 1):
#     #     print("x:", x)
#     #     print("y:", y)

#     # if "".join(map(str, x)) == "1" * length:
#     #     print("x:", x)
#     #     print("y:", y)
#     # break

# print(len(X))


# for i in range(test_dataset.__len__()):
#     x, y = test_dataset[i]
#     x = x.tolist()[:length]
#     y = y.tolist()[-length:]
#     # append the input and target sequences to the lists if they are not already in the lists
#     if x not in X:
#         X.append(x)
#     if y not in Y:
#         Y.append(y)

#     # if "".join(map(str, x)) == "0" * length:
#     #     print("x:", x)
#     #     print("y:", y)

#     # if "".join(map(str, x)) == "1" + "0" * (length - 1):
#     #     print("x:", x)
#     #     print("y:", y)

#     # if "".join(map(str, x)) == "1" * length:
#     #     print("x:", x)
#     #     print("y:", y)

#     # break

# print(len(X))
# %%

# create a GPT instance
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN

model_config = CN(
    n_layer=3,
    n_head=3,
    n_embd=2**4 * 3,
    model_type=None,
    vocab_size=train_dataset.get_vocab_size(),
    block_size=train_dataset.get_block_size(),
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
)

model = GPT(model_config)
# %%

# create a Trainer object
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 3e-4
train_config.max_iters = 3000
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)
# %%


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(
            f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
        )


trainer.set_callback("on_batch_end", batch_end_callback)


# %%
trainer.run()

# %%

model.eval()


# %%
import torch
from torch.utils.data.dataloader import DataLoader


def eval_split(trainer, split, max_batches):
    dataset = {"train": train_dataset, "test": test_dataset}[split]
    n = train_dataset.length
    results = []
    mistakes = []
    mistakes_printed_already = 0
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        # isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # let the model sample the rest of the sequence
        cat = model.generate(
            inp, n, do_sample=False
        )  # using greedy argmax, not sampling
        sol_candidate = cat[:, n:]  # isolate the filled in sequence
        # compare the predicted sequence to the true sequence
        correct = (sol == sol_candidate).all(1).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if (
                not correct[i] and "".join(map(str, inp[i].tolist())) not in mistakes
            ):  # and mistakes_printed_already < 3  # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                mistakes.append("".join(map(str, inp[i].tolist())))
                print(
                    "GPT claims that %s -> %s but g.t. is %s"
                    % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist())
                )
        if max_batches is not None and b + 1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print(
        "%s final score: %d/%d = %.2f%% correct"
        % (split, rt.sum(), len(results), 100 * rt.mean())
    )
    return rt.sum()


# %%

# run a lot of examples from both train and test through the model and verify the output correctness
with torch.no_grad():
    train_score = eval_split(trainer, "train", max_batches=None)  # 50
    test_score = eval_split(trainer, "test", max_batches=None)  # 50


# %%


# let's run a random given sequence through the model as well
n = train_dataset.length  # naugy direct access shrug
inp, sol = train_dataset[3]
inp = inp[:n]
sol = sol[-n:]

inp = inp.unsqueeze(0).to(trainer.device)
sol = sol.unsqueeze(0).to(trainer.device)

assert inp[0].nelement() == n
with torch.no_grad():
    cat = model.generate(inp, n, do_sample=False)

sol_candidate = cat[:, n:]
print("input sequence  :", inp.tolist())
print("output:         ", sol.tolist())
print("predicted:      ", sol_candidate.tolist())
# print('gt sort         :', sol.tolist())
print("matches         :", bool((sol == sol_candidate).all()))
# %%
