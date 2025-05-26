# %%

from utils.tentmapdataset import TentDataset
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader

# %%

data_type = "binary"
tokenized = True
n = 3
length = 10

for data_type in ["binary", "decimal"]:
    for tokenized in [True, False]:

        train_dataset = TentDataset(
            "train",
            length=length,
            n_iterations=n,
            type=data_type,
            tokenized=tokenized,
        )
        test_dataset = TentDataset(
            "test",
            length=length,
            n_iterations=n,
            type=data_type,
            tokenized=tokenized,
        )

        print("Size of train dataset:", train_dataset.__len__())
        print("Size of test dataset:", test_dataset.__len__())
        print(
            "Total size of dataset:",
            train_dataset.__len__() + test_dataset.__len__(),
            f"(==2**{length}={2**length})",
        )

        print("Example of train dataset:")
        x, y = train_dataset[0]

        print("x:", x)
        print("y:", y)

        print("Example of test dataset:")
        x, y = test_dataset[0]

        print("x:", x)
        print("y:", y)

# %%


data_type = "decimal"
tokenized = False

length = 23

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for n in [1, 2, 3, 4]:

    train_dataset = TentDataset(
        "train",
        length=length,
        n_iterations=n,
        type=data_type,
        tokenized=tokenized,
    )
    test_dataset = TentDataset(
        "test",
        length=length,
        n_iterations=n,
        type=data_type,
        tokenized=tokenized,
    )

    # Small batch size or full batch size
    batch_size = min([2**9, train_dataset.__len__() + test_dataset.__len__()])

    loader = DataLoader(
        train_dataset + test_dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
        shuffle=True,
    )

    print("First batch of data:")
    for i, (x, y) in enumerate(loader):
        print("Batch", i)
        print("x:", x.shape)
        print("y:", y.shape)
        print()

        break

    x = x.numpy().flatten()
    y = y.numpy().flatten()

    ax.plot(x, y, ".", markersize=2, label=f"$n={n}$")

ax.set_xlabel("x", fontsize=14)
ax.set_ylabel(f"$f^n(x)$", fontsize=14)
ax.legend(fontsize=10)
ax.set_aspect("equal")

# %%
