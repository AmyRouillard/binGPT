# %%

from utils.tentmapdataset import TentDataset
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import time

# %%


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, repeat=1):
        super(MLP, self).__init__()
        self.repeat = repeat
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

        self.activation = torch.nn.ReLU()
        # self.activation = torch.nn.PReLU(init=0.25)

    def forward(self, x):
        for i in range(self.repeat):
            x = self.fc1(x)
            x = self.activation(x)
            # if i < self.repeat - 1:
            #     x = self.activation(self.fc2(x))
            # else:
            #     x = self.fc2(x)
            x = self.fc2(x)

        return x


# class MLP(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, repeat=1):
#         super(MLP, self).__init__()
#         self.repeat = repeat
#         self.fc1 = torch.nn.Linear(input_size, hidden_size)
#         # self.fc2 = torch.nn.Linear(hidden_size, output_size)

#         # self.activation = torch.nn.ReLU()
#         self.activation = torch.nn.PReLU(init=0.2)

#     def forward(self, x):
#         for i in range(self.repeat):
#             x = self.fc1(x)
#             x = -self.activation(x)
#         return x


# %%

data_type = "decimal"
tokenized = False

n = 1
length = 8 + n

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
batch_size = min([2**6, train_dataset.__len__() + test_dataset.__len__()])

loader = DataLoader(
    train_dataset + test_dataset,
    batch_size=batch_size,
    num_workers=0,
    drop_last=False,
    shuffle=True,
    # shuffle=False,
)

model = MLP(
    input_size=1,
    hidden_size=2,
    output_size=1,
    repeat=n,
)

model.fc1.weight.data = torch.tensor([[1.0], [-1.0]])
model.fc1.bias.data = torch.tensor([-0.5, 0.5])
model.fc2.weight.data = torch.tensor([[-2.0, -2.0]])
model.fc2.bias.data = torch.tensor([1.0])

# model.fc1.weight.data = torch.tensor([[1.0], [-1.0]])
# model.fc1.bias.data = torch.tensor([-0.5, 0.5])
# model.fc2.weight.data = torch.tensor([[-2.0 / (1 - 0.25), -2.0 / (1 - 0.25)]])
# model.fc2.bias.data = torch.tensor([1.0])

# fixed points
x_fixed = torch.tensor([[0.0], [0.5], [1.0]])
y_fixed = torch.tensor([[0.0], [1.0], [0.5]])

# model = MLP(
#     input_size=1,
#     hidden_size=1,
#     output_size=1,
#     repeat=n,
# )
# model.fc1.weight.data = torch.tensor([[1.0]])
# model.fc1.bias.data = torch.tensor([0.0])
# model.activation.weight.data = torch.tensor([-1.0])


# print model weights
print("Model weights:")
for name, param in model.named_parameters():
    print(name, param.data)


criterion = torch.nn.MSELoss()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i, (x, y) in enumerate(loader):

    # x = x - 0.5
    # y = y - 0.5

    print("Batch", i)
    print("x:", x.shape)
    print("y:", y.shape)
    print()

    # forward pass
    y_pred = model(x)
    y_fixed_pred = model(x_fixed)

    # print input and output
    print("Input:", x.shape)
    print("Output:", y_pred.shape)

    ax.plot(x, y, "b.", markersize=2, label=f"$n={n}$")
    ax.plot(x, y_pred.detach().numpy().flatten(), "r.", markersize=2, label=f"$f^n(x)$")

    loss = criterion(y_pred, y)
    loss_fixed = criterion(y_fixed_pred, y_fixed)
    print(f"Loss: {loss.item():.2e}")
    print(f"Loss fixed points: {loss_fixed.item():.2e}")

ax.set_xlabel("x", fontsize=14)
ax.set_ylabel(f"$f^n(x)$", fontsize=14)
# ax.legend(fontsize=10)
ax.set_aspect("equal")

# %%

for name, param in model.named_parameters():
    print(name, param.data)

# %%


w = np.linspace(-3, 3, 500)
b = np.linspace(-3, 3, 500)
# x = np.linspace(0, 1, 500)


# check when w*x + b > 0
def condition(w, x, b):
    # return (w * x + b > 0 - 1e-5) & (w * x + b < 1 + 1e-5)
    return w * x + b > 0


W, B = np.meshgrid(w, b)

mask = np.full_like(W, 0)
for i, (x, y) in enumerate(loader):
    x = x.numpy().flatten()
    for x_val in x:
        mask += condition(W, x_val, B)

fig_cont, ax_cont = plt.subplots(2, 2, figsize=(10, 6))
# plot heatmap of the mask
for j in range(2):
    for i in range(1):
        c = ax_cont[i, j].pcolormesh(
            W, B, mask / len(x), cmap="viridis", shading="auto"
        )
        # overlay contour lines
        ax_cont[i, j].contour(
            W,
            B,
            mask / len(x),
            levels=[0.01, 0.3, 0.5, 0.7, 0.9],
            colors="black",
            linewidths=0.5,
        )

        ax_cont[i, j].plot(
            np.linspace(-3, 3, 10),
            -0.5 * np.linspace(-3, 3, 10),
            "--r",
        )
        # axis equal
        ax_cont[i, j].set_aspect("equal", adjustable="box")
        # axis labels
        ax_cont[i, j].set_xlabel("w")
        ax_cont[i, j].set_ylabel("b")
# colorbar
# cbar = fig_cont.colorbar(c, ax=ax_cont[1])
# for i in range(2):
#     ax_cont[1, i].plot(
#         np.linspace(0, 1000, 10),
#         -np.ones_like(np.linspace(0, 1000, 10)),
#         "--r",
#     )

labels = ["w0", "w1", "b0", "b1", "w'0", "w'1", "b'0"]
import matplotlib.colors as mcolors

c = [v for v in mcolors.TABLEAU_COLORS.values()]
# c = ["b", "g", "r", "c", "m", "y", "k"]
current_best_loss = 1e9
max_tries = 10
tries = 0
while current_best_loss > 1e-1 or tries < max_tries:

    model = MLP(
        input_size=1,
        hidden_size=2,
        output_size=1,
        repeat=n,
    )

    # model.activation.weight.requires_grad = False

    # for param in model.fc1.parameters():
    #     param.requires_grad = False

    model.fc1.weight.data = torch.randn(2, 1)  # torch.tensor([[0.7], [-0.2]])  #
    model.fc1.weight.data = torch.clamp(model.fc1.weight.data, -2.0, 2.0)

    model.fc1.bias.data = torch.randn(2)  # torch.tensor([-0.2, 0.6])  #
    model.fc1.bias.data = torch.clamp(model.fc1.bias.data, -2.0, 2.0)

    # mask = np.full_like(W, 0)
    # for x_val in x:
    #     mask += condition(W, x_val, B)

    model.fc2.weight.data = torch.randn(1, 2)  # torch.tensor([[-1.0, -1.0]])  #
    # model.fc2.weight.data = torch.tensor(
    #     [[2*model.fc1.weight.data[1, 0], -2*model.fc1.weight.data[0, 0]]]
    # )
    # model.fc2.weight.requires_grad = False
    model.fc2.bias.data = torch.randn(1)  # torch.tensor([1.0]) #
    # restrict magnitude of fc1.bais to 0.5
    # for param in model.fc2.parameters():
    #     param.requires_grad = False
    # model.fc1.weight.data = torch.randn(2, 1)
    # model.fc1.bias.data = torch.randn(2)
    # model.fc2.weight.data = torch.tensor([[-2.0, -2.0]])  # torch.randn(1, 2)  #
    # model.fc2.bias.data = torch.tensor([0.5])  # torch.randn(1)  #

    W = []
    weights = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights.extend(param.data.numpy().flatten().tolist())

    W.append(weights)

    G = []
    # w = []
    # for name, param in model.named_parameters():
    # w.extend(param.grad.data.numpy().flatten().tolist())
    # G.append(w)

    # # print model weights
    # print("Model weights:")
    # for name, param in model.named_parameters():
    #     print(name, param.data)

    criterion = torch.nn.MSELoss()
    # with regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=0,
    )
    tries += 1

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=0.001,
    #     momentum=0.3,
    #     weight_decay=1e-4,
    #     # nesterov=True,
    # )
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for epoch in range(1000):

        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()

            # x = x - 0.5
            # y = y - 0.5
            # add noise to x and y
            # x = x + torch.randn_like(x) * 0.1
            # y = y + torch.randn_like(y) * 0.1

            # forward pass
            y_pred = model(x)
            loss = criterion(y_pred, y)

            # y_fixed_pred = model(x_fixed)
            # loss_f = criterion(y_fixed_pred, y_fixed)
            # loss += loss_f  # * (1 + torch.rand(1).item() * 0.5)

            # y_sym = model(1 - x)
            # loss_sym = criterion(y_sym, y)
            # loss += loss_sym  # * (1 + torch.rand(1).item() * 0.5)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # model.fc1.weight.data = torch.clamp(model.fc1.weight.data, -1.0, 1.0)
            # model.fc1.bias.data = torch.clamp(model.fc1.bias.data, -0.5, 0.5)

            # print gradients
            # for name, param in model.named_parameters():
            #     print(name, param.grad.data)

            # print("Loss:", loss.item())

            # if epoch % 10 == 0:
            #     ax.plot(x, y_pred.detach().numpy().flatten(), ".", label=f"$f^n(x)$")
            #     ax.plot(x, y, ".", label=f"$n={n}$")

            w = []
            for name, param in model.named_parameters():
                try:
                    if param.requires_grad:
                        w.extend(param.grad.data.numpy().flatten().tolist())
                except:
                    # all 0
                    if param.requires_grad:
                        w.extend([0.0] * param.data.numel())
            G.append(w)

            weights = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    weights.extend(param.data.numpy().flatten().tolist())

            W.append(weights)

    current_best_loss = loss.item()
    print("Current best loss:", current_best_loss)

    # ["w0", "w1", "b0", "b1", "w'0", "w'1", "b'0"]
    # plot scatter plot of weights and gradients
    W_ = np.array(W)
    G_ = np.array(G)
    ax_cont[0, 0].plot(
        W_[:, 0],
        W_[:, 2],
        ".",
        markersize=1,
        c=c[tries % len(c)],
    )
    ax_cont[0, 0].plot(
        W_[-1, 0],
        W_[-1, 2],
        "*",
        markersize=5,
        label=f"{tries}",
        c=c[tries % len(c)],
    )

    ax_cont[0, 1].plot(
        W_[:, 1],
        W_[:, 3],
        ".",
        markersize=1,
        c=c[tries % len(c)],
    )
    ax_cont[0, 1].plot(
        W_[-1, 1],
        W_[-1, 3],
        "*",
        markersize=5,
        label=f"{tries}",
        c=c[tries % len(c)],
    )

    ax_cont[1, 0].plot(
        # W_[:, 4],  #
        W_[:, 0] * W_[:, 4],
        ".",
        markersize=1,
        label=f"{tries}",
        c=c[tries % len(c)],
    )

    ax_cont[1, 1].plot(
        # W_[:, 5],  #
        W_[:, 1] * W_[:, 5],
        ".",
        markersize=1,
        label=f"{tries}",
        c=c[tries % len(c)],
    )

    # ax_cont[0,0].legend(fontsize=10)
    # ax_cont[0,1].legend(fontsize=10)
    # break

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
W_ = np.array(W)
G_ = np.array(G)
for i in range(G_.shape[1]):
    ax.plot(W_[:, i], f"--{c[i]}", label=f"{labels[i]}")
    ax.plot(G_[:, i], f".{c[i]}", label=f"grad {labels[i]}")

ax.legend(fontsize=10)
# ax.set_title(f"$W_{i}$")
ax.set_xlabel("Epoch", fontsize=10)
# ax.set_ylabel(f"$W_{i}$", fontsize=14)
fig.tight_layout()


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i, (x, y) in enumerate(loader):

    # x = x - 0.5
    # y = y - 0.5

    # forward pass
    y_pred = model(x)

    ax.plot(x, y_pred.detach().numpy().flatten(), ".", label=f"$f^n(x)$")
    ax.plot(x, y, ".", label=f"$n={n}$")

ax.set_xlabel("x", fontsize=14)
ax.set_ylabel(f"$f^n(x)$", fontsize=14)
# ax.legend(fontsize=10)
ax.set_aspect("equal")

# print model weights
print("Model weights:")
for name, param in model.named_parameters():
    print(name, param.data)


# %%

print("continue...")
for param in model.fc1.parameters():
    param.requires_grad = True

W = []
G = []
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for epoch in range(500):

    for i, (x, y) in enumerate(loader):

        # x = x - 0.5
        # y = y - 0.5
        x = x + torch.randn_like(x) * 0.1
        y = y + torch.randn_like(y) * 0.1

        # forward pass
        y_pred = model(x)

        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # print gradients
        # for name, param in model.named_parameters():
        #     print(name, param.grad.data)

        # print("Loss:", loss.item())

        # if epoch % 10 == 0:
        #     ax.plot(x, y_pred.detach().numpy().flatten(), ".", label=f"$f^n(x)$")
        #     ax.plot(x, y, ".", label=f"$n={n}$")

        w = []
        for name, param in model.named_parameters():
            try:
                if param.requires_grad:
                    w.extend(param.grad.data.numpy().flatten().tolist())
            except:
                # all 0
                if param.requires_grad:
                    w.extend([0.0] * param.data.numel())
        G.append(w)

        weights = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights.extend(param.data.numpy().flatten().tolist())

        W.append(weights)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
W_ = np.array(W)
G_ = np.array(G)
for i in range(G_.shape[1]):
    ax.plot(W_[:, i], f"--{c[i]}", label=f"{labels[i]}")
    ax.plot(G_[:, i], f".{c[i]}", label=f"grad {labels[i]}")

ax.legend(fontsize=10)
# ax.set_title(f"$W_{i}$")
ax.set_xlabel("Epoch", fontsize=10)
# ax.set_ylabel(f"$W_{i}$", fontsize=14)
fig.tight_layout()

current_best_loss = loss.item()
print("Current best loss:", current_best_loss)


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i, (x, y) in enumerate(loader):

    # x = x - 0.5
    # y = y - 0.5

    # forward pass
    y_pred = model(x)

    ax.plot(x, y_pred.detach().numpy().flatten(), ".", label=f"$f^n(x)$")
    ax.plot(x, y, ".", label=f"$n={n}$")

ax.set_xlabel("x", fontsize=14)
ax.set_ylabel(f"$f^n(x)$", fontsize=14)
# ax.legend(fontsize=10)
ax.set_aspect("equal")

# print model weights
print("Model weights:")
for name, param in model.named_parameters():
    print(name, param.data)

# %%

# import sympy as sp


# def E(y, y_hat):
#     return (y - y_hat) ** 2


# def tent_map(x, n):
#     if n == 0:
#         return x
#     else:
#         return tent_map((1 - 2 * sp.Abs(x - 0.5)), n - 1)


# def reLU(x):
#     return (x + abs(x)) / 2


# def forward(W, b, Wp, bp, x):
#     x = W @ x + b
#     x = reLU(x)
#     x = Wp @ x + bp
#     return x


# y = sp.symbols("y", real=True)
# y_hat = sp.symbols("y_hat", real=True)

# error = E(y, y_hat)
# error_diff = sp.diff(error, y_hat)
# print("Error function:", error)


# x = sp.symbols("x", real=True)
# tent_map_expr = tent_map(x, n)
# print("Tent map functio:", tent_map_expr)

# W = sp.Matrix(2, 1, lambda i, j: sp.symbols(f"a{i}{j}", real=True))
# b = sp.Matrix(2, 1, lambda i, j: sp.symbols(f"b{i}{j}", real=True))
# Wp = sp.Matrix(1, 2, lambda i, j: sp.symbols(f"a{i}{j}", real=True))
# bp = sp.Matrix(1, 1, lambda i, j: sp.symbols(f"b{i}{j}", real=True))
# xx = sp.Matrix(1, 1, lambda i, j: sp.symbols(f"x{i}{j}", real=True))

# theta = W, b, Wp, bp

# forward_expr = forward(*theta, xx)
# print("Forward pass expression:", forward_expr[0])

# loss_exp = E(tent_map(xx[0], n), forward(W, b, Wp, bp, xx)[0])
# print("Loss expression:", loss_exp)

# grad_W = sp.diff(loss_exp, W, real=True)
# grad_b = sp.diff(loss_exp, b, real=True)
# grad_Wp = sp.diff(loss_exp, Wp, real=True)
# grad_bp = sp.diff(loss_exp, bp, real=True)


# grad = sp.Matrix.vstack(
#     grad_W,
#     grad_b,
#     grad_Wp.T,
#     grad_bp,
# )
# print("Gradient matrix:")
# print(grad)

# %%

# N = 5

# for w00 in np.linspace(-2, 2, N):

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)


t = time.time()
# random initialization
model.fc1.weight.data = torch.randn(2, 1) * 0.1
model.fc1.bias.data = torch.randn(2) * 0.1
model.fc2.weight.data = torch.randn(1, 2) * 0.1
model.fc2.bias.data = torch.randn(1) * 0.1

# W = []
# weights = []
# for name, param in model.named_parameters():
#     weights.extend(param.data.numpy().flatten().tolist())
# W.append(weights)

# G = []
# # w = []
# # for name, param in model.named_parameters():
# # w.extend(param.grad.data.numpy().flatten().tolist())
# # G.append(w)

# # # print model weights
# # print("Model weights:")
# # for name, param in model.named_parameters():
# #     print(name, param.data)


# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for epoch in range(500):

    for i, (x, y) in enumerate(loader):

        # x = x-0.5
        # y = y - 0.5

        # forward pass
        y_pred = model(x)

        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # print gradients
        # for name, param in model.named_parameters():
        #     print(name, param.grad.data)

        # print("Loss:", loss.item())

        # if epoch % 10 == 0:
        #     ax.plot(x, y_pred.detach().numpy().flatten(), ".", label=f"$f^n(x)$")
        #     ax.plot(x, y, ".", label=f"$n={n}$")

        # w = []
        # for name, param in model.named_parameters():
        #     w.extend(param.grad.data.numpy().flatten().tolist())
        # G.append(w)

        # weights = []
        # for name, param in model.named_parameters():
        #     weights.extend(param.data.numpy().flatten().tolist())
        # W.append(weights)


print("Training time:", time.time() - t)
# %%
