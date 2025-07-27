# %%

from utils.tentmapdataset import TentDataset
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import time
from utils.optimizers import SaddleFreeNewton


# %%

slope = 0.01


class shallowNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        super(shallowNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fcL = torch.nn.Linear(hidden_size, output_size)

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "prelu":
            self.activation = torch.nn.PReLU(init=slope)
        else:
            raise ValueError("Unknown activation function: {}".format(activation))

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fcL(x)

        return x

    def initialize_weights(self, method="He", args=None):
        if method == "He":
            torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            torch.nn.init.kaiming_uniform_(self.fcL.weight, nonlinearity="linear")
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fcL.bias)
        elif method == "Shin":
            torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")

            # init bias as in Shin et al. 2020
            if args is None:
                raise ValueError("args must be provided for Shin initialization")
            with torch.no_grad():
                b = torch.mul(self.fc1.weight, args["x1"]).reshape(self.fc1.bias.shape)
                b += (
                    torch.abs(torch.randn_like(b))
                    * torch.sqrt(self.fc1.weight.var(dim=0))
                    * 0.5
                )
                # set bias
            self.fc1.bias = torch.nn.Parameter(b)

            torch.nn.init.kaiming_normal_(self.fcL.weight, nonlinearity="linear")
            torch.nn.init.zeros_(self.fcL.bias)

        else:
            raise ValueError("Unknown initialization method: {}".format(method))


class deepNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hL=1, activation="relu"):
        super(deepNN, self).__init__()
        self.hL = hL  # number of hidden layers
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fcj = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(self.hL - 1)]
        )
        self.fcL = torch.nn.Linear(hidden_size, output_size)

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "prelu":
            self.activation = torch.nn.PReLU(init=slope)
        else:
            raise ValueError("Unknown activation function: {}".format(activation))

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        for z in self.fcj:
            x = z(x)
            x = self.activation(x)
        x = self.fcL(x)

        return x

    def initialize_weights(self, method="He", args=None):
        if method == "He":
            torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            torch.nn.init.kaiming_uniform_(self.fcL.weight, nonlinearity="linear")
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fcL.bias)
            for z in self.fcj:
                torch.nn.init.kaiming_uniform_(z.weight, nonlinearity="relu")
                torch.nn.init.zeros_(z.bias)
        elif method == "Shin":
            torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")

            # init bias as in Shin et al. 2020
            if args is None:
                raise ValueError("args must be provided for Shin initialization")
            with torch.no_grad():
                b = torch.mul(self.fc1.weight, args["x1"]).reshape(self.fc1.bias.shape)
                b += (
                    torch.abs(torch.randn_like(b))
                    * torch.sqrt(self.fc1.weight.var(dim=0))
                    * 0.5
                )
                # set bias
            self.fc1.bias = torch.nn.Parameter(b)

            for i, z in enumerate(self.fcj):

                with torch.no_grad():
                    x = self.fc1(args[f"x{i+1}"])
                    x = self.activation(x)
                    for zz in self.fcj[:i]:
                        x = zz(x)
                        x = self.activation(x)

                torch.nn.init.kaiming_uniform_(z.weight, nonlinearity="relu")
                with torch.no_grad():
                    b = torch.sum(torch.mul(z.weight, x), axis=0).reshape(z.bias.shape)
                    b += (
                        torch.abs(torch.randn_like(b))
                        * torch.sqrt(z.weight.var(dim=0))
                        * 0.5
                    )
                z.bias = torch.nn.Parameter(b)

            torch.nn.init.kaiming_normal_(self.fcL.weight, nonlinearity="linear")
            torch.nn.init.zeros_(self.fcL.bias)

        else:
            raise ValueError("Unknown initialization method: {}".format(method))


# %%

data_type = "decimal"
tokenized = False

q = 2  # number of repetitions
length = q * 2 + max(6 - q, 0)  # length of the binary sequence
shift = True
model_type = "deep"
# model_type = "shallow"
# activation = "relu"
activation = "prelu"

train_dataset = TentDataset(
    "train",
    length=length,
    n_iterations=q,
    type=data_type,
    tokenized=tokenized,
)
test_dataset = TentDataset(
    "test",
    length=length,
    n_iterations=q,
    type=data_type,
    tokenized=tokenized,
)

# Small batch size or full batch size
batch_size = min([2**10, train_dataset.__len__() + test_dataset.__len__()])

loader = DataLoader(
    train_dataset + test_dataset,
    batch_size=batch_size,
    num_workers=0,
    drop_last=False,
    shuffle=True,
)

# %%

if model_type == "shallow":
    W_1 = np.array([2**q for _ in range(2**q)]).reshape(2**q, 1)
    b_1 = np.array([-k for k in range(1, 2**q)] + [0])

    if activation == "relu":
        W_2 = np.array([(-1) ** k * 2 for k in range(1, 2**q)] + [1]).reshape(1, 2**q)
        b_2 = np.array([0])
    elif activation == "prelu":
        W_2 = np.array(
            [(-1) ** k * 2 / (1 - slope) for k in range(1, 2**q)]
            + [(1 + slope) / (1 - slope)]
        ).reshape(1, 2**q)
        b_2 = np.array([-(2**q) * slope / (1 - slope)])
    else:
        raise ValueError("Unknown activation function: {}".format(activation))

    if shift:
        b_1 = b_1 + 2 ** (q - 1)
        b_2 = b_2 - 0.5

    # W_1, b_1, W_2, b_2
    print(W_1.shape, b_1.shape, W_2.shape, b_2.shape)

    model = shallowNN(
        input_size=1,
        hidden_size=2**q,
        output_size=1,
        activation=activation,
    )

    print("Model weights:")
    for name, param in model.named_parameters():
        print(name, param.data, param.shape)
    print()

    # Initialize weights and biases
    with torch.no_grad():
        model.fc1.weight = torch.nn.Parameter(torch.tensor(W_1, dtype=torch.float32))
        model.fc1.bias = torch.nn.Parameter(torch.tensor(b_1, dtype=torch.float32))
        model.fcL.weight = torch.nn.Parameter(torch.tensor(W_2, dtype=torch.float32))
        model.fcL.bias = torch.nn.Parameter(torch.tensor(b_2, dtype=torch.float32))

    print("Model weights:")
    for name, param in model.named_parameters():
        print(name, param.data, param.shape)
    print()

elif model_type == "deep":

    W_1 = np.array([2**1 for _ in range(2**1)]).reshape(2, 1)
    b_1 = np.array([-k for k in range(1, 2**1)] + [0])

    if activation == "relu":
        W_L = np.array([(-1) ** k * 2 for k in range(1, 2**1)] + [1]).reshape(1, 2)
        b_L = np.array([0])
    elif activation == "prelu":
        W_L = np.array(
            [(-1) ** k * 2 / (1 - slope) for k in range(1, 2**1)]
            + [(1 + slope) / (1 - slope)]
        ).reshape(1, 2)
        b_L = np.array([-(2**1) * slope / (1 - slope)])
    else:
        raise ValueError("Unknown activation function: {}".format(activation))

    W_j = W_1 @ W_L
    b_j = W_1 @ b_L + b_1

    if shift:
        b_1 = b_1 + 1
        b_L = b_L - 0.5

    # W_1, b_1, W_j, b_j, W_L, b_L
    print(W_1.shape, b_1.shape, W_j.shape, b_j.shape, W_L.shape, b_L.shape)

    model = deepNN(
        input_size=1,
        hidden_size=2,
        output_size=1,
        hL=q,
        activation=activation,
    )

    print("Model weights:")
    for name, param in model.named_parameters():
        print(name, param.data, param.shape)
    print()

    # Initialize weights and biases
    with torch.no_grad():
        model.fc1.weight = torch.nn.Parameter(torch.tensor(W_1, dtype=torch.float32))
        model.fc1.bias = torch.nn.Parameter(torch.tensor(b_1, dtype=torch.float32))
        for i in range(len(model.fcj)):
            model.fcj[i].weight = torch.nn.Parameter(
                torch.tensor(W_j, dtype=torch.float32)
            )
            model.fcj[i].bias = torch.nn.Parameter(
                torch.tensor(b_j, dtype=torch.float32)
            )
        model.fcL.weight = torch.nn.Parameter(torch.tensor(W_L, dtype=torch.float32))
        model.fcL.bias = torch.nn.Parameter(torch.tensor(b_L, dtype=torch.float32))

    print("Model weights:")
    for name, param in model.named_parameters():
        print(name, param.data, param.shape)
    print()

# %%

criterion = torch.nn.MSELoss()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i, (x, y) in enumerate(loader):

    print("Batch", i)
    print("x:", x.shape)
    print("y:", y.shape)
    print()

    if shift:
        x = x - 0.5
        y = y - 0.5

    # forward pass
    y_pred = model(x)
    # y_fixed_pred = model(x_fixed)

    # print input and output
    print("Input:", x.shape)
    print("Output:", y_pred.shape)

    ax.plot(
        x,
        y,
        "b.",
        markersize=2,
        label=f"$true$" if i == 0 else None,
    )
    ax.plot(
        x,
        y_pred.detach().numpy().flatten(),
        "r.",
        markersize=2,
        label=f"$predicted$" if i == 0 else None,
    )

    loss = criterion(y_pred, y)
    print(f"Loss: {loss.item():.2e}")

ax.set_xlabel("x", fontsize=14)
ax.set_ylabel(f"$f^q(x)$", fontsize=14)
ax.legend(fontsize=10)
ax.set_aspect("equal")

# %%

if model_type == "deep":
    model = deepNN(
        input_size=1,
        hidden_size=2,
        output_size=1,
        hL=q,
        activation=activation,
    )
elif model_type == "shallow":
    model = shallowNN(
        input_size=1,
        hidden_size=2**q,
        output_size=1,
        activation=activation,
    )

# init_method = "He"
init_method = "Shin"

if init_method == "He":
    args = None
elif init_method == "Shin":
    n = model.fc1.bias.shape[0]
    if hasattr(model, "fcj"):
        for i, _ in enumerate(model.fcj):
            n += model.fcj[i].bias.shape[0]

    step = int(len(train_dataset) / 2 / n)
    # indices = np.random.choice(len(train_dataset), n, replace=False)
    indices = [step + i * 2 * step for i in range(n)]
    x1 = torch.tensor(
        [train_dataset[i][0].item() for i in indices], dtype=torch.float32
    ).reshape(n, -1)

    ind = model.fc1.bias.shape[0]
    args = {"x1": x1[0:ind]}
    if hasattr(model, "fcj"):
        for i, z in enumerate(model.fcj):
            args[f"x{i+1}"] = x1[ind : z.bias.shape[0] + ind]

model.initialize_weights(method=init_method, args=args)

# do not train activation parameters
for name, param in model.named_parameters():
    if "activation" in name:
        param.requires_grad = False

print("Model weights:")
for name, param in model.named_parameters():
    print(name, param.data, param.shape)
print()


W = []
weights = []
w_name = []
for name, param in model.named_parameters():
    if param.requires_grad:
        weights.extend(param.data.numpy().flatten().tolist())
        w_name += [name + f"_{i}" for i in range(len(param.data.numpy().flatten()))]

W.append(weights)

G = []

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=0.01,
#     momentum=0.0,
# )
# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=0.003,
#     betas=(0.9, 0.999),
#     eps=1e-08,
#     weight_decay=0.0,
#     amsgrad=False,
# )
optimizer = SaddleFreeNewton(
    model.parameters(),
    lr=0.003,
)

fig_model, ax_model = plt.subplots(1, 1, figsize=(10, 5))

current_loss = torch.inf
loss_threshold = 1e-6
stop_condition = False
for epoch in range(100000):

    for i, (x, y) in enumerate(loader):

        if shift:
            x = x - 0.5
            y = y - 0.5

        if epoch == 0:
            with torch.no_grad():
                y_pred = model(x)
            ax_model.plot(
                x, y_pred.detach().numpy().flatten(), ".", label=f"initial prediction"
            )

        def closure():
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)

            if isinstance(optimizer, SaddleFreeNewton):
                pass
            else:
                loss.backward()

            return loss

        if isinstance(optimizer, SaddleFreeNewton):
            loss, update = optimizer.step(closure)

            w = []
            for p in update:
                w.extend(p.detach().numpy().flatten())
            G.append(w)

            # y_pred = model(x)
            # loss = criterion(y_pred, y)
            # loss.backward()
            # w = []
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         w.extend(param.grad.data.numpy().flatten())
            # G.append(w)
        else:
            loss = optimizer.step(closure)
            w = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    w.extend(param.grad.data.numpy().flatten())
            G.append(w)

        weights = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights.extend(param.data.numpy().flatten())

        W.append(weights)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.2e}")

        if (
            current_loss < loss_threshold
        ):  # torch.abs(loss - current_loss) < loss_threshold:
            stop_condition = True
            break
        current_loss = loss.item()
    if stop_condition:
        print(
            f"Stopping training at epoch {epoch} due to convergence. Current loss: {current_loss:.2e}"
        )
        break

print()


# print model weights
print("Model weights:")
for name, param in model.named_parameters():
    print(name, param.data)

print()

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
W_ = np.array(W)
G_ = np.array(G)
for i in range(G_.shape[1]):
    mean_G = np.max(np.abs(G_[:, i]))
    # print(f"Weight {i}: {w_name[i]}, mean gradient: {mean_G:.2e}")
    if mean_G > 1e-3:
        name = w_name[i]
        if "bias" in name:
            ax[0, 1].plot(W_[:, i], ".", label=f"{name}")
            ax[1, 1].plot(G_[:, i], ".", label=f"{name}_grad")
        else:
            ax[0, 0].plot(W_[:, i], ".", label=f"{name}")
            ax[1, 0].plot(G_[:, i], ".", label=f"{name}_grad")
    else:
        print(f"{w_name[i]} is dead")

# legend outside the plot
for i in range(2):
    ax[0, i].legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    ax[1, i].legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    ax[0, i].set_xlabel("Epoch", fontsize=10)
    ax[1, i].set_xlabel("Epoch", fontsize=10)

ax[0, 0].set_title("Weights $W_i$", fontsize=14)
ax[1, 0].set_title("Gradients $W_i$", fontsize=14)
ax[0, 1].set_title("Bias $b_i$", fontsize=14)
ax[1, 1].set_title("Gradients $b_i$", fontsize=14)


fig.tight_layout()


for i, (x, y) in enumerate(loader):

    if shift:
        x = x - 0.5
        y = y - 0.5

    # forward pass
    y_pred = model(x)

    ax_model.plot(x, y_pred.detach().numpy().flatten(), ".", label=f"final prediction")
    ax_model.plot(x, y, "o", label=f"true", markersize=2, markerfacecolor="none")

ax_model.set_xlabel("x", fontsize=14)
ax_model.set_ylabel(f"$f^n(x)$", fontsize=14)
ax_model.legend(fontsize=10)
ax_model.set_aspect("equal")

fig_model.tight_layout()

# %%


y_pred = model(x)
loss = criterion(y_pred, y)

# Flatten parameters
params = [p for p in model.parameters() if p.requires_grad]
flat_params = torch.cat([p.contiguous().view(-1) for p in params])

# First-order gradient
grad = torch.autograd.grad(loss, params, create_graph=True)
flat_grad = torch.cat([g.contiguous().view(-1) for g in grad])

# Hessian matrix
hessian = []
for g in flat_grad:
    second_grads = torch.autograd.grad(g, params, retain_graph=True)
    h_row = torch.cat([sg.contiguous().view(-1) for sg in second_grads])
    hessian.append(h_row)

hessian = torch.stack(hessian)  # shape: [n_params, n_params]
print("Hessian shape:", hessian.shape)

eigenvalues, eigenvectors = torch.linalg.eig(hessian)

tol = 1e-8  # tolerance for numerical stability
# split into positive and negative eigenvalues
positive_mask = eigenvalues.real > tol
positive_eigenvalues = eigenvalues[positive_mask]
indices_positive = torch.where(positive_mask)[0]

negative_mask = eigenvalues.real < -tol
negative_eigenvalues = eigenvalues[negative_mask]
indices_negative = torch.where(negative_mask)[0]

zero_mask = torch.abs(eigenvalues.real) <= tol
zero_eigenvalues = eigenvalues[zero_mask]
indices_zero = torch.where(zero_mask)[0]

# plot real part of eigenvalues as bar chart
plt.figure(figsize=(10, 5))
plt.plot(
    indices_positive.numpy(),
    np.abs(positive_eigenvalues.real.numpy()),
    "or",
    markersize=3,
    label="Real Part of Positive Eigenvalues",
)
plt.plot(
    indices_negative.numpy(),
    np.abs(negative_eigenvalues.real.numpy()),
    "ob",
    markersize=3,
    label="Real Part of Negative Eigenvalues",
)
plt.plot(
    indices_zero.numpy(),
    np.abs(zero_eigenvalues.real.numpy()),
    "ok",
    markersize=3,
    label="Real Part of Zero Eigenvalues",
)
# log scale for better visibility
plt.yscale("log")
plt.xlabel("Eigenvalue Index")
plt.ylabel("Real Part of Eigenvalue")
plt.title("Eigenvalues of the Hessian Matrix")
plt.legend()

# if all eigenvalues are positive, the model is convex
if len(negative_eigenvalues) != 0 and len(positive_eigenvalues) != 0:
    print("Saddle point.")
elif len(positive_eigenvalues) == 0 and len(zero_eigenvalues) == 0:
    print("Local maximum.")
elif len(negative_eigenvalues) == 0 and len(zero_eigenvalues) == 0:
    print("Local minimum.")
elif len(negative_eigenvalues) == 0 and len(zero_eigenvalues) != 0:
    print("Inconclusive: zero eigenvalues present, rest positive.")
elif len(positive_eigenvalues) == 0 and len(zero_eigenvalues) != 0:
    print("Inconclusive: zero eigenvalues present, rest negative.")

# %%

# import seaborn as sns

# result = np.zeros((len(eigenvalues), len(eigenvalues)), dtype=np.complex128)
# for i1, e1 in enumerate(eigenvectors):
#     for i2, e2 in enumerate(eigenvectors):
#         V = torch.dot(e1, e2)
#         result[i1, i2] = V.real.item() + 1j * V.imag.item()

# fig, ax = plt.subplots(1, 2, figsize=(10, 10))
# # heat map
# sns.heatmap(
#     result.real,
#     ax=ax[0],
#     cmap="coolwarm",
#     cbar_kws={"label": "Real Part of Eigenvectors Inner Product"},
# )
# sns.heatmap(
#     result.imag,
#     ax=ax[1],
#     cmap="coolwarm",
#     cbar_kws={"label": "Imaginary Part of Eigenvectors Inner Product"},
# )

# %%
