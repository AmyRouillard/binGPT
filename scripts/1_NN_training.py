# %%

from utils.tentmapdataset import TentDataset
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import time
from utils.optimizers import SaddleFreeNewton


# %%


class ffNN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, hL=1, activation="relu", slope=0.01
    ):
        super(ffNN, self).__init__()
        self.hL = hL  # number of hidden layers
        self.dropout = torch.nn.Dropout(p=0.2)

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * hL
        elif isinstance(hidden_size, list):
            if len(hidden_size) != hL:
                raise ValueError("hidden_size must be an int or a list of length hL")

        self.fc1 = torch.nn.Linear(input_size, hidden_size[0])
        self.fcj = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size[i], hidden_size[i + 1]) for i in range(hL - 1)]
        )
        self.fcL = torch.nn.Linear(hidden_size[-1], output_size)

        self.slope = slope

        self.activation_name = activation
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "prelu":
            self.activation = torch.nn.PReLU(init=slope)
        elif activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU(negative_slope=slope)
        else:
            raise ValueError("Unknown activation function: {}".format(activation))

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        # x = self.dropout(x)
        for z in self.fcj:
            x = z(x)
            x = self.activation(x)
            # x = self.dropout(x)
        x = self.fcL(x)

        return x

    def count_active_neurons(self, x):
        """
        Returns a dict with layer names and counts of non-zero activations per neuron.
        """
        counts = {}
        with torch.no_grad():
            out = self.fc1(x)
            act1 = self.activation(out)
            counts["fc1"] = (act1 != 0).sum(dim=0).cpu().numpy()
            for i, layer in enumerate(self.fcj):
                out = layer(act1)
                act1 = self.activation(out)
                counts[f"fcj_{i}"] = (act1 != 0).sum(dim=0).cpu().numpy()
            # Output layer is usually not checked for dead neurons
        return counts

    def scale_weights(self, scale=1.0):
        with torch.no_grad():
            self.fc1.weight.mul_(scale)
            self.fcL.weight.mul_(scale)
            for z in self.fcj:
                z.weight.mul_(scale)

    def plot_z(self, x):
        X = self.activation(self.fc1(x))
        N = self.fc1.in_features
        fig, ax = plt.subplots(self.hL + 1, 1, figsize=(10, 6 * (self.hL + 1)))
        for i in range(X.shape[1]):
            ax[0].plot(
                x.detach().cpu().numpy().reshape(-1),
                X[:, i].detach().cpu().numpy(),
                ".",
                markersize=2,
                label=f"Neuron {i}",
            )

            y = X[:, i].detach().cpu().numpy()
            dx = np.diff(x.detach().cpu().numpy().reshape(-1))

            dy = np.diff(y)
            d2y = (dy[1:] / dx[1:] - dy[:-1] / dx[:-1]) / ((dx[1:] + dx[:-1]) / 2)

            indices = np.argsort(np.abs(d2y))[-N:]

            ax[0].plot(
                x.detach().cpu().numpy().reshape(-1)[:-2][indices],
                X[:, i].detach().cpu().numpy()[:-2][indices],
                "ok",
                markersize=5,
                label=f"Neuron {i}",
            )

        ax[0].set_xlabel("X", fontsize=14)
        ax[0].set_ylabel(f"z_{1}", fontsize=14)

        for i, z in enumerate(self.fcj):
            N *= z.in_features
            X = self.activation(z(X))
            for j in range(X.shape[1]):
                ax[i + 1].plot(
                    x.detach().cpu().numpy().reshape(-1),
                    X[:, j].detach().cpu().numpy(),
                    ".",
                    markersize=2,
                    label=f"Neuron {j}",
                )

                y = X[:, j].detach().cpu().numpy()
                dx = np.diff(x.detach().cpu().numpy().reshape(-1))
                dy = np.diff(y)
                d2y = (dy[1:] / dx[1:] - dy[:-1] / dx[:-1]) / ((dx[1:] + dx[:-1]) / 2)
                indices = np.argsort(np.abs(d2y))[-N:]

                ax[i + 1].plot(
                    x.detach().cpu().numpy().reshape(-1)[:-2][indices],
                    X[:, j].detach().cpu().numpy()[:-2][indices],
                    "ok",
                    markersize=5,
                    label=f"Neuron {i}",
                )

            ax[i + 1].set_xlabel("X", fontsize=14)
            ax[i + 1].set_ylabel(f"z_{i+2}", fontsize=14)

        N *= self.fcL.in_features
        X = self.fcL(X)
        for i in range(X.shape[1]):
            ax[self.hL].plot(
                x.detach().cpu().numpy().reshape(-1),
                X[:, i].detach().cpu().numpy(),
                ".",
                markersize=2,
                label=f"Neuron {i}",
            )

            y = X[:, i].detach().cpu().numpy()
            dx = np.diff(x.detach().cpu().numpy().reshape(-1))
            dy = np.diff(y)
            d2y = (dy[1:] / dx[1:] - dy[:-1] / dx[:-1]) / ((dx[1:] + dx[:-1]) / 2)
            indices = np.argsort(np.abs(d2y))[-N:]

            ax[self.hL].plot(
                x.detach().cpu().numpy().reshape(-1)[:-2][indices],
                X[:, i].detach().cpu().numpy()[:-2][indices],
                "ok",
                markersize=5,
                label=f"Neuron {i}",
            )
        fig.tight_layout()
        ax[self.hL].set_xlabel("X", fontsize=14)
        ax[self.hL].set_ylabel("Prediction", fontsize=14)

        fig.tight_layout()

        return fig, ax

    def get_inputs_for_biases(self, in_features, out_features):

        # Fixed distribution
        offset = 0.5 if out_features > 1 else 0.0
        x = torch.stack(
            [torch.linspace(0, 1, out_features) - offset for _ in range(in_features)],
            dim=1,
        )

        # # Random within bounds, fix vector
        # if out_features == 1:
        #     x = torch.rand(out_features) * 0.5
        # elif out_features % 2 == 0:
        #     bounds = torch.linspace(0, 0.5, out_features // 2 + 1)
        #     # generate a random x in range bounds[i] to bounds[i+1]
        #     xright = [
        #         torch.rand(1) * (bounds[i + 1] - bounds[i]) + bounds[i]
        #         for i in range(len(bounds) - 1)
        #     ]
        #     xsym = [-i for i in xright[::-1]]
        #     # concatenate the two halves
        #     x = torch.tensor(xsym + xright)
        # else:
        #     n = (out_features - 1) // 2
        #     bounds = torch.linspace(1 / n / 2, 0.5, n)
        #     xright = [
        #         torch.rand(1) * (bounds[i + 1] - bounds[i]) + bounds[i]
        #         for i in range(len(bounds) - 1)
        #     ]
        #     xmid = [(-2 * torch.rand(1) + 1) * bounds[0]]
        #     xsym = [-i for i in xright[::-1]]
        #     # concatenate the two halves
        #     x = torch.tensor(xsym + xmid + xright)

        # x = torch.stack(
        #     [x for _ in range(in_features)],
        #     dim=1,
        # )

        # # Random within bounds
        # X = []
        # for _ in range(in_features):
        #     if out_features == 1:
        #         x = torch.rand(out_features) * 0.5
        #     elif out_features % 2 == 0:
        #         bounds = torch.linspace(0, 0.5, out_features // 2 + 1)
        #         # generate a random x in range bounds[i] to bounds[i+1]
        #         xright = [
        #             torch.rand(1) * (bounds[i + 1] - bounds[i]) + bounds[i]
        #             for i in range(len(bounds) - 1)
        #         ]
        #         xsym = [-i for i in xright[::-1]]
        #         # concatenate the two halves
        #         x = torch.tensor(xsym + xright)
        #     else:
        #         n = (out_features - 1) // 2
        #         bounds = torch.linspace(1 / n / 2, 0.5, n)
        #         xright = [
        #             torch.rand(1) * (bounds[i + 1] - bounds[i]) + bounds[i]
        #             for i in range(len(bounds) - 1)
        #         ]
        #         xmid = [(-2 * torch.rand(1) + 1) * bounds[0]]
        #         xsym = [-i for i in xright[::-1]]
        #         # concatenate the two halves
        #         x = torch.tensor(xsym + xmid + xright)

        #     X.append(x)

        # x = torch.stack(
        #     X,
        #     dim=1,
        # )

        # # Random
        # X = []
        # for _ in range(in_features):
        #     if out_features == 1:
        #         x = torch.rand(out_features) * 0.5
        #     elif out_features % 2 == 0:
        #         xright = [0.5 - torch.rand(1) for _ in range(out_features // 2)]
        #         xsym = [-i for i in xright[::-1]]
        #         # concatenate the two halves
        #         x = torch.tensor(xsym + xright)
        #     else:
        #         n = (out_features - 1) // 2
        #         xright = [0.5 - torch.rand(1) for _ in range(n)]
        #         xmid = [-torch.rand(1) + 0.5]
        #         xsym = [-i for i in xright[::-1]]
        #         # concatenate the two halves
        #         x = torch.tensor(xsym + xmid + xright)

        #     X.append(x)

        # x = torch.stack(
        #     X,
        #     dim=1,
        # )

        return x

    def initialize_weights(self, method="He", args=None):
        if method == "He":
            torch.nn.init.kaiming_uniform_(
                self.fc1.weight,
                nonlinearity="relu",  # mode="fan_out"
            )
            torch.nn.init.kaiming_uniform_(
                self.fcL.weight,
                nonlinearity="linear",  # mode="fan_out"
            )
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fcL.bias)
            for z in self.fcj:
                torch.nn.init.kaiming_uniform_(
                    z.weight,
                    nonlinearity="relu",  # mode="fan_out"
                )
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
        elif method == "kink_maximizing":
            torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
            torch.nn.init.zeros_(self.fc1.bias)

            for z in self.fcj:
                torch.nn.init.kaiming_normal_(z.weight, nonlinearity="relu")
                torch.nn.init.zeros_(z.bias)

            torch.nn.init.kaiming_normal_(self.fcL.weight, nonlinearity="relu")
            torch.nn.init.zeros_(self.fcL.bias)

            with torch.no_grad():
                x = self.get_inputs_for_biases(
                    self.fc1.in_features, self.fc1.out_features
                )

                self.fc1.bias = torch.nn.Parameter(torch.diag(-self.fc1(x)))

                for i, z in enumerate(self.fcj):
                    x = self.get_inputs_for_biases(z.in_features, z.out_features)
                    z.bias = torch.nn.Parameter(torch.diag(-z(x)))

                # x = self.get_inputs_for_biases(
                #     self.fcL.in_features, self.fcL.out_features
                # )
                # self.fcL.bias = torch.nn.Parameter(torch.diag(-self.fcL(x)))

        elif method == "New":
            torch.nn.init.kaiming_uniform_(
                self.fc1.weight,
                nonlinearity="relu",  # mode="fan_out"
            )
            # make all weights positive
            # self.fc1.weight.data = torch.abs(self.fc1.weight.data)
            torch.nn.init.zeros_(self.fc1.bias)

            for z in self.fcj:
                torch.nn.init.kaiming_uniform_(
                    z.weight,
                    nonlinearity="relu",  # mode="fan_out"
                )
                # z.weight.data = torch.abs(z.weight.data)
                torch.nn.init.zeros_(z.bias)

            torch.nn.init.kaiming_uniform_(
                self.fcL.weight,
                nonlinearity="linear",  # mode="fan_out"
            )
            # self.fcL.weight.data = torch.abs(self.fcL.weight.data)
            torch.nn.init.zeros_(self.fcL.bias)

            x = args["x"]
            with torch.no_grad():
                self.fc1.bias = torch.nn.Parameter(torch.diag(-self.fc1(x)))
                x = self.activation(self.fc1(x))
                X = self.activation(self.fc1(X))

                for i, z in enumerate(self.fcj):
                    # z.bias = torch.nn.Parameter(torch.diag(-z(x)))
                    x = self.activation(z(x))

                self.fcL.bias = torch.nn.Parameter(torch.diag(-self.fcL(x)))

        elif method == "Newnew":
            torch.nn.init.kaiming_uniform_(
                self.fc1.weight,
                nonlinearity="relu",  # mode="fan_out"
            )
            # make all weights positive
            # self.fc1.weight.data = torch.abs(self.fc1.weight.data)
            torch.nn.init.zeros_(self.fc1.bias)

            for z in self.fcj:
                torch.nn.init.kaiming_uniform_(
                    z.weight,
                    nonlinearity="relu",  # mode="fan_out"
                )
                # z.weight.data = torch.abs(z.weight.data)
                torch.nn.init.zeros_(z.bias)

            torch.nn.init.kaiming_uniform_(
                self.fcL.weight,
                nonlinearity="linear",  # mode="fan_out"
            )
            # self.fcL.weight.data = torch.abs(self.fcL.weight.data)
            torch.nn.init.zeros_(self.fcL.bias)

            x = args["x"]
            with torch.no_grad():
                self.fc1.bias = torch.nn.Parameter(torch.diag(-self.fc1(x)))
                x = self.activation(self.fc1(x))

                for i, z in enumerate(self.fcj):

                    # z.bias = torch.nn.Parameter(torch.diag(-z(x)))
                    # stack args["x"].reshape(-1) z.out_features times
                    x = torch.stack(
                        [args["x"].reshape(-1) for _ in range(z.in_features)], dim=1
                    )

                    x = self.activation(z(x))

                # x = torch.stack([args["x"].reshape(-1) for _ in range(self.fcL.in_features)], dim=0 )
                # self.fcL.bias = torch.nn.Parameter(torch.diag(-self.fcL(x)))

        else:
            raise ValueError("Unknown initialization method: {}".format(method))

    def get_gradients(self, loss):
        """
        Returns a dictionary of gradients for each parameter.
        """
        grads = torch.autograd.grad(loss, self.parameters(), retain_graph=True)
        grad_dict = {}
        for (name, param), grad in zip(self.named_parameters(), grads):
            grad_dict[name] = grad.detach().cpu().numpy()
        return grad_dict

    def get_flat_gradient(self, loss):
        """
        Returns the flattened gradient vector.
        """
        grads = torch.autograd.grad(loss, self.parameters(), retain_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
        return flat_grad.detach().cpu().numpy()

    def compute_hessian(self, loss):
        """
        Computes the Hessian matrix of the loss with respect to model parameters.
        Returns a [n_params, n_params] torch.Tensor.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        flat_grad = torch.cat(
            [
                g.contiguous().view(-1)
                for g in torch.autograd.grad(loss, params, create_graph=True)
            ]
        )
        hessian = []
        for g in flat_grad:
            second_grads = torch.autograd.grad(g, params, retain_graph=True)
            h_row = torch.cat([sg.contiguous().view(-1) for sg in second_grads])
            hessian.append(h_row)
            print(h_row.shape)
        hessian = torch.stack(hessian)
        return hessian

    def compute_hessian_blocks(self, loss):
        """
        Returns a dict mapping (name0, name1) to Hessian blocks of shape
        [param0.numel(), param1.numel()].
        """
        params = [p for p in self.parameters() if p.requires_grad]
        names = [name for name, p in self.named_parameters() if p.requires_grad]
        hessian_dict = {}

        # Compute gradients for each parameter
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grads = []
        for g in grads:
            flat_grads.append(g.contiguous().view(-1))

        # For each pair of parameters, compute the Hessian block
        for i, (name_i, param_i) in enumerate(zip(names, params)):
            for j, (name_j, param_j) in enumerate(zip(names, params)):
                block = []
                for k in range(param_i.numel()):
                    grad_i_k = flat_grads[i][k]
                    second_grads = torch.autograd.grad(
                        grad_i_k, param_j, retain_graph=True
                    )[0]
                    block.append(second_grads.contiguous().view(-1))
                block_tensor = torch.stack(
                    block
                )  # shape: [param_i.numel(), param_j.numel()]
                hessian_dict[(name_i, name_j)] = block_tensor
        return hessian_dict

    def prune_dead_neurons(self, x, threshold=0):
        """
        Removes dead neurons (neurons with activation count <= threshold) from hidden layers.
        Updates weights and biases accordingly.
        """
        with torch.no_grad():
            # Prune fc1
            out = self.fc1(x)
            act1 = self.activation(out)
            alive_mask = (act1 != 0).sum(dim=0).cpu().numpy() > threshold
            if alive_mask.sum() <= 1:
                raise ValueError(
                    "All but one of the neurons in fc1 are dead. Cannot prune."
                )
            if alive_mask.sum() < self.fc1.out_features:
                new_out_features = int(alive_mask.sum())
                new_fc1 = torch.nn.Linear(self.fc1.in_features, new_out_features)
                new_fc1.weight.data = self.fc1.weight.data[alive_mask, :]
                new_fc1.bias.data = self.fc1.bias.data[alive_mask]
                self.fc1 = new_fc1

            # Prune fcj layers
            prev_mask = alive_mask
            for i, layer in enumerate(self.fcj):
                out = layer(act1)
                act1_tmp = self.activation(out)
                alive_mask = (act1_tmp != 0).sum(dim=0).cpu().numpy() > threshold
                # if alive_mask.sum() <= 1:
                #     raise ValueError(f"All but one of neurons in fcj_{i} are dead. Cannot prune.")
                if alive_mask.sum() <= 1:
                    # Mark this layer for removal
                    self.fcj[i] = None
                    # prev_mask = np.ones(layer.out_features, dtype=bool)
                    continue
                act1 = act1_tmp
                if alive_mask.sum() < layer.out_features:
                    new_layer = torch.nn.Linear(
                        layer.in_features, int(alive_mask.sum())
                    )
                    new_layer.weight.data = layer.weight.data[alive_mask, :]
                    new_layer.bias.data = layer.bias.data[alive_mask]
                    self.fcj[i] = new_layer
                # Prune input weights if previous layer was pruned
                if prev_mask.sum() < layer.in_features:
                    self.fcj[i].weight.data = self.fcj[i].weight.data[:, prev_mask]
                prev_mask = alive_mask

            # Remove None layers from fcj
            self.fcj = torch.nn.ModuleList(
                [layer for layer in self.fcj if layer is not None]
            )

            # Prune fcL input weights if last hidden layer was pruned
            if prev_mask.sum() < self.fcL.in_features:
                new_fcL = torch.nn.Linear(int(prev_mask.sum()), self.fcL.out_features)
                new_fcL.weight.data = self.fcL.weight.data[:, prev_mask]
                new_fcL.bias.data = self.fcL.bias.data
                self.fcL = new_fcL


# %%

data_type = "decimal"
tokenized = False

q = 4  # number of repetitions
length = 10  # q * 2 + max(6 - q, 0)  # length of the binary sequence
shift = True
# shift = False
model_type = "deep"
# model_type = "shallow"
activation = "relu"
# activation = "prelu"

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
size_dataset = train_dataset.__len__() + test_dataset.__len__()
# Small batch size or full batch size
batch_size = min([2**7, size_dataset])

loader_train = DataLoader(
    train_dataset + test_dataset,
    batch_size=batch_size,
    num_workers=0,
    drop_last=False,
    shuffle=True,
)

loader = DataLoader(
    train_dataset + test_dataset,
    batch_size=size_dataset,
    num_workers=0,
    drop_last=False,
    shuffle=False,
)
# %%

if model_type == "shallow":

    model = ffNN(
        input_size=1,
        hidden_size=2**q,
        output_size=1,
        hL=1,
        activation=activation,
    )

    W_1 = np.array([2**q for _ in range(2**q)]).reshape(2**q, 1)
    b_1 = np.array([0] + [-k for k in range(1, 2**q)])

    if activation == "relu":
        W_2 = np.array([1] + [(-1) ** k * 2 for k in range(1, 2**q)]).reshape(1, 2**q)
        b_2 = np.array([0])
    elif activation == "prelu":
        slope = model.slope
        W_2 = np.array(
            [(1 + slope) / (1 - slope)]
            + [(-1) ** k * 2 / (1 - slope) for k in range(1, 2**q)]
        ).reshape(1, 2**q)
        b_2 = np.array([-(2**q) * slope / (1 - slope)])
    else:
        raise ValueError("Unknown activation function: {}".format(activation))

    if shift:
        b_1 = b_1 + 2 ** (q - 1)
        b_2 = b_2 - 0.5

    # W_1, b_1, W_2, b_2
    print(W_1.shape, b_1.shape, W_2.shape, b_2.shape)

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

    model = ffNN(
        input_size=1,
        hidden_size=2,
        output_size=1,
        hL=q,
        activation=activation,
    )

    W_1 = np.array([2**1 for _ in range(2**1)]).reshape(2, 1)
    b_1 = np.array([-k for k in range(1, 2**1)] + [0])

    if activation == "relu":
        W_L = np.array([(-1) ** k * 2 for k in range(1, 2**1)] + [1]).reshape(1, 2)
        b_L = np.array([0])
    elif activation == "prelu":
        slope = model.slope
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
model.eval()
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

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_network_activations(model, data_loader):
    """
    Visualizes the network structure and colors neurons by activation counts.
    """
    counts_dict = {}
    # Count active neurons
    for name, param in model.named_parameters():
        if "weight" in name and "fcL" not in name:
            name = name.replace("weight", "").strip(".").replace(".", "_")
            counts_dict[name] = 0
    for x, _ in data_loader:
        if shift:
            x = x - 0.5

        x = x.view(x.shape[0], -1)
        counts = model.count_active_neurons(x)
        for name, count in counts.items():
            if name in counts_dict:
                counts_dict[name] += count

    layer_sizes = [model.fc1.in_features, model.fc1.out_features]
    for z in model.fcj:
        layer_sizes.append(z.out_features)
    layer_sizes.append(model.fcL.out_features)

    # get model weights and biases
    weights = []
    biases = []
    for name, param in model.named_parameters():
        if "weight" in name:
            weights.append(param.data.numpy())
        elif "bias" in name:
            biases.append(param.data.numpy())
    max_weight = max([w.max() for w in weights])
    min_weight = min([w.min() for w in weights])
    max_bias = max([b.max() for b in biases])
    min_bias = min([b.min() for b in biases])
    if min_weight == max_weight:
        min_weight = -1
        max_weight = 1
    if min_bias == max_bias:
        min_bias = -1
        max_bias = 1

    if model_type == "deep":
        fig, ax = plt.subplots(figsize=(10, 4))
        xoffset = 0.1
    elif model_type == "shallow":
        fig, ax = plt.subplots(figsize=(10, 8))
        xoffset = 0.05
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        xoffset = 0.1

    v_spacing = 1.0
    h_spacing = 1.0
    max_layer_size = max(layer_sizes)

    # Find max count for color normalization
    max_count = size_dataset
    cmap = plt.get_cmap("RdYlGn")
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cmapW = plt.get_cmap("jet")
    sm_w = plt.cm.ScalarMappable(
        cmap=cmapW, norm=plt.Normalize(vmin=min_weight, vmax=max_weight)
    )
    sm_w.set_array([])

    cmapB = plt.get_cmap("jet")
    sm_b = plt.cm.ScalarMappable(
        cmap=cmapB, norm=plt.Normalize(vmin=min_bias, vmax=max_bias)
    )

    # Draw neurons
    for layer_idx, layer_size in enumerate(layer_sizes):
        y_offset = (max_layer_size - layer_size) / 2 * v_spacing
        for neuron_idx in range(layer_size):
            if layer_idx == 0:
                color = "lightgray"
                label = "Input" if neuron_idx == 0 else None
            elif layer_idx == len(layer_sizes) - 1:
                color = "lightgray"
                label = "Output" if neuron_idx == 0 else None
            else:
                layer_name = "fc1" if layer_idx == 1 else f"fcj_{layer_idx-2}"
                count = counts_dict[layer_name][neuron_idx]
                norm_count = count / max_count if max_count > 0 else 0
                norm_count = count / max_count if max_count > 0 else 0
                color = cmap(norm_count)
                label = layer_name if neuron_idx == 0 else None
            ax.scatter(
                layer_idx * h_spacing,
                y_offset + neuron_idx * v_spacing,
                s=400,
                color=color,
                edgecolors="black",
                label=label,
                zorder=3,
            )

    for layer_idx in range(len(layer_sizes) - 1):
        prev_size = layer_sizes[layer_idx]
        next_size = layer_sizes[layer_idx + 1]
        y_offset_prev = (max_layer_size - prev_size) / 2 * v_spacing
        y_offset_next = (max_layer_size - next_size) / 2 * v_spacing
        for j in range(next_size):
            colorB = cmapB((biases[layer_idx][j] - min_bias) / (max_bias - min_bias))
            for i in range(prev_size):
                colorW = cmapW(
                    (weights[layer_idx][j, i] - min_weight) / (max_weight - min_weight)
                )
                if (
                    # layer_idx != 0
                    # and layer_idx != len(layer_sizes) - 2
                    prev_size != 1
                    and next_size != 1
                    and (
                        (i == 0 and j == 0)
                        or (i == prev_size - 1 and j == next_size - 1)
                    )
                ):
                    ax.plot(
                        [layer_idx * h_spacing, (layer_idx + 1) * h_spacing],
                        [
                            y_offset_prev + i * v_spacing + xoffset / 8,
                            y_offset_next + j * v_spacing + xoffset / 8,
                        ],
                        "-",
                        zorder=1,
                        color=colorW,
                        linewidth=3,
                    )
                    if i == 0:
                        ax.plot(
                            [layer_idx * h_spacing, (layer_idx + 1) * h_spacing],
                            [
                                y_offset_prev + i * v_spacing - xoffset / 8,
                                y_offset_next + j * v_spacing - xoffset / 8,
                            ],
                            "--",
                            zorder=1,
                            color=colorB,
                            linewidth=3,
                        )
                else:
                    ax.plot(
                        [
                            layer_idx * h_spacing + xoffset / 2,
                            (layer_idx + 1) * h_spacing + xoffset / 2,
                        ],
                        [y_offset_prev + i * v_spacing, y_offset_next + j * v_spacing],
                        "-",
                        zorder=1,
                        color=colorW,
                        linewidth=3,
                    )
                    if i == 0:
                        ax.plot(
                            [
                                layer_idx * h_spacing - xoffset / 2,
                                (layer_idx + 1) * h_spacing - xoffset / 2,
                            ],
                            [
                                y_offset_prev + i * v_spacing,
                                y_offset_next + j * v_spacing,
                            ],
                            "--",
                            zorder=1,
                            color=colorB,
                            linewidth=3,
                        )
    ax.set_xticks([i * h_spacing for i in range(len(layer_sizes))])
    ax.set_xticklabels(
        ["Input"] + [f"Hidden {i+1}" for i in range(len(layer_sizes) - 2)] + ["Output"]
    )
    ax.set_yticks([])

    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("% active", fontsize=12)
    cbar_w = plt.colorbar(
        sm_w,
        ax=ax,
        pad=0.02,
    )  # orientation="horizontal")
    cbar_w.set_label("Weight", fontsize=12)
    cbar_b = plt.colorbar(
        sm_b,
        ax=ax,
        pad=0.02,
    )  # orientation="horizontal")
    cbar_b.set_label("Bias", fontsize=12)

    plt.tight_layout()

    plt.show()


# Usage example after a forward pass:
x_batch, _ = next(iter(loader))
if shift:
    x_batch = x_batch - 0.5

plot_network_activations(model, loader)
# prune dead neurons
model.prune_dead_neurons(x_batch, threshold=0)
plot_network_activations(model, loader)

# %%

model = ffNN(
    input_size=1,
    hidden_size=2**q,
    output_size=1,
    hL=q,
    activation=activation,
)
model.eval()
# init_method = "He"
# init_method = "Shin"
# init_method = "New"
# init_method = "Newnew"
init_method = "kink_maximizing"

if init_method == "Shin":
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
elif init_method == "New" or init_method == "Newnew":
    x = (
        torch.arange(model.fc1.out_features, dtype=torch.float32).reshape(-1, 1)
        / model.fc1.out_features
    )
    # radnom x between 0 and 1
    # x = torch.rand(model.fc1.out_features, 1, dtype=torch.float32)
    # print(x)

    if shift:
        x = x - 0.5
    args = {"x": x, "X": x_batch}
elif init_method == "kink_maximizing":
    args = {"X": x_batch}
else:
    args = None


print("Model weights:")
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
print()

model.initialize_weights(method=init_method, args=args)
model.plot_z(x_batch)
# %%
# do not train activation parameters
for name, param in model.named_parameters():
    if "activation" in name:
        param.requires_grad = False
        print(f"Setting {name} to not require grad")

print("Model weights:")
for name, param in model.named_parameters():
    # print(name, param.data, param.shape)
    print(name, param.shape, param.requires_grad)

plot_network_activations(model, loader)


print("Model weights:")
for name, param in model.named_parameters():
    # print(name, param.data, param.shape)
    print(name, param.shape, param.requires_grad)
print()

print("Number of neurons in each layer:")
N = 1
for name, param in model.named_parameters():
    if "weight" in name:
        N *= param.shape[0]
print(f"current:{N} min:{2**q}")


model.prune_dead_neurons(x_batch, threshold=0)
model.plot_z(x_batch)
counts_dict = model.count_active_neurons(x_batch)
plot_network_activations(model, loader)

# print the product of the the number of neurons in each layer
print("Number of neurons in each layer:")
N = 1
for name, param in model.named_parameters():
    if "weight" in name:
        N *= param.shape[0]
print(f"current:{N} min:{2**q}")

# W = []
# weights = []
# w_name = []
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         weights.extend(param.data.numpy().flatten().tolist())
#         w_name += [name + f"_{i}" for i in range(len(param.data.numpy().flatten()))]

# W.append(weights)

# G = []


criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=0.01,
#     momentum=0.0,
# )


loss_threshold = 1e-8 * size_dataset  # 1e-6 * size_dataset
stop_condition = False


fig_model, ax_model = plt.subplots(1, 1, figsize=(10, 5))
ax_model.set_xlabel("x", fontsize=14)
ax_model.set_ylabel(f"$f^q(x)$", fontsize=14)

Y_hat_0 = []
X = []
Y = []
for i, (x, y) in enumerate(loader_train):
    if shift:
        x = x - 0.5
        y = y - 0.5
    with torch.no_grad():
        y_pred = model(x)
        Y_hat_0.append(y_pred.detach().numpy().flatten())
        Y.append(y.detach().numpy().flatten())
        X.append(x.detach().numpy().flatten())

X = np.concatenate(X)
Y = np.concatenate(Y)
Y_hat_0 = np.concatenate(Y_hat_0)

# ax_model.plot(X, Y_hat_0, ".", label=f"initial prediction")
ax_model.plot(X, Y, "o", label=f"true", markersize=2, markerfacecolor="none")

# ax_model.legend(fontsize=10)
# ax_model.set_aspect("equal")
# fig_model.tight_layout()
# fig_model

# %%

# # set all layers except the last one to not require grad
# for name, param in model.named_parameters():
#     if "fcL" not in name:
#         param.requires_grad = False
#         print(f"Setting {name} to not require grad")
#     else:
#         print(f"Keeping {name} to require grad")
#         # param.requires_grad = True


# %%
model.train()

offset = np.mean(Y_hat_0)
model.fcL.bias = torch.nn.Parameter(
    torch.tensor(model.fcL.bias.data.numpy() - offset, dtype=torch.float32)
)  # shift the bias to match the initial prediction

X_hat = []
Y_hat_1 = []
for i, (x, y) in enumerate(loader_train):
    if shift:
        x = x - 0.5
        y = y - 0.5
    with torch.no_grad():
        y_pred = model(x)
        Y_hat_1.append(y_pred.detach().numpy().flatten())
        X_hat.append(x.detach().numpy().flatten())

Y_hat_1 = np.concatenate(Y_hat_1)
X_hat = np.concatenate(X_hat)
ax_model.plot(X_hat, Y_hat_1, ".", label=f"initial prediction")
ax_model.legend(fontsize=10)
fig_model.tight_layout()
fig_model

# %%
# train all layers
for name, param in model.named_parameters():
    param.requires_grad = True


params = {
    "lr": 0.001,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
}
optimizer = torch.optim.Adam(model.parameters(), **params)
# optimizer = SaddleFreeNewton(
#     model.parameters(),
#     lr=0.003,
# )


for epoch in range(2000):
    loss_acc = 0
    for i, (x, y) in enumerate(loader_train):

        if shift:
            x = x - 0.5
            y = y - 0.5

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()

        optimizer.step()
        loss_acc += loss.item()

    if loss_acc < loss_threshold:
        stop_condition = True
        break

    if stop_condition:
        print(f"Epoch {epoch}, Batch {i}, Loss: {loss_acc:.2e}")
        print(f"current:{N} min:{2**q}")
        print(
            f"Stopping training at epoch {epoch} due to convergence. Current loss: {loss_acc:.2e}"
        )
        break

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Batch {i}, Loss: {loss_acc:.2e}")
        print(f"current:{N} min:{2**q}")


# print model weights
print("Model weights:")
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)

Y_hat = []
X_hat = []
for i, (x, y) in enumerate(loader_train):

    if shift:
        x = x - 0.5
        y = y - 0.5

    y_pred = model(x)
    Y_hat.append(y_pred.detach().numpy().flatten())
    X_hat.append(x.detach().numpy().flatten())


Y_hat = np.concatenate(Y_hat)
X_hat = np.concatenate(X_hat)

ax_model.plot(X_hat, Y_hat, ".", label=f"final prediction")

ax_model.legend(fontsize=10)
ax_model.set_aspect("equal")
fig_model.tight_layout()

fig_model

# %%

model.eval()
plot_network_activations(model, loader)

print("Number of neurons in each layer:")
N = 1
for name, param in model.named_parameters():
    if "weight" in name:
        N *= param.shape[0]
print(f"current:{N} min:{2**q}")


model.prune_dead_neurons(x_batch, threshold=0)
counts_dict = model.count_active_neurons(x_batch)
plot_network_activations(model, loader)

# print the product of the the number of neurons in each layer
print("Number of neurons in each layer:")
N = 1
for name, param in model.named_parameters():
    if "weight" in name:
        N *= param.shape[0]
print(f"current:{N} min:{2**q}")

Y_hat = []
X_hat = []
for i, (x, y) in enumerate(loader_train):

    if shift:
        x = x - 0.5
        y = y - 0.5

    y_pred = model(x)
    Y_hat.append(y_pred.detach().numpy().flatten())
    X_hat.append(x.detach().numpy().flatten())


Y_hat = np.concatenate(Y_hat)
X_hat = np.concatenate(X_hat)

ax_model.plot(X_hat, Y_hat, ".", label=f"pruned prediction")
ax_model.set_xlabel("x", fontsize=14)
ax_model.set_ylabel(f"$f^n(x)$", fontsize=14)
ax_model.legend(fontsize=10)
ax_model.set_aspect("equal")

fig_model.tight_layout()
fig_model

# %%
x_batch, y_batch = next(iter(loader))
if shift:
    x_batch = x_batch - 0.5
    y_batch = y_batch - 0.5

hidden_size = []
for name, param in model.named_parameters():
    if "weight" in name and "fcL" not in name:
        hidden_size.append(param.shape[0])

model_new = ffNN(
    input_size=1,
    hidden_size=hidden_size,
    output_size=1,
    hL=len(hidden_size),
    activation=activation,
)
# set wieghts and biases using the current model
with torch.no_grad():
    model_new.fc1.weight = model.fc1.weight
    model_new.fc1.bias = model.fc1.bias
    for i, layer in enumerate(model.fcj):
        if layer is not None:
            model_new.fcj[i].weight = layer.weight
            model_new.fcj[i].bias = layer.bias
    model_new.fcL.weight = model.fcL.weight
    model_new.fcL.bias = model.fcL.bias

y_pred = model_new(x_batch)
criterion = torch.nn.MSELoss()
loss = criterion(y_pred, y_batch)

hessian = model_new.compute_hessian(loss)
# # Flatten parameters
# params = [p for p in model_new.parameters() if p.requires_grad]
# flat_params = torch.cat([p.contiguous().view(-1) for p in params])

# # First-order gradient
# grad = torch.autograd.grad(loss, params, create_graph=True)
# flat_grad = torch.cat([g.contiguous().view(-1) for g in grad])

# # Hessian matrix
# hessian = []
# for g in flat_grad:
#     second_grads = torch.autograd.grad(g, params, retain_graph=True)
#     h_row = torch.cat([sg.contiguous().view(-1) for sg in second_grads])
#     hessian.append(h_row)

# hessian = torch.stack(hessian)  # shape: [n_params, n_params]
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

# print the number of positive, negative, and zero eigenvalues
print(f"Number of positive eigenvalues: {len(positive_eigenvalues)}")
print(f"Number of negative eigenvalues: {len(negative_eigenvalues)}")
print(f"Number of zero eigenvalues: {len(zero_eigenvalues)}")

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
