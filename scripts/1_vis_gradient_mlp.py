# %%


import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from utils.tentmapdataset import TentDataset
from torch.utils.data.dataloader import DataLoader
import torch


def E_sym(y_target, y_pred):
    return (y_target - y_pred) ** 2


def reLU_sym(x):
    # SymPy's Abs is fine here as it's part of the network
    # return (x + sp.Abs(x)) / 2
    return x / 2 + sp.Abs(x) / 2
    # A slightly more robust symbolic reLU if you encounter issues with derivatives at 0
    # return sp.Piecewise((0, x < 0), (x, x >= 0))


def preLU_sym(x, alpha=0.2):
    # SymPy's Abs is fine here as it's part of the network
    return (1 + alpha) / 2 * x + sp.Abs((1 - alpha) / 2 * x)
    # A slightly more robust symbolic reLU if you encounter issues with derivatives at 0
    # return sp.Piecewise((0, x < 0), (x, x >= 0))


def forward_sym(W_sym, b_sym, Wp_sym, bp_sym, x_input_sym):
    # Ensure x_input_sym is a Matrix for @ operator
    if not isinstance(x_input_sym, sp.MatrixBase):
        x_input_sym = sp.Matrix([x_input_sym])

    h = W_sym @ x_input_sym + b_sym
    h_activated = sp.Matrix([preLU_sym(el) for el in h])  # Apply reLU element-wise
    y_hat_sym = Wp_sym @ h_activated + bp_sym
    return y_hat_sym[0]  # Return scalar expression


# --- Define Symbolic Variables ---
y_target_sym = sp.symbols("y_target", real=True)  # Placeholder for numerical target
x_sym = sp.symbols("x_input", real=True)  # Symbolic input to the network

# Parameters (flattened for easier handling with lambdify)
a00, a10 = sp.symbols("a00 a10", real=True)
b00, b10 = sp.symbols("b00 b10", real=True)
ap00, ap01 = sp.symbols("ap00 ap01", real=True)  # Wp has shape (1,2)
bp00 = sp.symbols("bp00", real=True)

params_sym = [a00, a10, b00, b10, ap00, ap01, bp00]

# Reconstruct matrices for the forward pass
W_s = sp.Matrix([[a00], [a10]])
b_s = sp.Matrix([[b00], [b10]])
Wp_s = sp.Matrix([[ap00, ap01]])
bp_s = sp.Matrix([[bp00]])

# --- Symbolic Forward Pass and Loss ---
y_hat_expr = forward_sym(W_s, b_s, Wp_s, bp_s, x_sym)
loss_expr = E_sym(y_target_sym, y_hat_expr)

print("Symbolic y_hat expression:", y_hat_expr)
print("Symbolic Loss expression:", loss_expr)

# --- Symbolic Gradients ---
grad_exprs = [sp.diff(loss_expr, p) for p in params_sym]

# --- Lambdify ---
# Arguments for lambdified functions: (y_target, x_input, p1, p2, ..., p7)
# 'numpy' backend for speed
loss_fn_lambdified = sp.lambdify([y_target_sym, x_sym] + params_sym, loss_expr, "numpy")
grad_fn_lambdified = sp.lambdify(
    [y_target_sym, x_sym] + params_sym, grad_exprs, "numpy"
)

# --- Numerical Evaluation ---


def get_data_loader(n):
    data_type = "decimal"
    tokenized = False
    length = 6 + n

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
    batch_size = min([2**10, train_dataset.__len__() + test_dataset.__len__()])

    loader = DataLoader(
        train_dataset + test_dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
        shuffle=True,
        # shuffle=False,
    )

    return loader


# Function to compute total loss and gradients over the dataset for given parameters
def calculate_total_loss_and_grad(param_values_arr, X_d, Y_true_d):
    total_loss = 0
    # Initialize total_grad as a NumPy array of appropriate size
    total_grad = np.zeros(len(param_values_arr), dtype=float)

    for x_val, y_true_val in zip(X_d, Y_true_d):
        args = [y_true_val, x_val] + list(param_values_arr)
        total_loss += loss_fn_lambdified(*args)
        grads_for_x = grad_fn_lambdified(
            *args
        )  # This will be a list/tuple of gradients
        total_grad += np.array(grads_for_x)
    return total_loss / len(X_d), total_grad / len(X_d)  # Average loss and grad


# print("\nGenerating loss landscape (2D slice)...")
def get_surface(
    selected_1,
    selected_2,
    fixed_params,
    n_grid,
    p1,
    p2,
    X_data,
    Y_true_data,
):
    loss_surface = np.zeros((n_grid, n_grid))
    grad_surface_p1 = np.zeros((n_grid, n_grid))  # For dL/da00
    grad_surface_p2 = np.zeros((n_grid, n_grid))  # For dL/dap00

    tmp = np.array(
        [
            fixed_params[a00],
            fixed_params[a10],
            fixed_params[b00],
            fixed_params[b10],
            fixed_params[ap00],
            fixed_params[ap01],
            fixed_params[bp00],
        ]
    )

    for i, p1_val in enumerate(p1):
        for j, p2_val in enumerate(p2):
            # Update the parameters we are varying
            tmp[selected_1] = p1_val  # a00
            tmp[selected_2] = p2_val  # ap00

            loss, grad_vector = calculate_total_loss_and_grad(tmp, X_data, Y_true_data)
            loss_surface[i, j] = loss
            grad_surface_p1[i, j] = -grad_vector[selected_1]  # Grad for a00
            grad_surface_p2[i, j] = -grad_vector[selected_2]  # Grad for ap00
        # print(f"Progress: { (i+1)/N_GRID * 100 :.1f}%")

    return loss_surface, grad_surface_p1, grad_surface_p2


def update_params(params, local_grad_vector, alpha):
    return params - alpha * local_grad_vector


# %%
opt = [1.0, -1.0, -0.5, 0.5, -2.0 / (1 - 0.2), -2.0 / (1 - 0.2), 1.0]

n = 1  # forward not currently implemented for n > 1

params = torch.randn(7).numpy().flatten()
# params = np.array(opt)

# enforce params < 2 and > -2
# params = np.clip(params, -2, 2)

selected_1 = 0  # Indices of parameters to vary (a00 and ap00)

N_GRID = 41  # Resolution of the landscape plot

loader = get_data_loader(n)


alpha = 0.05

# %%

# get first batch
x, y = next(iter(loader))
X_data = x.detach().numpy().flatten()
Y_true_data = y.detach().numpy().flatten()

# Fixed parameter values (example)
fixed_params = {
    a00: params[0],
    a10: params[1],
    b00: params[2],
    b10: params[3],
    ap00: params[4],
    ap01: params[5],
    bp00: params[6],
}

tmp = np.array(
    [
        fixed_params[a00],
        fixed_params[a10],
        fixed_params[b00],
        fixed_params[b10],
        fixed_params[ap00],
        fixed_params[ap01],
        fixed_params[bp00],
    ]
)
local_loss, local_grad_vector = calculate_total_loss_and_grad(tmp, X_data, Y_true_data)

# print loss and magnitude of gradient
print(f"Local Loss: {local_loss:.4f}")
print(f"Local Gradient Magnitude: {np.linalg.norm(local_grad_vector):.4f}")

# %%

param1_vals = np.linspace(
    params[selected_1] - 2, params[selected_1] + 2, N_GRID
)  # Range for a00

FIG = []
for selected_2 in range(1, 7):

    param2_vals = np.linspace(
        params[selected_2] - 2, params[selected_2] + 2, N_GRID
    )  # Range for ap00
    P1, P2 = np.meshgrid(param1_vals, param2_vals)  # P1 corresponds to a00, P2 to ap00

    loss_surface, grad_surface_p1, grad_surface_p2 = get_surface(
        selected_1=selected_1,
        selected_2=selected_2,
        fixed_params=fixed_params,
        n_grid=N_GRID,
        p1=param1_vals,
        p2=param2_vals,
        X_data=X_data,
        Y_true_data=Y_true_data,
    )

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    colorbar = ax[0].contourf(
        P1, P2, np.log10(loss_surface.T), levels=50, cmap="viridis"
    )
    fig.colorbar(colorbar, ax=ax[0], label="Log Loss")
    ax[0].set_xlabel(f"Parameter: {params_sym[selected_1].name}")
    ax[0].set_ylabel(f"Parameter: {params_sym[selected_2].name}")
    # plot the current parameter point
    current_point = np.array(
        [
            fixed_params[params_sym[selected_1]],  # a00
            fixed_params[params_sym[selected_2]],  # ap00
        ]
    )
    ax[0].plot(
        current_point[0],
        current_point[1],
        "ro",
        markersize=8,
        label="Current Params",
        fillstyle="none",
    )
    ax[0].quiver(
        current_point[0],
        current_point[1],
        -local_grad_vector[selected_1],
        -local_grad_vector[selected_2],
        color="red",
        scale=None,
        headwidth=4,
        width=0.003,
    )
    # find index of P1 and P2 closest to current_point
    idx_p1 = np.argmin(np.abs(param1_vals - opt[selected_1]))
    idx_p2 = np.argmin(np.abs(param2_vals - opt[selected_2]))
    ax[0].plot(
        param1_vals[idx_p1],
        param2_vals[idx_p2],
        "go",
        markersize=8,
        label="Opt Params",
        fillstyle="none",
    )

    # Normalize gradients for better visualization if magnitudes vary wildly (optional)
    # For quiver, U is horizontal component (grad_surface_p1), V is vertical (grad_surface_p2)
    # Need to transpose because of how meshgrid and indexing worked out
    ax[1].quiver(
        P1,
        P2,
        grad_surface_p1.T,
        grad_surface_p2.T,
        color="red",
        scale=None,
        headwidth=4,
        width=0.003,
    )
    ax[1].set_xlim(param1_vals.min(), param1_vals.max())
    ax[1].set_ylim(param2_vals.min(), param2_vals.max())
    ax[1].set_xlabel(f"Parameter: {params_sym[selected_1].name}")
    ax[1].set_ylabel(f"Parameter: {params_sym[selected_2].name}")
    ax[1].plot(
        current_point[0],
        current_point[1],
        "ro",
        markersize=8,
        label="Current Params",
        fillstyle="none",
    )

    ax[1].plot(
        param1_vals[idx_p1],
        param2_vals[idx_p2],
        "go",
        markersize=8,
        label="Opt Params",
        fillstyle="none",
    )

    FIG.append(fig)

# %%

# update params with local gradient
params = update_params(
    params,
    local_grad_vector,
    alpha,
)

# get first batch
x, y = next(iter(loader))
X_data = x.detach().numpy().flatten()
Y_true_data = y.detach().numpy().flatten()

# Fixed parameter values (example)
fixed_params = {
    a00: params[0],
    a10: params[1],
    b00: params[2],
    b10: params[3],
    ap00: params[4],
    ap01: params[5],
    bp00: params[6],
}

tmp = np.array(
    [
        fixed_params[a00],
        fixed_params[a10],
        fixed_params[b00],
        fixed_params[b10],
        fixed_params[ap00],
        fixed_params[ap01],
        fixed_params[bp00],
    ]
)
local_loss, local_grad_vector = calculate_total_loss_and_grad(tmp, X_data, Y_true_data)

# print loss and magnitude of gradient
print(f"Local Loss: {local_loss:.4f}")
print(f"Local Gradient Magnitude: {np.linalg.norm(local_grad_vector):.4f}")

# %%
