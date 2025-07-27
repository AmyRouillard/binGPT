# %%

from utils.tentmapdataset import TentDataset
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import time
import plotly.graph_objects as go

# %%

w_01, w_02 = 1.5, -1.3
b_01, b_02 = -0.2, 1.5
w_10, w_11 = w_02, -w_01
b_1 = 2.0


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


n = 1
length = 6 + n
n_epochs = 1000


data_type = "decimal"
tokenized = False

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


w = np.linspace(-3, 3, 500)
b = np.linspace(-3, 3, 500)
# x = np.linspace(0, 1, 500)


# check when w*x + b > 0
def condition(w, x, b):
    # return (w * x + b > 0 - 1e-5) & (w * x + b < 1 + 1e-5)
    return w * x + b > 0


W, B = np.meshgrid(w, b)

mask = np.full_like(W, 0)
count = 0
for i, (x, y) in enumerate(loader):
    x = x.numpy().flatten()
    count += len(x)
    for x_val in x:
        mask += condition(W, x_val, B)
mask /= count

fig = go.Figure()
# add heatmap
fig.add_trace(
    go.Heatmap(
        z=mask,
        x=w,
        y=b,
        colorscale="Viridis",
        zmin=0,
        zmax=1,
        showscale=False,
    )
)
fig.add_trace(
    go.Scatter(
        x=np.linspace(-3, 3, 10),
        y=-0.5 * np.linspace(-3, 3, 10),
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="b = -0.5 * w",
        showlegend=False,
    )
)


model = MLP(
    input_size=1,
    hidden_size=2,
    output_size=1,
    repeat=n,
)

for i, (x, y) in enumerate(loader):
    y_pred = model(x)

fig2 = go.Figure()
# plot x, y and x, y_pred
fig2.add_trace(
    go.Scatter(
        x=x.detach().numpy().flatten(),
        y=y.detach().numpy().flatten(),
        mode="markers",
        name="True",
        marker=dict(color="blue"),
    )
)
fig2.add_trace(
    go.Scatter(
        x=x.detach().numpy().flatten(),
        y=y_pred.detach().numpy().flatten(),
        mode="markers",
        name="Predicted initial",
        marker=dict(color="orange"),
    )
)

model.fc1.weight.data = torch.tensor([[w_01], [w_02]])
model.fc1.bias.data = torch.tensor([b_01, b_02])
model.fc2.weight.data = torch.tensor([[w_10, w_11]])
model.fc2.bias.data = torch.tensor([b_1])

# model.fc1.weight.data = torch.tensor([[w_01]])
# model.fc1.bias.data = torch.tensor([b_01])
# model.fc2.weight.data = torch.tensor([[w_10]])
# model.fc2.bias.data = torch.tensor([b_1])

W = []
weights = []
for name, param in model.named_parameters():
    if param.requires_grad:
        weights.extend(param.data.numpy().flatten().tolist())

W.append(weights)

G = []

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=0,
)

for epoch in range(n_epochs):

    for i, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)

        loss.backward()

        optimizer.step()

        w = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                w.extend(param.grad.data.numpy().flatten().tolist())
        G.append(w)

        weights = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights.extend(param.data.numpy().flatten().tolist())

        W.append(weights)


# plot W_[:, 0], W_[:, 2] on fig
fig.add_trace(
    go.Scatter(
        x=[w[0] for w in W],
        y=[w[2] for w in W],
        mode="lines+markers",
        # name="W_[:, 0] vs W_[:, 2]",
        line=dict(color="blue"),
        showlegend=False,
    )
)

# plot W_[:, 1], W_[:, 3] on fig
fig.add_trace(
    go.Scatter(
        x=[w[1] for w in W],
        y=[w[3] for w in W],
        mode="lines+markers",
        # name="W_[:, 1] vs W_[:, 3]",
        line=dict(color="green"),
        # don't add to legend
        showlegend=False,
    )
)

# don't show colorbar
fig.update_layout(
    title="MLP Training Visualization",
    xaxis_title="w",
    yaxis_title="b",
    width=600,
    height=500,
)

fig2.add_trace(
    go.Scatter(
        x=x.detach().numpy().flatten(),
        y=y_pred.detach().numpy().flatten(),
        mode="markers",
        name="Predicted finial",
        marker=dict(color="red"),
    )
)

# %%

fig

# %%

fig2
# %%

for name, param in model.named_parameters():
    print(name, param.data.numpy().flatten().tolist())
# %%
