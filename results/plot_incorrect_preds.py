# %%

import numpy as np

data = np.load("./incorrect_preds.npy")
data1 = np.load("./train incorrect_preds.npy")

print(len(data) + len(data1), f"{2**22:.2e}")
print(f"{(len(data)+len(data1))/2**22:.2e}")

# %%

x = [d[0] for d in data]
y = [d[2] for d in data]
y_hat = [d[1] for d in data]

diff = [np.sum(np.abs(t - p)) for t, p in zip(y, y_hat)]

# counts of values in diff
counts = np.bincount(diff)
print(counts)


f = lambda x: [np.sum([2 ** (-i - 1) * d[i] for i in range(len(d))]) for d in x]

x = f(x)
y = f(y)
y_hat = f(y_hat)

# %%

tent_map = lambda x: 1 - 2 * np.abs(x - 0.5)

q = 4
X = np.linspace(0, 1, 1000)
Y = tent_map(X)
for i in range(q - 1):
    Y = tent_map(Y)
# %%

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
for l, t, p, d in zip(x, y, y_hat, diff):
    ax.plot([l, l], [t, p], "k-")
    # add diff as text
    ax.text(l, (t + p) / 2, str(d), fontsize=18)
ax.plot(x, y_hat, "o", label="Predicted")
ax.plot(X, Y, "-", label="Tent map")

# %%
