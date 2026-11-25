import numpy as np
import matplotlib.pyplot as plt
import fit

x, y = fit.load("data.csv")
(xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)

sizes = [0, 1, 2, 3, 5, 10, 20, 50, 100]
x_vals = [0.5 if h == 0 else h for h in sizes]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

# run with seed 0
tr0, va0 = [], []
for h in sizes:
    net, hist = fit.train(xtr, ytr, hidden=h, epochs=20000, seed=0)
    tr_loss = hist["train"][-1]
    va_loss = fit.mse(net, xva, yva)
    tr0.append(tr_loss)
    va0.append(va_loss)
    print(f"seed=0 H={h}: train={tr_loss:.4f}, val={va_loss:.4f}")

ax0.plot(x_vals, tr0, "o-", label="train")
ax0.plot(x_vals, va0, "o-", label="val")
ax0.set_xscale("log")
ax0.set_xticks(x_vals)
ax0.set_xticklabels([str(h) for h in sizes])
ax0.set_xlabel("H (H=0 at 0.5)")
ax0.set_ylabel("MSE")
ax0.set_title("seed = 0")
ax0.legend()

# repeat with seed 1
tr1, va1 = [], []
for h in sizes:
    net, hist = fit.train(xtr, ytr, hidden=h, epochs=20000, seed=1)
    tr_loss = hist["train"][-1]
    va_loss = fit.mse(net, xva, yva)
    tr1.append(tr_loss)
    va1.append(va_loss)
    print(f"seed=1 H={h}: train={tr_loss:.4f}, val={va_loss:.4f}")

ax1.plot(x_vals, tr1, "o-", label="train")
ax1.plot(x_vals, va1, "o-", label="val")
ax1.set_xscale("log")
ax1.set_xticks(x_vals)
ax1.set_xticklabels([str(h) for h in sizes])
ax1.set_xlabel("H (H=0 at 0.5)")
ax1.set_title("seed = 1")
ax1.legend()

plt.tight_layout()
plt.savefig("exercise2.png", dpi=120)
plt.show()