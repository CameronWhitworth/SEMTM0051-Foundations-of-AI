"""
exercise1.py -- a worked example to get you started.
"""
import numpy as np
import matplotlib.pyplot as plt
import fit

x, y = fit.load("data.csv")
xgrid = np.linspace(-2, 2, 400)            # where we evaluate the fitted curve

sizes = [0, 1, 2, 5, 20, 100]
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
for ax, H in zip(axes.ravel(), sizes):
    net, history = fit.train(x, y, hidden=H, epochs=20000)
    ax.plot(x, y, "o", ms=4)
    ax.plot(xgrid, fit.predict(net, xgrid), "r")
    ax.set_ylim(-2.5, 3)
    ax.set_title(f"H = {H}: {fit.n_params(net)} parameters, "
                 f"train MSE = {history['train'][-1]:.3f}")
# label only the outer axes: with sharex/sharey the inner tick labels are
# already hidden, so the figure reads as one grid rather than six plots
for ax in axes[-1, :]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("y")
    print(f"H = {H:4d}  parameters = {fit.n_params(net):4d}  "
          f"train MSE = {history['train'][-1]:.3f}")
plt.tight_layout()
plt.savefig("exercise1.png", dpi=120)
plt.show()





"""
exercise2.py
"""
import numpy as np
import matplotlib.pyplot as plt
import fit

x, y = fit.load("data.csv")
(xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)

sizes = [0, 1, 2, 3, 5, 10, 20, 50, 100]
x_vals = [0.5 if h == 0 else h for h in sizes]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

# seed 0
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

# 1
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



"""
exercise3.py
"""
import numpy as np
import matplotlib.pyplot as plt
import fit

x, y = fit.load("data.csv")
(xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)

net_full, hist = fit.train(xtr, ytr, hidden=20, epochs=20000, xval=xva, yval=yva)

tr_hist = hist["train"]
va_hist = hist["val"]
epochs = np.arange(1, len(tr_hist) + 1)

best_idx = int(np.argmin(va_hist))
best_epoch = best_idx + 1

print("Best epoch:", best_epoch)
print("At best epoch -> train:", round(tr_hist[best_idx], 4), "val:", round(va_hist[best_idx], 4))
print("At 20000 -> train:", round(tr_hist[-1], 4), "val:", round(va_hist[-1], 4))

net_early, _ = fit.train(xtr, ytr, hidden=20, epochs=best_epoch)
xgrid = np.linspace(-2, 2, 400)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.loglog(epochs, tr_hist, label="train")
ax1.loglog(epochs, va_hist, label="val")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")
ax1.set_title("Loss vs epoch (H = 20)")
ax1.legend()

ax2.plot(xtr, ytr, "o", ms=5, label="train data")
ax2.plot(xgrid, fit.predict(net_full, xgrid), label="20,000 epochs")
ax2.plot(xgrid, fit.predict(net_early, xgrid), label=f"early stop ({best_epoch})")
ax2.set_ylim(-2.5, 3)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("End of training vs early stopping")
ax2.legend()

plt.tight_layout()
plt.savefig("exercise3.png", dpi=120)
plt.show()





"""
exercise4.py
"""
import numpy as np
import matplotlib.pyplot as plt
import fit

x, y = fit.load("data.csv")
(xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)

wds = [0, 1e-4, 1e-3, 1e-2, 1e-1]
tr_losses = []
va_losses = []

for wd in wds:
    net, hist = fit.train(xtr, ytr, hidden=20, epochs=20000, weight_decay=wd)
    tr_loss = hist["train"][-1]
    va_loss = fit.mse(net, xva, yva)
    tr_losses.append(tr_loss)
    va_losses.append(va_loss)
    print(f"wd={wd}: train={tr_loss:.4f}, val={va_loss:.4f}")

x_plot = [1e-5 if w == 0 else w for w in wds]

plt.figure(figsize=(7, 4))
plt.plot(x_plot, tr_losses, "o-", label="train")
plt.plot(x_plot, va_losses, "o-", label="val")
plt.xscale("log")
plt.xticks(x_plot, ["0" if w == 0 else str(w) for w in wds])
plt.xlabel("weight decay (0 plotted at 1e-5)")
plt.ylabel("MSE")
plt.title("Weight decay sweep (H = 20)")
plt.legend()
plt.tight_layout()
plt.savefig("exercise4.png", dpi=120)
plt.show()


"""
exercise5.py
"""
import numpy as np
import fit

x, y = fit.load("data.csv")
(xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)

# final model chosen from exercises 2-4: H=20, weight_decay=1e-3, 20000 epochs
net_final, hist_final = fit.train(
    xtr, ytr, hidden=20, epochs=20000, weight_decay=1e-3, xval=xva, yval=yva
)

tr_loss = hist_final["train"][-1]
va_loss = hist_final["val"][-1]
# evaluate on the test set once at the very end
te_loss = fit.mse(net_final, xte, yte)

print(f"Final model (H=20, wd=1e-3, 20000 epochs):")
print(f"train MSE: {tr_loss:.4f}")
print(f"val MSE:   {va_loss:.4f}")
print(f"test MSE:  {te_loss:.4f}")



"""
exercise6.py
"""
import numpy as np
import fit
import make_data

x, y = fit.load("data.csv")
(xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)

rng = np.random.default_rng(123)
xbig, ybig = make_data.make(10000, rng)

# (a) final model from Exercise 5: H=20, wd=1e-3, 20000 epochs
net_final, _ = fit.train(xtr, ytr, hidden=20, epochs=20000, weight_decay=1e-3)
final_big = fit.mse(net_final, xbig, ybig)
final_te = fit.mse(net_final, xte, yte)

# (b) true function error on fresh data (estimates sigma^2)
true_mse = np.mean((make_data.true_function(xbig) - ybig) ** 2)

# (c) unregularised H=20, wd=0, 20000 epochs from Exercise 3
net_overfit, hist_overfit = fit.train(xtr, ytr, hidden=20, epochs=20000, weight_decay=0.0, xval=xva, yval=yva)
overfit_big = fit.mse(net_overfit, xbig, ybig)
overfit_te = fit.mse(net_overfit, xte, yte)

print(f"Final model -> val: 0.1533, test: {final_te:.4f}, 10k true error: {final_big:.4f}")
print(f"True function MSE on 10k points (sigma^2 estimate): {true_mse:.4f}")
print(f"Overfit H=20 -> val: {hist_overfit['val'][-1]:.4f}, test: {overfit_te:.4f}, 10k true error: {overfit_big:.4f}")






"""
exercise7.py
"""
import numpy as np
import matplotlib.pyplot as plt
import fit

x, y = fit.load("data.csv")
(xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)

runs = [
    ("sgd", 0.01),
    ("sgd", 0.1),
    ("sgd", 0.3),
    ("sgd", 1.0),
    ("adam", 0.01),
    ("adam", 0.001),
]

plt.figure(figsize=(7.5, 4.5))

for opt, lr in runs:
    net, hist = fit.train(xtr, ytr, hidden=20, epochs=20000, lr=lr, optimiser=opt, xval=xva, yval=yva)
    tr = np.array(hist["train"], dtype=float)
    va = np.array(hist["val"], dtype=float)
    epochs = np.arange(1, len(tr) + 1)

    below = np.where(tr < 0.2)[0]
    first_epoch = int(below[0] + 1) if len(below) > 0 else None
    print(f"{opt} lr={lr}: first <0.2 at epoch {first_epoch} | final train={tr[-1]:.4f}, final val={va[-1]:.4f}")

    # log-log needs positive values
    tr_safe = np.where(np.isfinite(tr) & (tr > 0), tr, np.nan)
    plt.loglog(epochs, tr_safe, label=f"{opt} lr={lr}")

plt.axhline(0.2, linestyle="--", linewidth=1, label="0.2 threshold")
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Training loss: SGD vs Adam (H = 20)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("exercise7.png", dpi=120)
plt.show()





"""
exercise8.py
"""
import numpy as np
import matplotlib.pyplot as plt
import fit

x, y = fit.load("football_data.csv")
(xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)

# 1. Capacity sweep across H
hs = [0, 1, 2, 3, 5, 10, 20, 50, 100]
tr_mses = []
va_mses = []

for h in hs:
    net, _ = fit.train(xtr, ytr, hidden=h, epochs=20000, seed=0)
    tr_val = fit.mse(net, xtr, ytr)
    va_val = fit.mse(net, xva, yva)
    tr_mses.append(tr_val)
    va_mses.append(va_val)
    print(f"H={h:3d} -> train MSE: {tr_val:.4f}, val MSE: {va_val:.4f}")

best_h = hs[int(np.argmin(va_mses))]
print(f"\nBest H by validation MSE: {best_h} (val MSE = {min(va_mses):.4f})")

# Plot capacity sweep (H=0 plotted at 0.5 on log scale)
x_plot = [0.5 if h == 0 else h for h in hs]
plt.figure(figsize=(7.2, 4.3))
plt.plot(x_plot, tr_mses, marker="o", label="Train MSE")
plt.plot(x_plot, va_mses, marker="s", label="Validation MSE")
plt.xscale("log")
plt.xticks(x_plot, [str(h) for h in hs])
plt.xlabel("Hidden units H (0 plotted at 0.5)")
plt.ylabel("MSE")
plt.title("Football Data: Capacity Sweep (30 train / 30 val)")
plt.legend()
plt.tight_layout()
plt.savefig("exercise8_capacity.png", dpi=120)
plt.close()

# 2. Early stopping with H=20
net_20, hist_20 = fit.train(xtr, ytr, hidden=20, epochs=20000, xval=xva, yval=yva, seed=0)
val_hist = np.array(hist_20["val"], dtype=float)
tr_hist = np.array(hist_20["train"], dtype=float)
best_epoch = int(np.argmin(val_hist)) + 1

print(f"\nH=20 best val epoch: {best_epoch}")
print(f"  At best epoch -> train MSE: {tr_hist[best_epoch - 1]:.4f}, val MSE: {val_hist[best_epoch - 1]:.4f}")
print(f"  At 20000 ep   -> train MSE: {tr_hist[-1]:.4f}, val MSE: {val_hist[-1]:.4f}")

net_es, _ = fit.train(xtr, ytr, hidden=20, epochs=best_epoch, seed=0)

x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)
y_grid_20 = fit.predict(net_20, x_grid)
y_grid_es = fit.predict(net_es, x_grid)

plt.figure(figsize=(7.2, 4.3))
plt.scatter(xtr, ytr, alpha=0.75, label="Training points")
plt.plot(x_grid, y_grid_20, label="H=20 full (20,000 epochs)")
plt.plot(x_grid, y_grid_es, linestyle="--", label=f"H=20 early-stopped ({best_epoch} epochs)")
plt.xlabel("Standardised Sh/90")
plt.ylabel("Standardised SoT/90")
plt.title("Football Data: Full Training vs. Early Stopping")
plt.legend()
plt.tight_layout()
plt.savefig("exercise8_early_stop.png", dpi=120)
plt.close()

# 3. Single evaluation on held-out test set
# Compare best capacity model vs early-stopped H=20 on validation MSE
if val_hist[best_epoch - 1] < min(va_mses):
    chosen_name = f"H=20 early-stopped at epoch {best_epoch}"
    chosen_net = net_es
    chosen_va = float(val_hist[best_epoch - 1])
    chosen_tr = float(tr_hist[best_epoch - 1])
else:
    chosen_name = f"H={best_h} (20,000 epochs)"
    chosen_net, _ = fit.train(xtr, ytr, hidden=best_h, epochs=20000, seed=0)
    chosen_va = float(min(va_mses))
    chosen_tr = float(fit.mse(chosen_net, xtr, ytr))

test_mse = fit.mse(chosen_net, xte, yte)
print(f"\nSelected final model: {chosen_name}")
print(f"  Train MSE: {chosen_tr:.4f}")
print(f"  Val MSE:   {chosen_va:.4f}")
print(f"  Test MSE:  {test_mse:.4f}")