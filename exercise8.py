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