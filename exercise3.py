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