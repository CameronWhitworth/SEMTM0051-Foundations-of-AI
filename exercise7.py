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