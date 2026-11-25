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