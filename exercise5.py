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