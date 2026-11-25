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