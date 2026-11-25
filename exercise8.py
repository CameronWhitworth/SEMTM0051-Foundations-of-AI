"""
Coursework submission script: Exercises 1-8.
When doing work, I used indivdiual scrips, 
done my best to consolidate them into one working script for ease of use when marking.
football_data will be needed. I'll submit that alongside my work.
Full repo can be found at: https://github.com/CameronWhitworth/SEMTM0051-Foundations-of-AI
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import fit
import make_data

IMG_DIR = "images"


def img_path(filename):
    return os.path.join(IMG_DIR, filename)


def exercise_1(x, y):
    print("\n--- Exercise 1 ---")
    xgrid = np.linspace(-2, 2, 400)
    sizes = [0, 1, 2, 5, 20, 100]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)

    for ax, H in zip(axes.ravel(), sizes):
        net, history = fit.train(x, y, hidden=H, epochs=20000)
        ax.plot(x, y, "o", ms=4)
        ax.plot(xgrid, fit.predict(net, xgrid), "r")
        ax.set_ylim(-2.5, 3)
        ax.set_title(
            f"H = {H}: {fit.n_params(net)} parameters, "
            f"train MSE = {history['train'][-1]:.3f}"
        )
        print(
            f"H = {H:4d}  parameters = {fit.n_params(net):4d}  "
            f"train MSE = {history['train'][-1]:.3f}"
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(img_path("exercise1.png"), dpi=120)
    plt.close(fig)


def exercise_2(xtr, ytr, xva, yva):
    print("\n--- Exercise 2 ---")
    sizes = [0, 1, 2, 3, 5, 10, 20, 50, 100]
    x_vals = [0.5 if h == 0 else h for h in sizes]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, seed in [(ax0, 0), (ax1, 1)]:
        tr_losses, va_losses = [], []
        for h in sizes:
            net, hist = fit.train(xtr, ytr, hidden=h, epochs=20000, seed=seed)
            tr_loss = hist["train"][-1]
            va_loss = fit.mse(net, xva, yva)
            tr_losses.append(tr_loss)
            va_losses.append(va_loss)
            print(f"seed={seed} H={h}: train={tr_loss:.4f}, val={va_loss:.4f}")

        ax.plot(x_vals, tr_losses, "o-", label="train")
        ax.plot(x_vals, va_losses, "o-", label="val")
        ax.set_xscale("log")
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(h) for h in sizes])
        ax.set_xlabel("H (H=0 at 0.5)")
        ax.set_title(f"seed = {seed}")
        ax.legend()

    ax0.set_ylabel("MSE")
    plt.tight_layout()
    plt.savefig(img_path("exercise2.png"), dpi=120)
    plt.close(fig)


def exercise_3(xtr, ytr, xva, yva):
    print("\n--- Exercise 3 ---")
    net_full, hist = fit.train(xtr, ytr, hidden=20, epochs=20000, xval=xva, yval=yva)

    tr_hist = hist["train"]
    va_hist = hist["val"]
    epochs = np.arange(1, len(tr_hist) + 1)

    best_idx = int(np.argmin(va_hist))
    best_epoch = best_idx + 1

    print("Best epoch:", best_epoch)
    print(
        "At best epoch -> train:",
        round(tr_hist[best_idx], 4),
        "val:",
        round(va_hist[best_idx], 4),
    )
    print(
        "At 20000 -> train:",
        round(tr_hist[-1], 4),
        "val:",
        round(va_hist[-1], 4),
    )

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
    plt.savefig(img_path("exercise3.png"), dpi=120)
    plt.close(fig)
    return best_epoch


def exercise_4(xtr, ytr, xva, yva):
    print("\n--- Exercise 4 ---")
    wds = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    tr_losses = []
    va_losses = []
    nets = {}

    for wd in wds:
        net, hist = fit.train(xtr, ytr, hidden=20, epochs=20000, weight_decay=wd)
        tr_loss = hist["train"][-1]
        va_loss = fit.mse(net, xva, yva)
        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        nets[wd] = net
        print(f"wd={wd}: train={tr_loss:.4f}, val={va_loss:.4f}")

    x_plot = [1e-5 if w == 0 else w for w in wds]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_plot, tr_losses, "o-", label="train")
    ax.plot(x_plot, va_losses, "o-", label="val")
    ax.set_xscale("log")
    ax.set_xticks(x_plot)
    ax.set_xticklabels(["0" if w == 0 else str(w) for w in wds])
    ax.set_xlabel("weight decay (0 plotted at 1e-5)")
    ax.set_ylabel("MSE")
    ax.set_title("Weight decay sweep (H = 20)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(img_path("exercise4.png"), dpi=120)
    plt.close(fig)

    best_wd = wds[int(np.argmin(va_losses))]
    xgrid = np.linspace(-2, 2, 400)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(xtr, ytr, "o", ms=5, label="train data")
    for wd_val, label_txt in [(0, "wd=0"), (best_wd, f"best wd={best_wd}"), (1e-1, "wd=0.1")]:
        ax2.plot(xgrid, fit.predict(nets[wd_val], xgrid), label=label_txt)
    ax2.set_ylim(-2.5, 3)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Fitted curves across weight decay (H = 20)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(img_path("exercise4_curves.png"), dpi=120)
    plt.close(fig2)
    return best_wd


def exercise_5(xtr, ytr, xva, yva, xte, yte, best_wd=1e-3):
    print("\n--- Exercise 5 ---")
    net_final, hist_final = fit.train(
        xtr, ytr, hidden=20, epochs=20000, weight_decay=best_wd, xval=xva, yval=yva
    )

    tr_loss = hist_final["train"][-1]
    va_loss = hist_final["val"][-1]
    te_loss = fit.mse(net_final, xte, yte)

    print(f"Final model (H=20, wd={best_wd}, 20000 epochs):")
    print(f"train MSE: {tr_loss:.4f}")
    print(f"val MSE:   {va_loss:.4f}")
    print(f"test MSE:  {te_loss:.4f}")
    return net_final, va_loss, te_loss


def exercise_6(xtr, ytr, xva, yva, xte, yte, net_final, final_va, final_te, best_wd=1e-3):
    print("\n--- Exercise 6 ---")
    rng = np.random.default_rng(123)
    xbig, ybig = make_data.make(10000, rng)

    final_big = fit.mse(net_final, xbig, ybig)
    true_mse = np.mean((make_data.true_function(xbig) - ybig) ** 2)

    net_overfit, hist_overfit = fit.train(
        xtr, ytr, hidden=20, epochs=20000, weight_decay=0.0, xval=xva, yval=yva
    )
    overfit_big = fit.mse(net_overfit, xbig, ybig)
    overfit_te = fit.mse(net_overfit, xte, yte)

    print(
        f"Final model (wd={best_wd}) -> val: {final_va:.4f}, "
        f"test: {final_te:.4f}, 10k true error: {final_big:.4f}"
    )
    print(f"True function MSE on 10k points (sigma^2 estimate): {true_mse:.4f}")
    print(
        f"Overfit H=20 -> val: {hist_overfit['val'][-1]:.4f}, "
        f"test: {overfit_te:.4f}, 10k true error: {overfit_big:.4f}"
    )


def exercise_7(xtr, ytr, xva, yva):
    print("\n--- Exercise 7 ---")
    runs = [
        ("sgd", 0.01),
        ("sgd", 0.1),
        ("sgd", 0.3),
        ("sgd", 1.0),
        ("adam", 0.01),
        ("adam", 0.001),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for opt, lr in runs:
        net, hist = fit.train(
            xtr, ytr, hidden=20, epochs=20000, lr=lr, optimiser=opt, xval=xva, yval=yva
        )
        tr = np.array(hist["train"], dtype=float)
        va = np.array(hist["val"], dtype=float)
        epochs = np.arange(1, len(tr) + 1)

        below = np.where(tr < 0.2)[0]
        first_epoch = int(below[0] + 1) if len(below) > 0 else None
        print(
            f"{opt} lr={lr}: first <0.2 at epoch {first_epoch} | "
            f"final train={tr[-1]:.4f}, final val={va[-1]:.4f}"
        )

        tr_safe = np.where(np.isfinite(tr) & (tr > 0), tr, np.nan)
        ax.loglog(epochs, tr_safe, label=f"{opt} lr={lr}")

    ax.axhline(0.2, linestyle="--", linewidth=1, label="0.2 threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title("Training loss: SGD vs Adam (H = 20)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(img_path("exercise7.png"), dpi=120)
    plt.close(fig)


def exercise_8(data_path="football_data.csv"):
    print("\n--- Exercise 8 ---")
    if not os.path.exists(data_path):
        print(f"Skipping Exercise 8: '{data_path}' not found in working directory.")
        return

    x, y = fit.load(data_path)
    (xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)

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

    x_plot = [0.5 if h == 0 else h for h in hs]
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.3))
    ax1.plot(x_plot, tr_mses, marker="o", label="Train MSE")
    ax1.plot(x_plot, va_mses, marker="s", label="Validation MSE")
    ax1.set_xscale("log")
    ax1.set_xticks(x_plot)
    ax1.set_xticklabels([str(h) for h in hs])
    ax1.set_xlabel("Hidden units H (0 plotted at 0.5)")
    ax1.set_ylabel("MSE")
    ax1.set_title("Football Data: Capacity Sweep (30 train / 30 val)")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(img_path("exercise8_capacity.png"), dpi=120)
    plt.close(fig1)

    net_20, hist_20 = fit.train(xtr, ytr, hidden=20, epochs=20000, xval=xva, yval=yva, seed=0)
    val_hist = np.array(hist_20["val"], dtype=float)
    tr_hist = np.array(hist_20["train"], dtype=float)
    best_epoch = int(np.argmin(val_hist)) + 1

    print(f"\nH=20 best val epoch: {best_epoch}")
    print(
        f"  At best epoch -> train MSE: {tr_hist[best_epoch - 1]:.4f}, "
        f"val MSE: {val_hist[best_epoch - 1]:.4f}"
    )
    print(
        f"  At 20000 ep   -> train MSE: {tr_hist[-1]:.4f}, "
        f"val MSE: {val_hist[-1]:.4f}"
    )

    net_es, _ = fit.train(xtr, ytr, hidden=20, epochs=best_epoch, seed=0)

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)
    y_grid_20 = fit.predict(net_20, x_grid)
    y_grid_es = fit.predict(net_es, x_grid)

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.3))
    ax2.scatter(xtr, ytr, alpha=0.75, label="Training points")
    ax2.plot(x_grid, y_grid_20, label="H=20 full (20,000 epochs)")
    ax2.plot(x_grid, y_grid_es, linestyle="--", label=f"H=20 early-stopped ({best_epoch} epochs)")
    ax2.set_xlabel("Standardised Sh/90")
    ax2.set_ylabel("Standardised SoT/90")
    ax2.set_title("Football Data: Full Training vs. Early Stopping")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(img_path("exercise8_early_stop.png"), dpi=120)
    plt.close(fig2)

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


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    x, y = fit.load("data.csv")
    (xtr, ytr), (xva, yva), (xte, yte) = fit.split(x, y, n_train=30, n_val=30)

    exercise_1(x, y)
    exercise_2(xtr, ytr, xva, yva)
    exercise_3(xtr, ytr, xva, yva)
    best_wd = exercise_4(xtr, ytr, xva, yva)
    net_final, final_va, final_te = exercise_5(xtr, ytr, xva, yva, xte, yte, best_wd=best_wd)
    exercise_6(
        xtr, ytr, xva, yva, xte, yte, net_final, final_va, final_te, best_wd=best_wd
    )
    exercise_7(xtr, ytr, xva, yva)
    exercise_8("football_data.csv")


if __name__ == "__main__":
    main()
