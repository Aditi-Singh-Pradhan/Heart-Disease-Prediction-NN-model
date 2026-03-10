import numpy as np
import matplotlib.pyplot as plt

from model import init_params, forward, bce_loss, backward, update_params


def smooth(x, window=7):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or window <= 1:
        return x
    pad  = np.full(window - 1, x[0])
    xpad = np.concatenate([pad, x])
    csum = np.cumsum(xpad)
    return (csum[window:] - csum[:-window]) / window


def train(X_train, y_train, X_val=None, y_val=None,
          layers=None, lr=0.01, epochs=300, print_every=50):
    """
    Train the neural network.

    Parameters
    ----------
    X_train / y_train : training data
    X_val   / y_val   : optional validation data
    layers  : list like [None, 32, 16, 1]  (first element ignored; n_features used instead)
    lr      : learning rate
    epochs  : number of full passes over the training set
    print_every : how often to print a loss line

    Returns
    -------
    params  : trained weight/bias dict
    history : {'train_loss': [...], 'val_loss': [...]}
    """
    if layers is None:
        layers = [None, 32, 16, 1]

    n_features = X_train.shape[1]
    layer_sizes = [n_features] + layers[1:]
    params  = init_params(layer_sizes, seed=123)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        A_train, caches = forward(X_train, params)
        train_loss      = bce_loss(y_train, A_train)
        grads           = backward(params, caches, X_train, y_train)
        params          = update_params(params, grads, lr)

        if X_val is not None:
            A_val, _ = forward(X_val, params)
            loss_val = bce_loss(y_val, A_val.flatten())
        else:
            loss_val = np.nan

        history['train_loss'].append(train_loss)
        history['val_loss'].append(loss_val)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            msg = f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.6f}"
            if not np.isnan(loss_val):
                msg += f" — val_loss: {loss_val:.6f}"
            print(msg)

    # ── Plot loss curves (once, after training) ──────────────────────────────
    train_loss_arr = np.array(history['train_loss'], dtype=float)
    val_loss_arr   = np.array(history['val_loss'],   dtype=float)

    plt.figure(figsize=(9, 4))
    if train_loss_arr.size:
        plt.plot(train_loss_arr, label='train_loss', alpha=0.6)
        plt.plot(smooth(train_loss_arr, window=7), label='train_loss (smoothed)', linewidth=2)
    if val_loss_arr.size and not np.all(np.isnan(val_loss_arr)):
        val_plot = np.where(np.isnan(val_loss_arr), np.nan, val_loss_arr)
        plt.plot(val_plot, label='val_loss', alpha=0.6)
        valid_vals = val_plot[~np.isnan(val_plot)]
        if valid_vals.size:
            sm_val = smooth(valid_vals, window=7)
            plt.plot(range(len(val_plot) - len(sm_val), len(val_plot)),
                     sm_val, label='val_loss (smoothed)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Training / Validation Loss')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()

    return params, history
