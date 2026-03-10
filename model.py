import numpy as np


# ── Activations ────────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(a):
    """Expects a = sigmoid(z) already computed."""
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)


# ── Parameter initialisation ───────────────────────────────────────────────────

def init_params(layer_sizes, seed=123):
    """
    He-initialised weights and zero biases for every layer.

    Example: layer_sizes = [13, 32, 16, 1]
      → 13 input features, two hidden layers (32, 16 neurons), 1 output neuron.
    """
    rng    = np.random.default_rng(seed)
    params = {}
    L      = len(layer_sizes) - 1

    for l in range(1, L + 1):
        fan_in  = layer_sizes[l - 1]
        fan_out = layer_sizes[l]
        params[f'W{l}'] = rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_out, fan_in))
        params[f'b{l}'] = np.zeros((fan_out, 1))

    return params


# ── Forward pass ───────────────────────────────────────────────────────────────

def forward(X, params):
    """
    Push X through the network.

    Returns
    -------
    A_final : ndarray, shape (1, n_samples)  – predicted probabilities
    caches  : dict  – stores Z and A for every layer (needed by backward)
    """
    caches    = {}
    A         = X.T          # shape: (n_features, n_samples)
    caches['A0'] = A

    L = len([k for k in params if k.startswith('W')])

    for l in range(1, L + 1):
        W = params[f'W{l}']
        b = params[f'b{l}']
        Z = W.dot(A) + b
        caches[f'Z{l}'] = Z

        A = sigmoid(Z) if l == L else relu(Z)
        caches[f'A{l}'] = A

    return A, caches


# ── Loss ───────────────────────────────────────────────────────────────────────

def bce_loss(y_true, y_pred_probs):
    """Binary cross-entropy averaged over all samples."""
    eps = 1e-12
    p   = np.clip(y_pred_probs, eps, 1 - eps)
    y   = y_true.reshape(1, -1)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# ── Backward pass ──────────────────────────────────────────────────────────────

def backward(params, caches, X, y):
    """Compute gradients via backpropagation."""
    grads   = {}
    m       = X.shape[0]
    L       = len([k for k in params if k.startswith('W')])
    A_final = caches[f'A{L}']          # shape (1, m)
    Y       = y.reshape(1, -1)

    dA = -(np.divide(Y, A_final + 1e-12)) + np.divide(1 - Y, 1 - A_final + 1e-12)

    for l in reversed(range(1, L + 1)):
        Z      = caches[f'Z{l}']
        A_prev = caches[f'A{l-1}']
        W      = params[f'W{l}']

        dZ = dA * (sigmoid_grad(A_final) if l == L else relu_grad(Z))
        dW = (1 / m) * dZ.dot(A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        grads[f'dW{l}'] = dW
        grads[f'db{l}'] = db

        if l > 1:
            dA = W.T.dot(dZ)

    return grads


# ── Parameter update ───────────────────────────────────────────────────────────

def update_params(params, grads, lr):
    L = len([k for k in params if k.startswith('W')])
    for l in range(1, L + 1):
        params[f'W{l}'] -= lr * grads[f'dW{l}']
        params[f'b{l}'] -= lr * grads[f'db{l}']
    return params


# ── Inference helpers ──────────────────────────────────────────────────────────

def predict_prob(X, params):
    A_final, _ = forward(X, params)
    return A_final.flatten()

def predict(X, params, threshold=0.5):
    return (predict_prob(X, params) >= threshold).astype(int)
