"""
test.py — tests for all modules (only numpy, pandas, matplotlib used).

Run with:
    python test.py
"""

import numpy as np

from preprocess import split_features_target, train_test_split, normalize, make_val_split
from model      import (sigmoid, relu, sigmoid_grad, relu_grad,
                        init_params, forward, bce_loss, backward,
                        update_params, predict_prob, predict)
from metrics    import (accuracy_np, precision_np, recall_np, f1_np,
                        confusion_matrix_np, roc_curve_np, auc_from_roc)


# ── tiny helpers ───────────────────────────────────────────────────────────────

def make_dummy_data(n=100, n_features=13, seed=0):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, n_features))
    y   = rng.integers(0, 2, size=n)
    return X, y

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1

def close(a, b, atol=1e-6):
    return np.allclose(a, b, atol=atol)


# ── Preprocess ─────────────────────────────────────────────────────────────────

print("\n── preprocess ──")
X, y = make_dummy_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
check("train size is 80",             X_train.shape[0] == 80)
check("test size is 20",              X_test.shape[0]  == 20)
check("no rows lost in split",        X_train.shape[0] + X_test.shape[0] == 100)
check("y_train length matches",       y_train.shape[0] == 80)

X_train_norm, X_test_norm, _, _ = normalize(X_train, X_test)
check("train mean ≈ 0 after norm",    close(X_train_norm.mean(axis=0), np.zeros(13), atol=1e-10))
check("train std  ≈ 1 after norm",    close(X_train_norm.std(axis=0),  np.ones(13),  atol=1e-10))
check("test uses train stats (mean != 0)", not close(X_test_norm.mean(axis=0), np.zeros(13), atol=1e-2))

X_tr, X_val, y_tr, y_val = make_val_split(X_train_norm, y_train, val_frac=0.1)
check("val split sizes sum correctly", X_tr.shape[0] + X_val.shape[0] == X_train_norm.shape[0])
check("val frac ≈ 10%",                abs(X_val.shape[0] / X_train_norm.shape[0] - 0.1) < 0.02)


# ── Activations ────────────────────────────────────────────────────────────────

print("\n── activations ──")
check("sigmoid(0) == 0.5",            close(sigmoid(np.array([0.0])), [0.5]))
check("sigmoid output in [0,1]",      np.all(sigmoid(np.array([-1000., 0., 1000.])) >= 0)
                                       and np.all(sigmoid(np.array([-1000., 0., 1000.])) <= 1))
check("relu negatives → 0",           close(relu(np.array([-3., -1., 0.])), [0., 0., 0.]))
check("relu positives unchanged",     close(relu(np.array([1., 2., 3.])), [1., 2., 3.]))
check("relu_grad at -1 == 0",         relu_grad(np.array([-1.]))[0] == 0.0)
check("relu_grad at +1 == 1",         relu_grad(np.array([ 1.]))[0] == 1.0)
a = sigmoid(np.random.randn(4, 5))
check("sigmoid_grad shape preserved", sigmoid_grad(a).shape == (4, 5))


# ── init_params ────────────────────────────────────────────────────────────────

print("\n── init_params ──")
params = init_params([13, 32, 16, 1], seed=123)
check("W1 shape == (32, 13)",  params['W1'].shape == (32, 13))
check("W2 shape == (16, 32)",  params['W2'].shape == (16, 32))
check("W3 shape == (1,  16)",  params['W3'].shape == (1,  16))
check("b1 shape == (32,  1)",  params['b1'].shape == (32,  1))
check("biases initialised to 0", close(params['b1'], np.zeros((32, 1))))


# ── forward ────────────────────────────────────────────────────────────────────

print("\n── forward ──")
X_small = np.random.randn(10, 13)
A_out, caches = forward(X_small, params)
check("output shape == (1, 10)",     A_out.shape == (1, 10))
check("output in (0, 1) — sigmoid",  np.all(A_out > 0) and np.all(A_out < 1))
check("caches has A0",               'A0' in caches)
check("caches has Z3",               'Z3' in caches)


# ── bce_loss ───────────────────────────────────────────────────────────────────

print("\n── bce_loss ──")
y_perfect = np.array([1, 0, 1, 0])
p_perfect = np.array([0.9999, 0.0001, 0.9999, 0.0001])
p_random  = np.array([0.5, 0.5, 0.5, 0.5])
check("perfect preds → low loss",   bce_loss(y_perfect, p_perfect) < 0.01)
check("random preds → loss ≈ ln2",  close(bce_loss(y_perfect, p_random), np.log(2), atol=1e-4))
check("loss is a scalar",           np.isscalar(bce_loss(y_perfect, p_perfect)))


# ── backward + update_params ──────────────────────────────────────────────────

print("\n── backward + update_params ──")
grads = backward(params, caches, X_small, np.random.randint(0, 2, 10))
check("dW1 shape matches W1",  grads['dW1'].shape == params['W1'].shape)
check("db1 shape matches b1",  grads['db1'].shape == params['b1'].shape)

W1_before = params['W1'].copy()
params = update_params(params, grads, lr=0.01)
check("W1 changed after update", not close(params['W1'], W1_before))


# ── predict ────────────────────────────────────────────────────────────────────

print("\n── predict ──")
probs = predict_prob(X_small, params)
preds = predict(X_small, params, threshold=0.5)
check("predict_prob shape == (10,)",  probs.shape == (10,))
check("predict_prob values in [0,1]", np.all(probs >= 0) and np.all(probs <= 1))
check("predict outputs only 0s/1s",   set(preds.tolist()).issubset({0, 1}))


# ── metrics ────────────────────────────────────────────────────────────────────

print("\n── metrics ──")
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 1, 1])

check("accuracy correct",   close(accuracy_np(y_true, y_pred), 4/6))
check("precision correct",  close(precision_np(y_true, y_pred), 3/4))
check("recall correct",     close(recall_np(y_true, y_pred),    3/4))
check("f1 correct",         close(f1_np(y_true, y_pred),        3/4))

cm = confusion_matrix_np(y_true, y_pred)
check("cm shape == (2,2)", cm.shape == (2, 2))
check("cm TP == 3",        cm[1, 1] == 3)
check("cm TN == 1",        cm[0, 0] == 1)
check("cm FP == 1",        cm[0, 1] == 1)
check("cm FN == 1",        cm[1, 0] == 1)

y_scores = np.array([0.9, 0.2, 0.4, 0.8, 0.3, 0.7])
fpr, tpr, _ = roc_curve_np(y_true, y_scores)
auc = auc_from_roc(fpr, tpr)
check("AUC in [0, 1]",      0.0 <= auc <= 1.0)
check("perfect AUC == 1.0", close(auc_from_roc(np.array([0., 0., 1.]),
                                                np.array([0., 1., 1.])), 1.0))


# ── Summary ────────────────────────────────────────────────────────────────────

print(f"\n{'='*40}")
print(f"  {passed} passed  |  {failed} failed")
print(f"{'='*40}\n")
if failed:
    raise SystemExit(1)
