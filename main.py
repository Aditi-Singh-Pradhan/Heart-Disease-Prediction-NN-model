"""
main.py — entry point for heart-disease prediction.

Usage (local):
    python main.py

Update DATA_PATH to wherever heart.csv lives on your machine.
In Colab, mount Drive first and set DATA_PATH accordingly.
"""

import matplotlib.pyplot as plt

from preprocess import load_data, split_features_target, train_test_split, normalize, make_val_split
from model      import predict_prob, predict
from train      import train
from metrics    import print_metrics, roc_curve_np, auc_from_roc

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = 'heart.csv'          # update path as needed
LAYERS      = [None, 32, 16, 1]
LR          = 0.01
EPOCHS      = 300
PRINT_EVERY = 50
VAL_FRAC    = 0.1
THRESHOLD   = 0.5


def main():
    # 1. Load & inspect
    data = load_data(DATA_PATH)

    # 2. Features / target split
    X, y = split_features_target(data, target_col='target')

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    # 4. Normalise (fit on train only)
    X_train_norm, X_test_norm, _, _ = normalize(X_train, X_test)

    # 5. Carve out a validation set from the training data
    X_tr, X_val, y_tr, y_val = make_val_split(X_train_norm, y_train, val_frac=VAL_FRAC)

    # 6. Train
    params, history = train(
        X_tr, y_tr,
        X_val, y_val,
        layers=LAYERS,
        lr=LR,
        epochs=EPOCHS,
        print_every=PRINT_EVERY,
    )

    # 7. Evaluate on held-out test set
    y_test_probs = predict_prob(X_test_norm, params)
    y_test_pred  = predict(X_test_norm, params, threshold=THRESHOLD)

    fpr, tpr, auc_val = print_metrics(y_test, y_test_pred, y_test_probs)

    # 8. ROC curve plot
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve (AUC={auc_val:.3f})')
    plt.grid(alpha=0.3)
    plt.show()

    # 9. Sample predictions
    print("\nSample test predictions (index, true, prob, pred):")
    for i in range(min(8, len(y_test))):
        print(i, int(y_test[i]), f"{y_test_probs[i]:.3f}", int(y_test_pred[i]))


if __name__ == '__main__':
    main()
