"""
late_delivery_model.py — predict Late_delivery_risk AT ORDER TIME (leakage-free).
Gradient boosting, isotonic-calibrated. Uses only pre-shipment features, so no
post-delivery leakage. Reports ROC-AUC, PR-AUC, Brier, confusion @F1 threshold,
feature importance; scores every order for the dashboard review queue.
Writes model_metrics.json, feature_importance.csv, late_scores.parquet, pr_roc.npz.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_curve, confusion_matrix)

try:
    from .config import OUT, SEED
    from .clean_validate import modeling_frame, MODEL_FEATURES
except ImportError:
    from config import OUT, SEED
    from clean_validate import modeling_frame, MODEL_FEATURES

NUM = ["sched_days", "quantity", "discount_rate", "product_price", "sales",
       "order_month", "order_dow"]
CAT = ["Shipping Mode", "Market", "Order Region", "Customer Segment", "Department Name"]


def _nn_auc(Xtr, ytr, Xte, yte):
    """Neural-network benchmark (Keras if available, else sklearn MLP)."""
    from sklearn.preprocessing import StandardScaler
    Xtr = Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr
    Xte = Xte.toarray() if hasattr(Xte, "toarray") else Xte
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    try:
        import tensorflow  # noqa
        from tensorflow import keras
        from tensorflow.keras import layers
        m = keras.Sequential([layers.Input((Xtr.shape[1],)),
                              layers.Dense(128, activation="relu"), layers.Dropout(0.2),
                              layers.Dense(64, activation="relu"),
                              layers.Dense(1, activation="sigmoid")])
        m.compile(optimizer="adam", loss="binary_crossentropy")
        m.fit(Xtr, ytr, epochs=15, batch_size=512, verbose=0)
        p = m.predict(Xte, verbose=0).ravel()
        backend = "keras"
    except Exception:
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=40, random_state=SEED)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        backend = "sklearn-mlp"
    return round(float(roc_auc_score(yte, p)), 4), backend


def train():
    df = modeling_frame()
    X = df[MODEL_FEATURES].copy()
    for c in NUM:
        X[c] = X[c].astype(float)
    y = df["Late_delivery_risk"].values
    pre = ColumnTransformer([("num", "passthrough", NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    base = Pipeline([("pre", pre), ("gb", GradientBoostingClassifier(random_state=SEED))])
    base.fit(Xtr, ytr)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xtr, ytr)
    p = cal.predict_proba(Xte)[:, 1]

    prec, rec, thr = precision_recall_curve(yte, p)
    fpr, tpr, _ = roc_curve(yte, p)
    f1 = 2*prec*rec/(prec+rec+1e-9); best = int(np.nanargmax(f1[:-1])); thr_star = float(thr[best])
    cm = confusion_matrix(yte, (p >= thr_star).astype(int))
    metrics = {
        "roc_auc": round(float(roc_auc_score(yte, p)), 4),
        "pr_auc": round(float(average_precision_score(yte, p)), 4),
        "brier": round(float(brier_score_loss(yte, p)), 4),
        "threshold_f1": round(thr_star, 4),
        "precision_at_thr": round(float(prec[best]), 4),
        "recall_at_thr": round(float(rec[best]), 4),
        "confusion_matrix": cm.tolist(),
        "late_base_rate": round(float(y.mean()), 4),
    }
    # neural-network benchmark on the same features
    pre_fitted = base.named_steps["pre"]
    nn_auc, nn_backend = _nn_auc(pre_fitted.transform(Xtr), ytr, pre_fitted.transform(Xte), yte)
    metrics["nn_auc"] = nn_auc
    metrics["nn_backend"] = nn_backend

    gb = base.named_steps["gb"]
    names = NUM + list(base.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CAT))
    imp = (pd.DataFrame({"feature": names, "importance": gb.feature_importances_})
           .sort_values("importance", ascending=False).head(15))
    imp.to_csv(OUT/"feature_importance.csv", index=False)
    np.savez(OUT/"pr_roc.npz", prec=prec, rec=rec, fpr=fpr, tpr=tpr)

    df = df.copy()
    df["late_prob"] = cal.predict_proba(X)[:, 1].astype("float32")
    df[["order_id", "order_item_id", "late_prob", "Late_delivery_risk"]].to_parquet(
        OUT/"late_scores.parquet", index=False)
    (OUT/"model_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    m = train()
    print(f"GBM ROC-AUC {m['roc_auc']} | PR-AUC {m['pr_auc']} | Brier {m['brier']}")
    print(f"Neural net ({m['nn_backend']}) ROC-AUC {m['nn_auc']}")
    print(f"@F1 thr {m['threshold_f1']}: precision {m['precision_at_thr']}, recall {m['recall_at_thr']}")
    print("confusion [[TN,FP],[FN,TP]]:", m["confusion_matrix"])
    print("wrote model_metrics.json, feature_importance.csv, late_scores.parquet, pr_roc.npz")


if __name__ == "__main__":
    main()
