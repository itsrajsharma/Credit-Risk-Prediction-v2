# build_pipeline.py
"""
Offline script — run once to fit the full preprocessing pipeline on data.xlsx
and save all artifacts that app.py needs at inference time.

Produces:
  credit_risk_model.pkl   — fitted RandomForestClassifier (post-preprocessing)
  imputer.pkl             — fitted KNNImputer
  scaler.pkl              — fitted StandardScaler
  features.pkl            — list[str] of the 14 selected feature names
  iqr_bounds.pkl          — dict {col: (lower, upper)} for IQR capping

Pipeline order (mirrors app.py inference):
  1. Drop non-feature cols (ID, TARGET)
  2. Drop high-null cols  (> 40% missing)
  3. IQR outlier capping  (fit bounds from training data)
  4. KNN imputation        (k=5, fit on training data)
  5. Feature selection     (top 14 by RF feature importance on scaled training data)
  6. Standard scaling      (fit on training data after feature selection)
  7. Train final models    (RF saved as credit_risk_model.pkl)
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Config ───────────────────────────────────────────────────────────
DATA_PATH      = "data.xlsx"
TARGET_NAMES   = ["target", "default", "loan_status", "credit_risk"]  # lower-case candidates
DROP_COLS      = ["id"]          # lower-case; always removed before modelling
NULL_THRESHOLD = 0.40            # drop columns with > 40% missing
N_FEATURES     = 14              # top N by RF importance
N_ESTIMATORS   = 200
RANDOM_STATE   = 42
# ─────────────────────────────────────────────────────────────────────


def find_target(df: pd.DataFrame) -> str:
    """Return the target column name (case-insensitive match or first binary 0/1 col)."""
    # 1. exact lower-case name match
    col_lower = {c.lower(): c for c in df.columns}
    for name in TARGET_NAMES:
        if name in col_lower:
            return col_lower[name]
    # 2. any binary column
    for col in df.columns:
        try:
            unique = set(df[col].dropna().unique())
            if unique <= {0, 1, 0.0, 1.0}:
                return col
        except Exception:
            pass
    raise ValueError(
        f"Could not identify a target column. "
        f"Expected one of {TARGET_NAMES} or a binary 0/1 column."
    )


def fit_iqr_bounds(df: pd.DataFrame) -> dict:
    """Return IQR clip bounds for every numeric column in df."""
    bounds = {}
    for col in df.select_dtypes(include="number").columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return bounds


def apply_iqr_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def main():
    print("=" * 60)
    print("Building credit risk preprocessing pipeline")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        sys.exit(f"ERROR: {DATA_PATH} not found.")

    print(f"\n[1] Loading {DATA_PATH} ...")
    df = pd.read_excel(DATA_PATH, header=0)
    print(f"    Raw shape: {df.shape}")

    # ── 2. Identify target & drop non-features ───────────────────
    print("\n[2] Identifying target column ...")
    target_col = find_target(df)
    print(f"    Target column: '{target_col}'")

    y = df[target_col].astype(int)
    print(f"    Class counts — 0: {(y==0).sum()}, 1: {(y==1).sum()}")

    col_lower = {c.lower(): c for c in df.columns}
    to_drop = [col_lower[d] for d in DROP_COLS if d in col_lower]
    to_drop.append(target_col)
    X = df.drop(columns=to_drop, errors="ignore")
    print(f"    Feature cols after dropping {to_drop}: {X.shape[1]}")

    # ── 3. Drop high-null columns ────────────────────────────────
    print(f"\n[3] Dropping columns with > {NULL_THRESHOLD*100:.0f}% missing ...")
    null_pct = X.isnull().mean()
    high_null_cols = null_pct[null_pct > NULL_THRESHOLD].index.tolist()
    if high_null_cols:
        print(f"    Dropped: {high_null_cols}")
        X = X.drop(columns=high_null_cols)
    else:
        print("    None found — all columns kept.")

    all_feature_cols = X.columns.tolist()
    print(f"    Remaining features: {len(all_feature_cols)}")

    # ── 4. Train / test split (on raw data, before any fitting) ──
    print("\n[4] Splitting 80/20 train/test ...")
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Train: {X_tr_raw.shape}, Test: {X_te_raw.shape}")

    # ── 5. IQR capping (fit on train only) ───────────────────────
    print("\n[5] Fitting IQR bounds (training data only) ...")
    iqr_bounds = fit_iqr_bounds(X_tr_raw)
    X_tr_capped = apply_iqr_bounds(X_tr_raw, iqr_bounds)
    X_te_capped = apply_iqr_bounds(X_te_raw, iqr_bounds)

    # ── 6. KNN imputation (fit on train only) ────────────────────
    print("\n[6] Fitting KNNImputer ...")
    imputer = KNNImputer(n_neighbors=5)
    X_tr_imp = pd.DataFrame(
        imputer.fit_transform(X_tr_capped), columns=all_feature_cols
    )
    X_te_imp = pd.DataFrame(
        imputer.transform(X_te_capped), columns=all_feature_cols
    )

    # ── 7. Feature selection via RF importance ───────────────────
    print(f"\n[7] Selecting top {N_FEATURES} features by RF importance ...")
    # Temporary scaler for feature selection step
    tmp_scaler = StandardScaler()
    X_tr_sc_tmp = tmp_scaler.fit_transform(X_tr_imp)

    rf_selector = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_selector.fit(X_tr_sc_tmp, y_tr)

    importances = pd.Series(rf_selector.feature_importances_, index=all_feature_cols)
    selected_features = importances.nlargest(N_FEATURES).index.tolist()
    print(f"    Selected: {selected_features}")

    X_tr_sel = X_tr_imp[selected_features]
    X_te_sel = X_te_imp[selected_features]

    # ── 8. StandardScaler (fit on train, selected features only) ─
    print("\n[8] Fitting StandardScaler on selected features ...")
    scaler = StandardScaler()
    X_tr_final = scaler.fit_transform(X_tr_sel)
    X_te_final = scaler.transform(X_te_sel)

    # ── 9. Train final RandomForest ──────────────────────────────
    print(f"\n[9] Training RandomForest (n_estimators={N_ESTIMATORS}) ...")
    t0 = time.perf_counter()
    neg, pos = np.bincount(y_tr)
    rf_final = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_final.fit(X_tr_final, y_tr)
    rf_time = (time.perf_counter() - t0) * 1000

    # Evaluate RF
    y_prob_rf = rf_final.predict_proba(X_te_final)[:, 1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)
    rf_f1  = f1_score(y_te, y_pred_rf, zero_division=0)
    rf_auc = roc_auc_score(y_te, y_prob_rf)
    print(f"    RF — F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}, Train time: {rf_time:.0f}ms")

    # ── 10. Save all fitted artifacts ────────────────────────────
    print("\n[10] Saving artifacts ...")
    joblib.dump(imputer,           "imputer.pkl")
    joblib.dump(scaler,            "scaler.pkl")
    joblib.dump(selected_features, "features.pkl")
    joblib.dump(iqr_bounds,        "iqr_bounds.pkl")
    joblib.dump(rf_final,          "credit_risk_model.pkl")
    print("     Saved: imputer.pkl, scaler.pkl, features.pkl, iqr_bounds.pkl, credit_risk_model.pkl")

    # Also save train/test split arrays for the comparison tab
    # (so app.py doesn't have to redo the full pipeline at runtime)
    split_data = {
        "X_tr": X_tr_final,
        "X_te": X_te_final,
        "y_tr": y_tr.values,
        "y_te": y_te.values,
    }
    joblib.dump(split_data, "split_data.pkl")
    print("     Saved: split_data.pkl (pre-split arrays for comparison tab)")

    # ── 11. Quick sanity check ───────────────────────────────────
    print("\n[11] Sanity check — reloading artifacts ...")
    imp2    = joblib.load("imputer.pkl")
    sc2     = joblib.load("scaler.pkl")
    feats2  = joblib.load("features.pkl")
    bounds2 = joblib.load("iqr_bounds.pkl")
    rf2     = joblib.load("credit_risk_model.pkl")
    assert feats2 == selected_features, "features.pkl mismatch!"
    test_row = X_tr_sel.iloc[[0]]
    test_sc  = sc2.transform(test_row)
    prob     = rf2.predict_proba(test_sc)[0, 1]
    print(f"     Sanity prediction on first train row: prob={prob:.4f} ✓")

    print("\n✅ Pipeline build complete!")
    print(f"   Features ({len(selected_features)}): {selected_features}")
    print(f"   RF  — F1={rf_f1:.4f}, AUC={rf_auc:.4f}")


if __name__ == "__main__":
    main()
