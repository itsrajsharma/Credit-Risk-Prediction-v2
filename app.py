# app.py
"""
Credit Risk — Extended Gradio app (v2)

Data pipeline:
  data.xlsx → drop ID/TARGET → IQR cap → KNN impute → select top 14 → scale → RF

Artifacts (built once by build_pipeline.py, loaded at startup):
  credit_risk_model.pkl  — fitted RandomForestClassifier
  imputer.pkl            — fitted KNNImputer  (on all 28 raw features)
  scaler.pkl             — fitted StandardScaler (on 14 selected features)
  features.pkl           — list of 14 selected feature names
  iqr_bounds.pkl         — IQR clip bounds dict  {col: (lo, hi)}
  split_data.pkl         — pre-split X_tr/X_te/y_tr/y_te for comparison tab

Inference (single + batch):
  raw input → IQR cap → impute → select features → scale → predict

Run:  python app.py
"""

import io
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import gradio as gr
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
DATA_XLSX  = "data.xlsx"
PORT       = 7860
SAMPLES_CSV = "samples.csv"

# Columns that exist in the raw data but are NOT model features
_NON_FEATURE_COLS = {"target", "default", "loan_status", "credit_risk", "id"}

# ═══════════════════════════════════════════════════════════════
# Load all fitted artifacts
# ═══════════════════════════════════════════════════════════════
_REQUIRED = ["credit_risk_model.pkl", "imputer.pkl", "scaler.pkl",
             "features.pkl", "iqr_bounds.pkl"]

for _f in _REQUIRED:
    if not os.path.exists(_f):
        raise FileNotFoundError(
            f"Artifact '{_f}' not found. Run build_pipeline.py first."
        )

rf_clf      = joblib.load("credit_risk_model.pkl")   # RandomForestClassifier
imputer     = joblib.load("imputer.pkl")              # KNNImputer (28 features)
scaler      = joblib.load("scaler.pkl")               # StandardScaler (14 features)
FEATURES    = joblib.load("features.pkl")             # list[str] — 14 selected
iqr_bounds  = joblib.load("iqr_bounds.pkl")           # {col: (lo, hi)}

# All raw feature columns that the imputer was fitted on (28 cols, no ID/TARGET)
# Reconstruct from the imputer's feature count — we need the column order
# The imputer was fit on all columns except ID/TARGET/high-null.
# We store the all-feature column list so batch CSV can be validated.
# Derive it: all columns minus non-features. We stored this in imputer indirectly;
# we reconstruct the ordered list from the data.xlsx header (always available).
_df_cols_cache: list = []

def _get_all_raw_features() -> list:
    """Return the ordered list of raw feature columns (as seen by the imputer)."""
    global _df_cols_cache
    if _df_cols_cache:
        return _df_cols_cache
    if os.path.exists(DATA_XLSX):
        df_hdr = pd.read_excel(DATA_XLSX, nrows=0)
        cols = [c for c in df_hdr.columns if c.lower() not in _NON_FEATURE_COLS]
        _df_cols_cache = cols
    else:
        # Fallback: use whatever the imputer was trained on (no names stored)
        # Infer from feature count
        n = imputer.n_features_in_
        _df_cols_cache = [f"feature_{i}" for i in range(n)]
    return _df_cols_cache

ALL_RAW_FEATURES = _get_all_raw_features()

# ═══════════════════════════════════════════════════════════════
# SHAP explainer — on the RF classifier
# ═══════════════════════════════════════════════════════════════
shap_explainer = shap.TreeExplainer(rf_clf)

# ═══════════════════════════════════════════════════════════════
# Load split_data.pkl for model comparison
# ═══════════════════════════════════════════════════════════════
_split_data = None
if os.path.exists("split_data.pkl"):
    _split_data = joblib.load("split_data.pkl")

# ═══════════════════════════════════════════════════════════════
# Load / build sample rows for the single-predict UI
# ═══════════════════════════════════════════════════════════════
sample_df = None

# Try samples.csv first (must have FEATURES cols)
if os.path.exists(SAMPLES_CSV):
    try:
        _s = pd.read_csv(SAMPLES_CSV)
        # Accept if it at least has FEATURES columns (post-processed values)
        if set(FEATURES).issubset(_s.columns):
            sample_df = _s[FEATURES + (["TARGET"] if "TARGET" in _s.columns else [])].reset_index(drop=True)
    except Exception as e:
        print(f"samples.csv error: {e}")

# If samples.csv is old format or missing, rebuild from data.xlsx
if sample_df is None and os.path.exists(DATA_XLSX):
    try:
        _raw = pd.read_excel(DATA_XLSX, header=0)
        # Find target column
        _tcol = next(
            (c for c in _raw.columns if c.lower() in {"target", "default", "loan_status", "credit_risk"}),
            None
        )
        _raw_feat = _raw.drop(columns=[c for c in _raw.columns if c.lower() in _NON_FEATURE_COLS], errors="ignore")

        # Take 15 stratified samples (raw, unscaled)
        _sample_idx = []
        if _tcol:
            for _cls in [0, 1]:
                _idx = _raw[_raw[_tcol] == _cls].sample(
                    n=min(8, (_raw[_tcol] == _cls).sum()), random_state=42
                ).index
                _sample_idx.extend(_idx.tolist())
        else:
            _sample_idx = _raw.sample(n=15, random_state=42).index.tolist()

        _s_raw = _raw_feat.loc[_sample_idx].reset_index(drop=True)

        # Apply the same inference transform to get post-processed values for the UI
        _s_capped = _s_raw.copy()
        for col, (lo, hi) in iqr_bounds.items():
            if col in _s_capped.columns:
                _s_capped[col] = _s_capped[col].clip(lo, hi)

        _s_imp = pd.DataFrame(
            imputer.transform(_s_capped[ALL_RAW_FEATURES]),
            columns=ALL_RAW_FEATURES
        )
        _s_sel = _s_imp[FEATURES]

        # Store the un-scaled selected features as defaults (scale at predict time)
        sample_df = _s_sel.copy()
        if _tcol:
            sample_df["TARGET"] = _raw.loc[_sample_idx, _tcol].values
        sample_df = sample_df.reset_index(drop=True)

        # Save this as the new samples.csv
        sample_df.to_csv(SAMPLES_CSV, index=False)
        print(f"Rebuilt samples.csv with {len(sample_df)} rows from data.xlsx.")
    except Exception as e:
        print(f"Could not rebuild samples from data.xlsx: {e}")

# Final fallback: zeros
if sample_df is None:
    print("Generating placeholder samples.")
    sample_df = pd.DataFrame(
        [{f: 0.0 for f in FEATURES} for _ in range(3)]
    )
    sample_df["TARGET"] = [0, 1, 0]

sample_keys = [f"sample_{i}" for i in range(len(sample_df))]

# ═══════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════
def _apply_iqr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in iqr_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def _preprocess_raw(df_raw: pd.DataFrame) -> np.ndarray:
    """
    Full inference transform on a raw DataFrame that contains ALL_RAW_FEATURES columns.
    Returns numpy array ready for rf_clf.predict_proba.
    """
    # IQR cap
    df_capped = _apply_iqr(df_raw[ALL_RAW_FEATURES])
    # KNN impute
    X_imp = pd.DataFrame(imputer.transform(df_capped), columns=ALL_RAW_FEATURES)
    # Select features
    X_sel = X_imp[FEATURES]
    # Scale
    X_sc = scaler.transform(X_sel)
    return X_sc


def _preprocess_selected(df_sel: pd.DataFrame) -> np.ndarray:
    """
    Inference transform when the input already contains the 14 FEATURES columns
    (i.e. values from the UI sliders — already imputed, just need scaling).
    """
    return scaler.transform(df_sel[FEATURES].astype(float))


def predict_from_selected(df_sel: pd.DataFrame) -> np.ndarray:
    """Run prediction on a DataFrame that has exactly the FEATURES columns (post-imputation, pre-scale)."""
    X_sc = _preprocess_selected(df_sel)
    return rf_clf.predict_proba(X_sc)[:, 1]

# ═══════════════════════════════════════════════════════════════
# Confidence tier
# ═══════════════════════════════════════════════════════════════
def _confidence_tier(prob: float) -> tuple:
    """Return (tier_label, hex_color)."""
    if prob < 0.3:
        return "🟢 Low Risk", "#1a6636"
    elif prob < 0.6:
        return "🟡 Review", "#7a5c00"
    else:
        return "🔴 High Risk", "#8b1a1a"


def _tier_badge_html(prob: float) -> str:
    label, color = _confidence_tier(prob)
    return (
        f'<div style="display:inline-block;padding:8px 20px;border-radius:24px;'
        f'background:{color};color:#fff;font-size:1.1em;font-weight:700;'
        f'letter-spacing:0.5px;margin-top:4px">{label}</div>'
        f'<div style="color:#ccc;font-size:0.85em;margin-top:4px">'
        f'Default probability: <b>{prob*100:.2f}%</b></div>'
    )

# ═══════════════════════════════════════════════════════════════
# SHAP image
# ═══════════════════════════════════════════════════════════════
def _make_shap_image(X_sel: pd.DataFrame) -> Image.Image:
    """
    Generate SHAP bar plot.
    X_sel: DataFrame with FEATURES columns (post-imputation, pre-scale).
    """
    X_sc = _preprocess_selected(X_sel)
    sv = shap_explainer.shap_values(X_sc)

    if isinstance(sv, list):
        shap_vals = sv[1][0]
    elif sv.ndim == 3:
        shap_vals = sv[0, :, 1] if sv.shape[2] == 2 else sv[0]
    else:
        shap_vals = sv[0]

    indices   = np.argsort(np.abs(shap_vals))[::-1]
    top_n     = min(10, len(FEATURES))
    top_idx   = indices[:top_n]
    top_names = [FEATURES[i] for i in top_idx]
    top_vals  = shap_vals[top_idx]
    colors    = ["#e05252" if v > 0 else "#5294e0" for v in top_vals]

    fig, ax = plt.subplots(figsize=(6, max(3, 0.38 * top_n)), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    ax.barh(range(top_n), top_vals[::-1], color=colors[::-1], edgecolor="none", height=0.65)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9, color="#e0e0e0")
    ax.set_xlabel("SHAP value (impact on default probability)", fontsize=8, color="#aaa")
    ax.set_title("Feature Contributions (SHAP)", fontsize=10, color="#fff", pad=8)
    ax.tick_params(colors="#aaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.axvline(0, color="#666", linewidth=0.8, linestyle="--")
    fig.tight_layout(pad=1.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img

# ═══════════════════════════════════════════════════════════════
# Model Comparison — uses pre-split data from split_data.pkl
# ═══════════════════════════════════════════════════════════════
_comparison_results = None


def _build_comparison_table():
    """Build model comparison metrics. Returns styled HTML."""
    global _comparison_results
    if _comparison_results is not None:
        return _comparison_results

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    if _split_data is None:
        return (
            "<p style='color:#f88;'>split_data.pkl not found. "
            "Run build_pipeline.py to generate it.</p>"
        )

    X_tr = _split_data["X_tr"]
    X_te = _split_data["X_te"]
    y_tr = _split_data["y_tr"]
    y_te = _split_data["y_te"]
    data_source = "data.xlsx — proper 80/20 stratified split"

    neg, pos = np.bincount(y_tr)
    spw = neg / max(pos, 1)

    models_cfg = {
        "Random Forest": rf_clf,       # already trained, just evaluate
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
    }

    rows = []
    for name, clf in models_cfg.items():
        if name != "Random Forest":
            clf.fit(X_tr, y_tr)            # train on the same pre-processed split

        t0 = time.perf_counter()
        y_prob = clf.predict_proba(X_te)[:, 1]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        y_pred = (y_prob >= 0.5).astype(int)
        f1  = f1_score(y_te, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_te, y_prob)
        except ValueError:
            auc = float("nan")

        rows.append({
            "Model": name,
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "Inference (ms)": round(elapsed_ms, 3),
        })

    df_cmp = pd.DataFrame(rows)
    best_f1  = df_cmp["F1-Score"].idxmax()
    best_auc = df_cmp["ROC-AUC"].idxmax()
    best_ms  = df_cmp["Inference (ms)"].idxmin()

    def _row_html(r, idx):
        cells = []
        for col, val in r.items():
            is_best = (
                (col == "F1-Score"        and idx == best_f1)  or
                (col == "ROC-AUC"         and idx == best_auc) or
                (col == "Inference (ms)"  and idx == best_ms)
            )
            style = (
                "background:#1a6636;color:#fff;font-weight:700;"
                "border-radius:4px;padding:4px 10px;"
            ) if is_best else "padding:4px 10px;"
            cells.append(f'<td style="{style}">{val}</td>')
        return "<tr>" + "".join(cells) + "</tr>"

    header = "".join(
        f"<th style='padding:6px 14px;border-bottom:2px solid #444;text-align:left'>{c}</th>"
        for c in df_cmp.columns
    )
    body = "".join(_row_html(row, idx) for idx, row in df_cmp.iterrows())
    note = (
        f"<p style='font-size:0.8em;color:#aaa;margin-top:8px'>"
        f"Data source: {data_source}</p>"
    )
    html = (
        f"<div style='overflow-x:auto'>"
        f"<table style='border-collapse:collapse;width:100%;font-family:monospace'>"
        f"<thead><tr style='background:#1e1e2e'>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        f"</table>{note}</div>"
    )
    _comparison_results = html
    return html


def refresh_comparison():
    global _comparison_results
    _comparison_results = None
    return _build_comparison_table()

# ═══════════════════════════════════════════════════════════════
# UI callbacks
# ═══════════════════════════════════════════════════════════════
def load_sample(key):
    if key is None or key not in sample_keys:
        return [0.0] * len(FEATURES)
    idx = int(key.split("_")[1])
    row = sample_df.iloc[idx]
    return [float(row[f]) for f in FEATURES]


def randomize_example():
    rng = np.random.RandomState(int(np.random.randint(0, 10000)))
    med = sample_df[FEATURES].median()
    vals = [
        float(med[f] + rng.normal(scale=max(0.1, 0.1 * abs(med[f]))))
        for f in FEATURES
    ]
    return vals


def predict_single(*args):
    """
    args: feature1..feature14, threshold, sample_key
    Returns: label_map, prob_str, true_val, tier_html, shap_image
    """
    *feat_vals, threshold, sample_key = args
    # Build DataFrame with un-scaled selected features
    X_sel = pd.DataFrame(
        [dict(zip(FEATURES, [float(v) for v in feat_vals]))],
        columns=FEATURES
    )
    prob = float(predict_from_selected(X_sel)[0])
    prob_str = f"{prob * 100:.2f}%"
    decision = "REJECT" if prob >= float(threshold) else "APPROVE"

    true_val = "N/A"
    if sample_key in sample_keys and "TARGET" in sample_df.columns:
        idx = int(sample_key.split("_")[1])
        true_val = str(sample_df.iloc[idx]["TARGET"])

    label_map = {"REJECT": round(prob, 3), "APPROVE": round(1 - prob, 3)}
    tier_html = _tier_badge_html(prob)

    try:
        shap_img = _make_shap_image(X_sel)
    except Exception as ex:
        print(f"SHAP error: {ex}")
        shap_img = None

    return label_map, prob_str, true_val, tier_html, shap_img


def batch_predict(uploaded_file):
    """
    Batch CSV prediction.
    The CSV may contain either:
      (a) ALL_RAW_FEATURES columns → apply full pipeline
      (b) FEATURES (14) columns only → apply scale only
    """
    if uploaded_file is None:
        return "No file uploaded.", None

    try:
        path = uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file
        df_in = pd.read_csv(path)
    except Exception as e:
        return f"Failed to read CSV: {e}", None

    # Determine which mode
    has_raw = all(c in df_in.columns for c in ALL_RAW_FEATURES)
    has_sel = all(c in df_in.columns for c in FEATURES)

    if not has_raw and not has_sel:
        # Report what's missing
        missing_sel = [c for c in FEATURES if c not in df_in.columns]
        missing_raw = [c for c in ALL_RAW_FEATURES if c not in df_in.columns]
        if len(missing_sel) <= len(missing_raw):
            return (
                f"Missing columns for prediction. "
                f"Your CSV needs at least these {len(FEATURES)} feature columns:\n"
                f"{FEATURES}\n\nMissing: {missing_sel}",
                None
            )
        else:
            return (
                f"Missing columns. Need either the 14 selected features or "
                f"all raw features.\nMissing (14-col mode): {missing_sel}",
                None
            )

    try:
        if has_raw:
            # Full pipeline: IQR → impute → select → scale
            X_sc = _preprocess_raw(df_in)
        else:
            # Scale only
            X_sc = scaler.transform(df_in[FEATURES].astype(float))

        probs = rf_clf.predict_proba(X_sc)[:, 1]
    except Exception as e:
        return f"Preprocessing/prediction failed: {e}", None

    df_out = df_in.copy()
    df_out["pred_default_prob"] = probs.round(4)
    df_out["pred_decision"]     = np.where(probs >= 0.5, "REJECT", "APPROVE")
    df_out["confidence_tier"]   = [_confidence_tier(p)[0] for p in probs]

    n = len(df_out)
    status = (
        f"OK — {n} rows processed. "
        f"Mode: {'full pipeline (raw features)' if has_raw else '14-feature mode (scale only)'}."
    )
    return status, df_out.head(50)

# ═══════════════════════════════════════════════════════════════
# Build Gradio UI
# ═══════════════════════════════════════════════════════════════
CUSTOM_CSS = """
body, .gradio-container { background: #12121e !important; }
.gr-tab-item { font-weight: 600; }
h1, h2, h3 { color: #e8e8f0 !important; }
.gr-button.primary { background: linear-gradient(135deg,#5b73e8,#9050e8) !important; border:none; }
.gr-button { border-radius: 8px !important; }
"""

_feat_desc = ", ".join(FEATURES)

with gr.Blocks(title="Credit Risk — AI Demo", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# 📊 Credit Risk Prediction\n"
        "Powered by Random Forest · SHAP explanations · Model comparison · Confidence tiers"
    )

    # ──────────────────────────────────────────────────
    # Tab 1: Single Prediction
    # ──────────────────────────────────────────────────
    with gr.Tab("🔍 Single Prediction"):
        gr.Markdown(
            "Select a sample applicant, randomise, or enter values manually, "
            "then click **Predict**.\n\n"
            f"*Active features ({len(FEATURES)}): {_feat_desc}*"
        )

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    sample_dropdown = gr.Dropdown(
                        choices=sample_keys, label="Choose sample applicant",
                        value=sample_keys[0]
                    )
                    btn_load   = gr.Button("📂 Load sample")
                    btn_random = gr.Button("🎲 Randomise")

                # numeric inputs — one per selected feature
                input_components = []
                defaults = sample_df[FEATURES].median().to_dict()
                for feat in FEATURES:
                    comp = gr.Number(label=feat, value=float(defaults.get(feat, 0.0)))
                    input_components.append(comp)

                threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01,
                    label="Decision threshold", value=0.5
                )
                submit = gr.Button("⚡ Predict", variant="primary")

            with gr.Column(scale=1):
                out_label = gr.Label(num_top_classes=2, label="Model Decision (probabilities)")
                out_prob  = gr.Textbox(label="Predicted Default Probability (%)")
                out_true  = gr.Textbox(label="True TARGET (for selected sample)")
                out_tier  = gr.HTML(label="Confidence Tier")
                out_shap  = gr.Image(label="SHAP Feature Explanation", type="pil")

        # ── Batch prediction ──
        gr.Markdown("---\n### 📁 Batch CSV Prediction")
        gr.Markdown(
            f"Upload a CSV with the **{len(FEATURES)} selected feature columns** "
            f"(or all raw feature columns). "
            f"Output includes `pred_default_prob`, `pred_decision`, and `confidence_tier`."
        )
        with gr.Row():
            csv_uploader = gr.File(
                label="Upload CSV for batch prediction", file_types=[".csv"]
            )
        with gr.Row():
            batch_status = gr.Textbox(label="Batch status")
            batch_table  = gr.Dataframe(label="Batch predictions preview (first 50 rows)")

        # Wire actions
        btn_load.click(fn=load_sample, inputs=sample_dropdown, outputs=input_components)
        btn_random.click(fn=randomize_example, inputs=[], outputs=input_components)
        submit.click(
            fn=predict_single,
            inputs=input_components + [threshold, sample_dropdown],
            outputs=[out_label, out_prob, out_true, out_tier, out_shap]
        )
        csv_uploader.upload(
            fn=batch_predict, inputs=csv_uploader,
            outputs=[batch_status, batch_table]
        )

    # ──────────────────────────────────────────────────
    # Tab 2: Model Comparison
    # ──────────────────────────────────────────────────
    with gr.Tab("📊 Model Comparison"):
        gr.Markdown(
            "### Random Forest vs XGBoost vs LightGBM\n"
            "All models trained on the same **80/20 stratified split** of `data.xlsx` "
            "after the full preprocessing pipeline.\n\n"
            "**Green highlight = best value per column.**"
        )
        cmp_html    = gr.HTML(label="Comparison Table")
        btn_refresh = gr.Button("🔄 Refresh / Re-evaluate", variant="secondary")

        demo.load(fn=_build_comparison_table, inputs=[], outputs=cmp_html)
        btn_refresh.click(fn=refresh_comparison, inputs=[], outputs=cmp_html)

if __name__ == "__main__":
    print(f"Starting Gradio on 0.0.0.0:{PORT} ...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        inbrowser=True,
        debug=True,
        show_error=True
    )
