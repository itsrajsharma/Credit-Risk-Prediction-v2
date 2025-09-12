# app.py
"""
Credit Risk — Final Gradio app
- Loads pipeline "credit_risk_model.pkl" (preprocessing + RandomForestClassifier).
- Loads samples from "samples.csv" if present (must contain FEATURES + TARGET).
- Features:
  * Dropdown to pick a sample applicant and autofill inputs
  * Random example generator
  * CSV upload for batch predictions (returns a preview)
  * Single applicant predict that shows model probability + true TARGET (if sample chosen)
Notes:
- Ensure credit_risk_model.pkl and samples.csv are in the same folder as this script.
- Run: python app.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import gradio as gr

# ---------- CONFIG ----------
MODEL_PATH = "credit_risk_model.pkl"
SAMPLES_CSV = "samples.csv"   # optional, but recommended (must include FEATURES + TARGET)
PORT = 7865
# ----------------------------

# FEATURES: exact names & order that pipeline expects
FEATURES = [
    "DerogCnt",
    "InqCnt06",
    "InqTimeLast",
    "InqFinanceCnt24",
    "TLTimeFirst",
    "TLTimeLast",
    "TLCnt12",
    "TLCnt24",
    "TLBalHCPct",
    "TLSatPct",
    "TLDel3060Cnt24",
    "TLOpenPct",
    "TLBadDerogCnt",
    "TLDel60Cnt24", 
    "TLOpen24Pct"
]


# ------ Load model ------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Place it in the app folder.")

model = joblib.load(MODEL_PATH)  # pipeline with preprocessing + classifier

# ------ Load or synthesize samples ------
sample_df = None
if os.path.exists(SAMPLES_CSV):
    try:
        sample_df = pd.read_csv(SAMPLES_CSV)
        required = set(FEATURES + ["TARGET"])
        if not required.issubset(set(sample_df.columns)):
            missing = required - set(sample_df.columns)
            raise ValueError(f"samples.csv missing required columns: {missing}")
        sample_df = sample_df[FEATURES + ["TARGET"]].reset_index(drop=True)
    except Exception as e:
        print("Failed to load samples.csv:", e)
        sample_df = None

# fallback: try to load data.csv to create samples
if sample_df is None and os.path.exists("data.csv"):
    try:
        df_all = pd.read_csv("data.csv")
        if set(FEATURES + ["TARGET"]).issubset(df_all.columns):
            sample_df = df_all.sample(n=min(5, len(df_all)), random_state=42)[FEATURES + ["TARGET"]].reset_index(drop=True)
            sample_df.to_csv(SAMPLES_CSV, index=False)
            print("Created samples.csv from data.csv.")
    except Exception:
        sample_df = None

# final fallback: synthetic placeholders
if sample_df is None:
    print("No samples.csv or data.csv found. Generating 3 placeholder samples.")
    medians = {f: 0.0 for f in FEATURES}
    sample_df = pd.DataFrame([
        {**medians, **{"TARGET": 0}},
        {**medians, **{"TARGET": 1}},
        {**medians, **{"TARGET": 0}},
    ])
    sample_df = sample_df[FEATURES + ["TARGET"]]

sample_keys = [f"sample_{i}" for i in range(len(sample_df))]

# ------ Prediction helpers ------
def predict_df(X):
    Xc = X.copy()[FEATURES]
    try:
        probs = model.predict_proba(Xc)[:, 1]
    except Exception:
        pre = getattr(model, "named_steps", {}).get("pre", None)
        clf = getattr(model, "named_steps", {}).get("clf", None)
        if pre is not None and clf is not None:
            X_pre = pre.transform(Xc)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_pre)[:, 1]
            elif hasattr(clf, "decision_function"):
                scores = clf.decision_function(X_pre)
                probs = 1 / (1 + np.exp(-scores))
            else:
                probs = clf.predict(X_pre).astype(float)
        else:
            probs = model.predict(Xc).astype(float)
    return probs

def inputs_to_df(values):
    return pd.DataFrame([dict(zip(FEATURES, [float(v) for v in values]))], columns=FEATURES)

# ------ UI callbacks ------
def load_sample(key):
    if key is None or key not in sample_keys:
        return [0.0] * len(FEATURES)
    idx = int(key.split("_")[1])
    row = sample_df.iloc[idx]
    return [float(row[f]) for f in FEATURES]

def randomize_example(seed=42):
    rng = np.random.RandomState(int(seed))
    med = sample_df[FEATURES].median()
    vals = [max(0.0, float(med[f] + rng.normal(scale=max(1.0, 0.1 * abs(med[f]))))) for f in FEATURES]
    return vals

def predict_single(*args):
    """
    args: feature1..featureN, threshold, sample_key
    returns: label mapping, prob string, true target (if sample chosen)
    """
    *feat_vals, threshold, sample_key = args
    X = inputs_to_df(feat_vals)
    prob = float(predict_df(X)[0])
    decision = "REJECT" if prob >= float(threshold) else "APPROVE"
    prob_str = f"{prob*100:.2f}%"
    true_val = "N/A"
    if sample_key in sample_keys:
        idx = int(sample_key.split("_")[1])
        true_val = str(sample_df.iloc[idx]["TARGET"])
    label_map = {"REJECT": round(prob, 3), "APPROVE": round(1 - prob, 3)}
    return label_map, prob_str, true_val

def batch_predict(uploaded_file):
    if uploaded_file is None:
        return "No file uploaded", None
    try:
        df_in = pd.read_csv(uploaded_file.name) if hasattr(uploaded_file, "name") else pd.read_csv(uploaded_file)
    except Exception as e:
        return f"Failed to read CSV: {e}", None
    missing = [c for c in FEATURES if c not in df_in.columns]
    if missing:
        return f"CSV missing columns: {missing}", None
    X = df_in[FEATURES].astype(float)
    probs = predict_df(X)
    df_out = df_in.copy()
    df_out["pred_default_prob"] = probs
    df_out["pred_decision"] = np.where(df_out["pred_default_prob"] >= 0.5, "REJECT", "APPROVE")
    return "OK", df_out.head(50)

# ------ Build Gradio UI ------
with gr.Blocks(title="Credit Risk — Demo (samples + CSV + random)") as demo:
    gr.Markdown("# Credit Risk — Demo")
    gr.Markdown("Select a sample applicant, randomize, or upload a CSV. Predictions show model probability and true TARGET for samples.")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                sample_dropdown = gr.Dropdown(choices=sample_keys, label="Choose sample applicant", value=sample_keys[0])
                btn_load = gr.Button("Load sample")
                btn_random = gr.Button("Random example")
                csv_uploader = gr.File(label="Upload CSV for batch prediction (columns must match FEATURES)", file_types=[".csv"])
            # numeric inputs
            input_components = []
            defaults = sample_df[FEATURES].median().to_dict()
            for feat in FEATURES:
                comp = gr.Number(label=feat, value=float(defaults.get(feat, 0.0)))
                input_components.append(comp)
            threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Decision threshold", value=0.5)
            submit = gr.Button("Predict (single)")

        with gr.Column(scale=1):
            out_label = gr.Label(num_top_classes=2, label="Model Decision (probabilities)")
            out_prob = gr.Textbox(label="Predicted Default Probability (%)")
            out_true = gr.Textbox(label="True TARGET (for selected sample)")
            batch_status = gr.Textbox(label="Batch upload status / messages")
            batch_table = gr.Dataframe(label="Batch predictions (preview)")

    # actions
    btn_load.click(fn=load_sample, inputs=sample_dropdown, outputs=input_components)
    btn_random.click(fn=lambda: randomize_example(), inputs=[], outputs=input_components)
    submit.click(fn=predict_single, inputs=input_components + [threshold, sample_dropdown], outputs=[out_label, out_prob, out_true])
    csv_uploader.upload(fn=batch_predict, inputs=csv_uploader, outputs=[batch_status, batch_table])

if __name__ == "__main__":
    print(f"Starting Gradio on 127.0.0.1:{PORT} ...")
    demo.launch(server_name="127.0.0.1", server_port=PORT, share=False, inbrowser=False, debug=True, show_error=True)
