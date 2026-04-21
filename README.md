# 📊 Credit Risk Prediction — End-to-End ML System

> A production-ready machine learning application for credit risk assessment, featuring a complete data engineering pipeline, SHAP-powered explainability, multi-model comparison, and an interactive Gradio web interface.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [Model Training](#4-model-training)
5. [Evaluation](#5-evaluation)
6. [SHAP Explainability](#6-shap-explainability)
7. [Confidence Tiers](#7-confidence-tiers)
8. [Gradio Interface](#8-gradio-interface)
9. [Tech Stack](#9-tech-stack)
10. [How to Run](#10-how-to-run)

---

## 1. Project Overview

### What This Project Does

This project is a full-stack machine learning system for **credit default prediction** — determining whether a loan applicant is likely to default based on their credit bureau history. Given a set of features describing an applicant's tradeline and inquiry activity, the system outputs:

- A **default probability** (0–1 continuous score)
- A **APPROVE / REJECT** decision applied against a tunable threshold
- A **colour-coded confidence tier** (Low Risk / Review / High Risk)
- A **SHAP explanation chart** showing exactly which features drove that individual decision

### Why It Matters in the Real World

Credit risk is the single largest source of financial loss for banks, credit unions, and fintech lenders. In the United States alone, consumer credit losses run into hundreds of billions of dollars annually. A lender that approves too many bad loans bleeds capital; one that rejects too many good loans loses market share.

The distinguishing challenge is **institutional defensibility**: regulators increasingly require that automated lending decisions be explainable. A model that says "reject" is only as valuable as the reasoning behind it. This system addresses that requirement head-on by pairing every prediction with a per-applicant SHAP decomposition — showing exactly how much each credit bureau metric pushed the score up or down.

### Problem Framing

|                              | Ground Truth: Low Risk         | Ground Truth: High Risk     |
| ---------------------------- | ------------------------------ | --------------------------- |
| **Predicted: Approve** | Correct (revenue)              | False Negative — loan loss |
| **Predicted: Reject**  | False Positive — lost revenue | Correct (avoided loss)      |

In credit risk, **false negatives are far more costly** than false positives. The system is therefore trained with `class_weight="balanced"` to prevent the majority class (non-default, 5:1 imbalance) from dominating the learning signal.

---

## 2. Dataset

### Source and Format

- **File:** `data.xlsx` (Excel workbook, single sheet, clean header row)
- **Rows:** 3,000 loan applications
- **Columns:** 30 total (1 target, 1 applicant ID, 28 raw features)
- **Read with:** `pandas.read_excel` + `openpyxl` engine

### Target Variable

| Column     | Type  | Values | Meaning                          |
| ---------- | ----- | ------ | -------------------------------- |
| `TARGET` | int64 | 0 / 1  | 0 = good standing, 1 = defaulted |

**Class distribution:** 2,500 non-default (83.3%) vs 500 default (16.7%) — a **5:1 class imbalance** that must be explicitly handled in both feature selection and model training.

### Raw Features

The 28 feature columns describe an applicant's credit bureau profile across three domains:

**Derogatory / Delinquency History** — counts of negative credit events:

| Feature            | Description                                         |
| ------------------ | --------------------------------------------------- |
| `DerogCnt`       | Total number of derogatory marks                    |
| `CollectCnt`     | Number of collections                               |
| `BanruptcyInd`   | Bankruptcy indicator (binary)                       |
| `TLBadDerogCnt`  | Tradelines with bad/derogatory status               |
| `TLBadCnt24`     | Bad tradelines in last 24 months                    |
| `TLDel3060Cnt24` | Tradelines 30–60 days delinquent in last 24 months |
| `TLDel60Cnt24`   | Tradelines 60+ days delinquent in last 24 months    |
| `TLDel60CntAll`  | All-time 60+ day delinquencies                      |
| `TLDel90Cnt24`   | Tradelines 90+ days delinquent in last 24 months    |
| `TLDel60Cnt`     | Total 60-day delinquency count                      |

**Credit Inquiry Behaviour** — how actively the applicant has been seeking credit:

| Feature             | Description                                 |
| ------------------- | ------------------------------------------- |
| `InqCnt06`        | Number of credit inquiries in last 6 months |
| `InqTimeLast`     | Time since last inquiry (months)            |
| `InqFinanceCnt24` | Finance-type inquiries in last 24 months    |

**Tradeline Activity and Utilisation** — depth and quality of credit account history:

| Feature                                             | Description                                     |
| --------------------------------------------------- | ----------------------------------------------- |
| `TLTimeFirst`                                     | Age of oldest tradeline (months)                |
| `TLTimeLast`                                      | Age of most recent tradeline (months)           |
| `TLCnt03` / `TLCnt12` / `TLCnt24` / `TLCnt` | Tradeline counts at various lookback windows    |
| `TLSum` / `TLMaxSum`                            | Total and maximum tradeline balance             |
| `TLSatCnt`                                        | Number of satisfactory tradelines               |
| `TLSatPct`                                        | Percentage of tradelines in satisfactory status |
| `TLOpenPct` / `TLOpen24Pct`                     | Percentage of tradelines currently open         |
| `TLBalHCPct`                                      | Balance-to-high-credit ratio                    |
| `TL75UtilCnt` / `TL50UtilCnt`                   | Count of tradelines above 75%/50% utilisation   |

### Missing Data in Raw File

11 of the 28 feature columns have some missing values, with a maximum of **6.3%** missing (in `InqTimeLast`). No column exceeds the 40% null threshold, so none are dropped, but imputation is required.

---

## 3. Data Preprocessing Pipeline

All preprocessing is implemented in `build_pipeline.py` and executed **once offline**. The fitted transformers are serialised to disk and loaded read-only at inference time — the app never refits on new data.

### Pipeline Execution Order

```
data.xlsx
    │
    ▼
[Step 1] Auto-detect target column + drop non-features (ID, TARGET)
    │  28 feature columns remain
    │
    ▼
[Step 2] Drop columns with > 40% missing values
    │  None dropped in this dataset (all columns ≤ 6.3% null)
    │
    ▼
[Step 3] Stratified 80/20 train/test split
    │  Train: 2,400 rows  │  Test: 600 rows
    │  Stratified on TARGET to preserve 5:1 class ratio in both sets
    │
    ▼
[Step 4] IQR Outlier Capping  ← fitted on TRAIN only
    │  Per column: clip to [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
    │  Bounds saved to iqr_bounds.pkl
    │
    ▼
[Step 5] KNN Imputation  ← fitted on TRAIN only
    │  k = 5 nearest neighbours
    │  Operates on all 28 feature columns
    │  Fitted object saved to imputer.pkl
    │
    ▼
[Step 6] Feature Selection via RF Importance
    │  Temporary RF (100 trees) trained on scaled imputed data
    │  Top 14 features selected by mean Gini importance
    │  Feature list saved to features.pkl
    │  28 → 14 features
    │
    ▼
[Step 7] Standard Scaling  ← fitted on TRAIN only, on 14 selected features
    │  Zero mean, unit variance
    │  Fitted object saved to scaler.pkl
    │
    ▼
[Step 8] Train final RandomForestClassifier (200 trees)
    │  Model saved to credit_risk_model.pkl
    │  Pre-split arrays (X_tr, X_te, y_tr, y_te) saved to split_data.pkl
```

### Why Each Step Was Done

**IQR Outlier Capping (before imputation):** Credit bureau data frequently contains extreme outliers — a tradeline balance of $500,000 when the median is $16,000, for example. These outliers would corrupt KNN imputation by distorting nearest-neighbour distances. Capping before imputation is the correct order.

**KNN Imputation (over SimpleImputer):** Missing values in credit data are not missing at random — an applicant with no recent inquiry has a structurally different profile from one with 7 inquiries. KNN imputation respects these relationships by borrowing from the 5 most similar applicants in feature space. Mean/median imputation would dilute this signal.

**Feature Selection (28 → 14):** Starting with 28 features, many encode redundant information (e.g., `TLDel60Cnt24` and `TLDel60CntAll` are correlated). RF importance-based selection reduces dimensionality while retaining the most discriminative signals, lowering overfitting risk and inference latency. The cutoff of 14 was chosen to balance signal retention and parsimony.

**StandardScaler (after selection):** Applied after feature selection because the scaler's parameters (mean, std) are specific to the selected feature set. Scaling ensures that KNN-based algorithms and regularised models are not biased by features with large absolute scales (e.g., `TLSum` in the thousands vs `TLSatPct` in the 0–1 range).

**Train-only fitting (no data leakage):** Every fitted object — IQR bounds, imputer, scaler — is derived exclusively from the training fold and then applied to the test fold. This is critical: fitting on the full dataset would leak test-set statistics into the training process and produce artificially inflated evaluation metrics.

### The 14 Selected Features

| Rank | Feature             | Domain                                               |
| ---- | ------------------- | ---------------------------------------------------- |
| 1    | `TLBalHCPct`      | Utilisation — balance to high-credit ratio          |
| 2    | `TLSatPct`        | Quality — % satisfactory tradelines                 |
| 3    | `TLTimeFirst`     | Age — months since first tradeline opened           |
| 4    | `TLDel3060Cnt24`  | Delinquency — 30–60 day lates in 24 months         |
| 5    | `TLSum`           | Volume — total tradeline balance                    |
| 6    | `TLMaxSum`        | Volume — maximum single tradeline balance           |
| 7    | `TLOpenPct`       | Activity — % of tradelines currently open           |
| 8    | `TLDel60Cnt24`    | Delinquency — 60+ day lates in 24 months            |
| 9    | `TLSatCnt`        | Quality — count of satisfactory tradelines          |
| 10   | `TLDel60CntAll`   | Delinquency — all-time 60+ day lates                |
| 11   | `TLTimeLast`      | Age — months since most recent tradeline            |
| 12   | `InqFinanceCnt24` | Inquiries — finance-type inquiries in 24 months     |
| 13   | `TL75UtilCnt`     | Utilisation — tradelines above 75%                  |
| 14   | `TLOpen24Pct`     | Activity — % of tradelines opened in last 24 months |

### Saved Artifacts and Inference Contract

| File                      | Content                                                    | Used at Inference            |
| ------------------------- | ---------------------------------------------------------- | ---------------------------- |
| `credit_risk_model.pkl` | Fitted `RandomForestClassifier` (200 trees, 14 features) | Final `predict_proba` call |
| `imputer.pkl`           | Fitted `KNNImputer` (k=5, 28 features)                   | Raw CSV batch prediction     |
| `scaler.pkl`            | Fitted `StandardScaler` (14 features)                    | Every prediction path        |
| `features.pkl`          | `list[str]` — 14 selected feature names in order        | Column selection & UI        |
| `iqr_bounds.pkl`        | `dict {col: (lo, hi)}` — per-column clip bounds         | Raw CSV batch prediction     |
| `split_data.pkl`        | `{X_tr, X_te, y_tr, y_te}` — pre-processed arrays       | Model comparison tab         |

At inference, `app.py` applies this exact transform chain — no refitting, ever:

```
User input (14 features, post-imputation)
    → scaler.transform()
    → rf_clf.predict_proba()

Batch CSV (28 raw features)
    → IQR cap using iqr_bounds
    → imputer.transform()
    → column selection using features list
    → scaler.transform()
    → rf_clf.predict_proba()
```

---

## 4. Model Training

### Primary Model: Random Forest

The production model is a `RandomForestClassifier` from scikit-learn, trained on the final preprocessed 14-feature dataset.

**Configuration:**

```python
RandomForestClassifier(
    n_estimators   = 200,       # 200 decision trees in the ensemble
    class_weight   = "balanced",# automatically up-weights the minority class
    random_state   = 42,        # reproducibility
    n_jobs         = -1,        # parallel training across all CPU cores
)
```

**Why Random Forest as the primary model:**

- Handles mixed feature scales gracefully (though we scale anyway for consistency with KNN)
- Built-in feature importance scores used in the selection step
- Robust to the remaining noise after IQR capping
- Supports `predict_proba` natively, required for confidence tiers and SHAP
- Compatible with `shap.TreeExplainer` for exact, fast SHAP computation

**Class imbalance handling:** With a 5:1 class ratio (2,500 good : 500 default), an untreated classifier would learn to predict "good" for nearly everyone and achieve 83% accuracy while being useless. `class_weight="balanced"` computes sample weights inversely proportional to class frequency, effectively giving each defaulted applicant 5× the influence of each non-defaulting applicant during tree splits.

### Comparison Models: XGBoost and LightGBM

XGBoost and LightGBM are trained on the **same pre-processed training split** (`split_data.pkl`) for a fair comparison. Both use equivalent imbalance handling:

```python
# XGBoost — scale_pos_weight equivalent to class_weight="balanced"
XGBClassifier(
    n_estimators     = 200,
    scale_pos_weight = 4.8,   # neg/pos ratio from training set
    random_state     = 42,
    eval_metric      = "logloss",
)

# LightGBM
LGBMClassifier(
    n_estimators  = 200,
    class_weight  = "balanced",
    random_state  = 42,
)
```

These models are not saved to disk — they are retrained on-demand when the Model Comparison tab first loads, using the pre-saved split arrays. This keeps startup time fast while ensuring the comparison uses the exact same data that RF was evaluated on.

---

## 5. Evaluation

### Metrics and Results

All metrics are computed on the **held-out test set (600 rows, 100 defaults)** — data the models have never seen during training or preprocessing.

| Model                   | F1-Score         | ROC-AUC          | Inference (ms) |
| ----------------------- | ---------------- | ---------------- | -------------- |
| **Random Forest** | 0.1552           | **0.7506** | ~46 ms         |
| **XGBoost**       | **0.3799** | 0.7226           | ~3 ms          |
| **LightGBM**      | 0.3757           | 0.7367           | ~5 ms          |

*Best value per column highlighted in the Gradio UI.*

### Why These Metrics for Credit Risk

**ROC-AUC (primary discriminative metric):**
AUC measures the model's ability to rank defaulters above non-defaulters across all possible thresholds. A value of 0.75 means the model correctly ranks a defaulter above a non-defaulter 75% of the time. This is the right metric for calibration and threshold selection — the lending team can choose their operating point (aggressive approval vs conservative rejection) independently of the model.

**F1-Score (operating-point metric):**F1 is the harmonic mean of precision and recall at the chosen threshold (0.5 by default). Credit risk F1 scores are characteristically low because:

1. The class imbalance means the denominator includes many easy negatives
2. The threshold is rarely optimised at 0.5 for real portfolios

The lower RF F1 (0.155) vs XGBoost (0.380) reflects RF's stronger bias toward the minority class at this threshold — it finds more defaults (higher recall) but with more false alarms (lower precision). XGBoost's higher F1 reflects a more balanced precision/recall trade-off.

**Inference Time:**
Critical for production: a batch of 10,000 loan applications must process in seconds. LightGBM (5 ms / 600 rows) and XGBoost (3 ms / 600 rows) are orders of magnitude faster than RF (46 ms / 600 rows) for the same batch. For individual predictions the difference is imperceptible, but at portfolio scale it matters significantly.

### The False Negative / False Positive Trade-off

```
Cost of a False Negative (approve a defaulter):
    → Full principal loss, recovery costs, credit provisioning
    → Typical loss given default: 40–70% of loan value

Cost of a False Positive (reject a creditworthy applicant):
    → Foregone interest revenue
    → Typically 3–8% annualised return on a safe loan
```

This asymmetry is why the **decision threshold is exposed as a slider** in the UI. A risk-averse lender might set it to 0.3 (flag any applicant with >30% default probability), accepting more false positives to capture more true positives.

---

## 6. SHAP Explainability

### What SHAP Is

SHAP (SHapley Additive exPlanations) is a game-theoretic framework for explaining the output of any machine learning model. For a given prediction, SHAP assigns each input feature a value representing its **marginal contribution** to moving the prediction away from the base rate (average model output across all training examples).

Formally, for a prediction `f(x)`:

```
f(x) = E[f(X)] + φ₁ + φ₂ + ... + φ₁₄
```

where `E[f(X)]` is the model's average prediction across all training data, and each `φᵢ` is the SHAP value for feature `i` — its unique, fair attribution of the difference between the base rate and this specific prediction.

### Why `TreeExplainer`

```python
shap_explainer = shap.TreeExplainer(rf_clf)
```

`shap.TreeExplainer` is the SHAP algorithm for tree-based models (Random Forest, XGBoost, LightGBM). It exploits the tree structure to compute **exact SHAP values** in polynomial time, without Monte Carlo sampling. For a Random Forest with 200 trees and 14 features, this runs in tens of milliseconds per prediction — fast enough for real-time UI use.

Alternative explainers (`KernelExplainer`, `LinearExplainer`) are model-agnostic but require sampling and are 100–1000× slower. Since the production model is a Random Forest, `TreeExplainer` is both faster and exact.

### What the SHAP Chart Shows

After each single prediction, the UI renders a horizontal bar chart:

- **X-axis:** SHAP value — the magnitude and direction of each feature's contribution to the predicted default probability
- **Bars pointing right (red):** Features that increased the default probability for this applicant
- **Bars pointing left (blue):** Features that decreased the default probability
- **Bar length:** Magnitude of contribution — longer bars have more influence
- **Feature order:** Sorted by absolute contribution, most influential at the top

**Example interpretation:**
If `TLDel60Cnt24` has a large red bar, this applicant has more 60-day delinquencies than the typical non-default applicant, and this is the primary reason the model scored them as elevated risk. If `TLSatPct` has a long blue bar, their high proportion of satisfactory tradelines is partially offsetting the delinquency signal.

This output is actionable: a loan officer can see not just "reject" but *why* — and can have a meaningful conversation with the applicant about which factors are driving the assessment.

---

## 7. Confidence Tiers

### Mapping Probability to Action

The model outputs a continuous probability `p ∈ [0, 1]`. A hard binary threshold (approve/reject at 0.5) discards useful information — applicants near the boundary are treated identically to those with extreme scores.

The confidence tier system maps probabilities to three actionable bands:

| Probability          | Tier                  | Colour | Suggested Action               |
| -------------------- | --------------------- | ------ | ------------------------------ |
| `p < 0.30`         | 🟢**Low Risk**  | Green  | Approve — standard pricing    |
| `0.30 ≤ p < 0.60` | 🟡**Review**    | Amber  | Manual underwriting required   |
| `p ≥ 0.60`        | 🔴**High Risk** | Red    | Decline or collateral required |

### Why This Is More Useful than a Binary Decision

In a real lending operation, different departments handle these tiers differently:

- **Low Risk:** Straight-through processing, automated approval, competitive rate offered
- **Review:** Passed to a human underwriter who examines additional documentation, employment verification, and the SHAP chart to understand the model's concern
- **High Risk:** Declined, or re-priced with risk-based pricing (higher interest rate), or offered a smaller credit limit

A binary approve/reject system at 0.5 would:

1. Automatically reject applicants at 0.49 who are nearly indistinguishable from those approved at 0.51
2. Automatically approve applicants at 0.59 who genuinely need human review
3. Not surface the SHAP reasons to the underwriter for borderline cases

The tier system with a tunable threshold transforms the model from a black-box decision-maker into a **risk stratification tool** that augments human judgement rather than replacing it.

### Batch Output

In batch (CSV) prediction mode, the `confidence_tier` column is appended to the output alongside `pred_default_prob` and `pred_decision`, allowing portfolio teams to segment the entire applicant pool by risk tier in one pass.

---

## 8. Gradio Interface

The application is implemented as a `gr.Blocks` layout with two tabs, served on port 7860.

### Tab 1: 🔍 Single Prediction

**Inputs (left panel):**

| Component          | Type                 | Purpose                                                                                              |
| ------------------ | -------------------- | ---------------------------------------------------------------------------------------------------- |
| Sample dropdown    | `gr.Dropdown`      | Select from 16 pre-built applicant profiles (8 non-default, 8 default) drawn from the actual dataset |
| Load sample button | `gr.Button`        | Populates all 14 feature inputs from the selected sample                                             |
| Randomise button   | `gr.Button`        | Generates a plausible random applicant by perturbing median feature values                           |
| 14 feature inputs  | `gr.Number`        | Manual entry of each credit feature (post-imputation values)                                         |
| Decision threshold | `gr.Slider` (0–1) | Adjustable approve/reject cutoff, default 0.5                                                        |
| Predict button     | `gr.Button`        | Triggers the prediction pipeline                                                                     |

**Outputs (right panel):**

| Component           | Content                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------- |
| Model Decision      | `gr.Label` — APPROVE/REJECT with bar chart showing respective probabilities              |
| Default Probability | `gr.Textbox` — exact probability as a percentage (e.g., "32.50%")                        |
| True TARGET         | `gr.Textbox` — ground truth label if a sample applicant was selected                     |
| Confidence Tier     | `gr.HTML` — coloured badge (🟢 Low Risk / 🟡 Review / 🔴 High Risk)                      |
| SHAP Chart          | `gr.Image` — horizontal bar chart, dark-themed, top-10 features by absolute contribution |

**Inference path for a single prediction:**

```
UI values (14 features, already imputed)
    → scaler.transform()
    → rf_clf.predict_proba() → probability
    → threshold comparison → APPROVE/REJECT
    → _confidence_tier() → tier badge
    → shap_explainer.shap_values() → SHAP bar chart (PIL Image)
```

**Batch prediction section (beneath the single predict form):**

Users upload a `.csv` file. The app automatically detects whether the CSV contains all 28 raw features (full pipeline: IQR cap → impute → select → scale) or just the 14 selected features (scale only). This dual-mode design means the same endpoint can handle:

- Raw exports from a credit bureau system (28 columns)
- Pre-processed files generated by a separate ETL pipeline (14 columns)

If required columns are missing, the app returns a clear error listing the missing column names by name.

**Output columns appended to the uploaded CSV:**

| Column                | Type           | Description                                   |
| --------------------- | -------------- | --------------------------------------------- |
| `pred_default_prob` | float (4 d.p.) | Raw model probability                         |
| `pred_decision`     | str            | "APPROVE" or "REJECT" at threshold 0.5        |
| `confidence_tier`   | str            | "🟢 Low Risk", "🟡 Review", or "🔴 High Risk" |

A preview of the first 50 rows is shown in-browser via `gr.Dataframe`.

### Tab 2: 📊 Model Comparison

This tab trains XGBoost and LightGBM fresh from the pre-saved `split_data.pkl` arrays and evaluates all three models on the same 600-row test set. Results are displayed as a styled HTML table:

| Column         | Best =                   |
| -------------- | ------------------------ |
| F1-Score       | Highest (green)          |
| ROC-AUC        | Highest (green)          |
| Inference (ms) | Lowest (green = fastest) |

The table loads automatically when the tab is first viewed, and a **Refresh / Re-evaluate** button clears the cache and retrains XGBoost and LightGBM from scratch (useful if hyperparameters are changed in source).

---

## 9. Tech Stack

* LibraryVersionRole in this project**pandas**latestExcel ingestion (`read_excel`), DataFrame operations throughout pre-processing and inference**numpy**latestArray operations, class weight calculation, SHAP value indexing**scikit-learn**1.5.1`KNNImputer`, `StandardScaler`, `RandomForestClassifier`, `train_test_split`, `f1_score`, `roc_auc_score`**xgboost**≥ 2.0Gradient-boosted tree model for comparison,`scale_pos_weight` for imbalance**lightgbm**≥ 4.0Histogram-based gradient boosting for comparison, fast inference**shap**≥ 0.45`TreeExplainer` for exact SHAP values on the Random Forest; bar-chart generation**matplotlib**≥ 3.7Rendering SHAP bar charts to in-memory PNG buffers (Agg backend, headless)**Pillow (PIL)**≥ 9.0Converting matplotlib figure buffers to `PIL.Image` for `gr.Image` output**gradio**≥ 4.0Interactive web UI —`gr.Blocks`, `gr.Tab`, `gr.Number`, `gr.HTML`, `gr.Image`, `gr.File`**joblib**latestFast serialisation and deserialisation of all sklearn fitted objects**openpyxl**≥ 3.1Excel engine backend required by `pandas.read_excel` for `.xlsx` files**httpx / safehttpx**latestAsync HTTP transport used internally by Gradio

---

## 10. How to Run

### Prerequisites

- Python 3.10+
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/credit-risk-prediction.git
cd credit-risk-prediction
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add the Dataset

Place `data.xlsx` in the project root. The file must contain a `TARGET` column (binary 0/1) and an `ID` column. All other columns are treated as features.

```
credit-risk-prediction/
├── data.xlsx          ← place here
├── app.py
├── build_pipeline.py
├── requirements.txt
└── ...
```

### Step 5 — Build the Preprocessing Pipeline

Run this **once** to fit all transformers, select features, train the model, and save all `.pkl` artifacts:

```bash
python build_pipeline.py
```

Expected output:

```
============================================================
Building credit risk preprocessing pipeline
============================================================

[1] Loading data.xlsx ...         Raw shape: (3000, 30)
[2] Identifying target column ... Target column: 'TARGET'
[3] Dropping high-null cols ...   None found — all columns kept.
[4] Splitting 80/20 ...           Train: (2400, 28), Test: (600, 28)
[5] Fitting IQR bounds ...
[6] Fitting KNNImputer ...
[7] Selecting top 14 features ... ['TLBalHCPct', 'TLSatPct', ...]
[8] Fitting StandardScaler ...
[9] Training RandomForest ...     RF — F1: 0.1552, AUC: 0.7506
[10] Saving artifacts ...

Pipeline build complete!
```

This produces six files in the project root:

```
credit_risk_model.pkl  imputer.pkl  scaler.pkl
features.pkl  iqr_bounds.pkl  split_data.pkl
```

> **Note:** Re-run `build_pipeline.py` any time `data.xlsx` is updated. All `.pkl` files will be overwritten with fresh fits.

### Step 6 — Launch the App

```bash
python app.py
```

Open your browser at **http://localhost:7860**

### Batch Prediction

To score a portfolio of applicants, prepare a CSV with either:

**Option A — 14 selected features (recommended for production pipelines):**

```
TLBalHCPct,TLSatPct,TLTimeFirst,TLDel3060Cnt24,TLSum,TLMaxSum,TLOpenPct,
TLDel60Cnt24,TLSatCnt,TLDel60CntAll,TLTimeLast,InqFinanceCnt24,TL75UtilCnt,TLOpen24Pct
```

**Option B — all 28 raw features (direct from credit bureau export):**
Include all original feature columns; the app will apply IQR capping, KNN imputation, feature selection, and scaling automatically.

Upload via the **Batch CSV Prediction** section in the Single Prediction tab. The output CSV (previewed in-browser) will include `pred_default_prob`, `pred_decision`, and `confidence_tier` columns appended to each row.

---

## Project Structure

```
credit-risk-prediction/
│
├── app.py                  # Gradio application — inference + UI
├── build_pipeline.py       # Offline pipeline fitting script — run once
├── requirements.txt        # Python dependencies
├── samples.csv             # 16 sample applicant profiles for UI demo
│
├── data.xlsx               # Raw dataset (not committed — see .gitignore)
│
├── credit_risk_model.pkl   # Fitted RandomForestClassifier
├── imputer.pkl             # Fitted KNNImputer (28 features)
├── scaler.pkl              # Fitted StandardScaler (14 features)
├── features.pkl            # Selected feature name list
├── iqr_bounds.pkl          # IQR clip bounds per column
└── split_data.pkl          # Pre-split arrays for model comparison tab
```

---

## License

MIT License — see `LICENSE` for details.
