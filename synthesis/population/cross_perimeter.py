import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from data.hts.edgt_74.adisp_merge.merge import EDUCATION_ORDINAL_MAP

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

"""
Predicts cross-perimeter likelihood (is_crossperim_person) from the merged
Annemasse/Annecy EDGT persons (data.hts.edgt_74.adisp_merge.merge).

Ported from Documents/Tests/EDGT/EDGT_cross_perimeter/crossperim_model.py:
two models are compared:
  1. Logistic regression (statsmodels) - interpretability: odds ratios, p-values
  2. CatBoost                         - predictive accuracy + feature importance
                                        (requires: pip install catboost)

Both use survey weights (person_weight, i.e. COEP). No oversampling -> the
mean predicted probability stays calibrated against the actual crossing share.

Only 5 features are used on purpose: the event is rare and the sample is
small, so a richer model would not be trustworthy. Features are picked to
keep one variable per causal channel (geography, household car ownership,
individual mobility resource, socio-economic status) and prefer ordinal/
binary encodings over high-cardinality dummies to keep the parameter count
low. Mode-use-frequency variables are deliberately excluded even though they
would rank highly: they describe travel behaviour, not a person/household
characteristic, and are close enough to the outcome itself (crossing trips)
to leak information rather than explain it.

Outputs saved to context.path():
  logit_coefficients.csv   odds ratios, 95 % CI, p-values
  model_metrics.csv        AUC-ROC, avg precision, Brier score, mean pred. prob.
  or_forest_plot.png       forest plot of logit odds ratios
  roc_curves.png           ROC curves from 5-fold CV
  calibration_plot.png     calibration from 5-fold CV
  feature_importance.png   CatBoost gain-based importance (if installed)
"""

warnings.filterwarnings("ignore")

TARGET = "is_crossperim_person"
WEIGHT = "person_weight"

COLOR_LOGIT = "#2a78d6"
COLOR_CATBOOST = "#e34948"

# Keys = internal column names; values = display labels
FEATURE_LABELS = {
    "log_dist_perimeter":  "log(distance to perimeter border + 1 m)",
    "number_of_cars":      "Number of cars in household",
    "is_annemasse":        "Survey = Annemasse (vs Annecy)",
    "has_driving_license": "Has driving license",
    "educ_ord":            "Education (0=primary/none … 3=higher ed. bac+3+)",
}


def configure(context):
    context.stage("data.hts.edgt_74.adisp_merge.merge")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare(context):
    df_households, df_persons, _ = context.stage("data.hts.edgt_74.adisp_merge.merge")

    df = df_persons.merge(
        df_households[["household_id", "M6"]].rename(columns = { "M6": "number_of_cars" }),
        on = "household_id", how = "left"
    )

    df["log_dist_perimeter"]  = np.log1p(df["distance_to_perimeter_border"])
    df["is_annemasse"]        = (df["edgt_area"] == "annemasse").astype(int)
    df["has_driving_license"] = df["has_license"].astype(int)
    df["educ_ord"]            = pd.to_numeric(df["P8"], errors = "coerce").map(EDUCATION_ORDINAL_MAP)
    df[TARGET]                = df[TARGET].astype(int)

    features = list(FEATURE_LABELS.keys())
    df_model = df[[TARGET, WEIGHT] + features].dropna()

    n_total = len(df_model)
    n_cross = int(df_model[TARGET].sum())
    w_share = (df_model.loc[df_model[TARGET] == 1, WEIGHT].sum()
               / df_model[WEIGHT].sum() * 100)
    print(f"Rows (after dropna)  : {n_total:,}")
    print(f"Cross-perimeter      : {n_cross:,} ({n_cross / n_total * 100:.1f} % unweighted)")
    print(f"Weighted share       : {w_share:.2f} %\n")

    return df_model, features


# ---------------------------------------------------------------------------
# Logistic regression (statsmodels) - full-data inference table
# ---------------------------------------------------------------------------

def fit_logit_table(df, features):
    """Weighted logit on the full dataset; returns an odds-ratio table and the
    raw fitted coefficients (indexed "const" + features) needed to score new
    observations, e.g. from synthesis.population.enriched."""
    y      = df[TARGET].values.astype(float)
    X      = df[features].values.astype(float)
    w      = df[WEIGHT].values
    w_norm = w / w.mean()   # normalise so effective N ~ actual sample size

    X_c    = sm.add_constant(X)
    result = sm.GLM(y, X_c,
                    family = sm.families.Binomial(),
                    var_weights = w_norm).fit(disp = False)

    params = pd.Series(result.params, index = ["const"] + features)

    coefs = result.params[1:]          # skip intercept
    ci    = np.array(result.conf_int())[1:]
    pvals = result.pvalues[1:]

    table = pd.DataFrame({
        "feature":  features,
        "label":    [FEATURE_LABELS[f] for f in features],
        "coef":     coefs,
        "OR":       np.exp(coefs),
        "CI_lower": np.exp(ci[:, 0]),
        "CI_upper": np.exp(ci[:, 1]),
        "p_value":  pvals,
    })
    table["sig"] = table["p_value"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01
        else "*" if p < 0.05 else "ns"
    )
    return table, params


# ---------------------------------------------------------------------------
# 5-fold cross-validated evaluation
# ---------------------------------------------------------------------------

def run_cv(df, features, n_splits = 5):
    """Return out-of-fold predicted probabilities and summary metrics."""
    X = df[features].values.astype(float)
    y = df[TARGET].values.astype(int)
    w = df[WEIGHT].values

    cv  = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    oof = {"Logit": np.zeros(len(y))}
    if HAS_CATBOOST:
        oof["CatBoost"] = np.zeros(len(y))

    for fold, (tr, te) in enumerate(cv.split(X, y), 1):
        print(f"  Fold {fold}/{n_splits} …")
        X_tr, X_te = X[tr], X[te]
        y_tr, w_tr = y[tr], w[tr]

        # Logistic regression (C very large ~ no regularisation)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C = 1e6, max_iter = 1000, solver = "lbfgs")),
        ])
        pipe.fit(X_tr, y_tr, clf__sample_weight = w_tr)
        oof["Logit"][te] = pipe.predict_proba(X_te)[:, 1]

        if HAS_CATBOOST:
            # Shallow, few iterations on purpose: only 5 features and a
            # handful of positive events per fold, a deep/long-boosted model
            # would just memorise noise.
            cb = CatBoostClassifier(
                iterations = 200, learning_rate = 0.05, depth = 3,
                eval_metric = "AUC", verbose = 0, random_seed = 42,
            )
            cb.fit(X_tr, y_tr, sample_weight = w_tr)
            oof["CatBoost"][te] = cb.predict_proba(X_te)[:, 1]

    actual_share = np.average(y, weights = w) * 100
    metrics = {}
    for name, proba in oof.items():
        metrics[name] = {
            "ROC-AUC":              roc_auc_score(y, proba, sample_weight = w),
            "Avg Precision":        average_precision_score(y, proba, sample_weight = w),
            "Brier score":          brier_score_loss(y, proba, sample_weight = w),
            "Mean predicted P (%)": np.average(proba, weights = w) * 100,
            "Actual share (%)":     actual_share,
        }
        print(f"\n  {name} results:")
        for k, v in metrics[name].items():
            print(f"    {k:<28} {v:.4f}")

    return oof, y, w, metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_or_forest(table, results_dir):
    """Horizontal forest plot of odds ratios (log scale)."""
    t      = table.sort_values("OR")
    labels = t["label"].tolist()
    ors    = t["OR"].values
    xerr   = np.array([ors - t["CI_lower"].values, t["CI_upper"].values - ors])

    fig, ax = plt.subplots(figsize = (8, max(4, 0.5 * len(labels) + 1.5)))
    ax.errorbar(ors, np.arange(len(labels)), xerr = xerr,
                fmt = "o", color = COLOR_LOGIT, capsize = 4, lw = 1.2, markersize = 6)
    ax.axvline(1.0, color = "gray", linestyle = "--", lw = 1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize = 9)
    ax.set_xlabel("Odds ratio (95 % CI, log scale)")
    ax.set_title("Logistic regression — odds ratios\n(weighted, full dataset)")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "or_forest_plot.png"), dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    print("  Saved: or_forest_plot.png")


def plot_roc(oof, y, w, results_dir):
    colors = {"Logit": COLOR_LOGIT, "CatBoost": COLOR_CATBOOST}
    fig, ax = plt.subplots(figsize = (5, 5))
    for name, proba in oof.items():
        fpr, tpr, _ = roc_curve(y, proba, sample_weight = w)
        auc = roc_auc_score(y, proba, sample_weight = w)
        ax.plot(fpr, tpr, label = f"{name}  (AUC = {auc:.3f})",
                color = colors.get(name, "black"), lw = 1.8)
    ax.plot([0, 1], [0, 1], "k--", lw = 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (5-fold CV)")
    ax.legend(fontsize = 9)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "roc_curves.png"), dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    print("  Saved: roc_curves.png")


def plot_calibration(oof, y, w, results_dir):
    """Quantile-binned calibration plot (robust at low prevalence)."""
    colors = {"Logit": COLOR_LOGIT, "CatBoost": COLOR_CATBOOST}
    n_bins = 10
    fig, ax = plt.subplots(figsize = (5, 5))

    p_max = 0.0
    for name, proba in oof.items():
        # Quantile bins so each bin has equal weight
        quantiles = np.percentile(proba, np.linspace(0, 100, n_bins + 1))
        quantiles[0], quantiles[-1] = 0.0, 1.0
        bin_idx = np.digitize(proba, quantiles[1:-1])

        mean_pred, mean_obs = [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() < 10:
                continue
            wb = w[mask]
            mean_pred.append(np.average(proba[mask], weights = wb))
            mean_obs.append(np.average(y[mask].astype(float), weights = wb))

        ax.plot(mean_pred, mean_obs, "o-", label = name,
                color = colors.get(name, "black"), lw = 1.5, markersize = 5)
        p_max = max(p_max, max(mean_pred + mean_obs, default = 0))

    upper = min(1.0, p_max * 1.15 + 0.01)
    ax.plot([0, upper], [0, upper], "k--", lw = 1, label = "Perfect calibration")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration plot (5-fold CV)")
    ax.legend(fontsize = 9)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "calibration_plot.png"), dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    print("  Saved: calibration_plot.png")


def plot_feature_importance(df, features, results_dir):
    """Fit CatBoost on the full dataset and plot gain-based feature importance."""
    if not HAS_CATBOOST:
        return
    X = df[features].values.astype(float)
    y = df[TARGET].values.astype(int)
    w = df[WEIGHT].values

    cb = CatBoostClassifier(
        iterations = 200, learning_rate = 0.05, depth = 3, verbose = 0, random_seed = 42
    )
    cb.fit(X, y, sample_weight = w)

    imp = pd.Series(cb.get_feature_importance(),
                    index = [FEATURE_LABELS[f] for f in features]).sort_values()

    fig, ax = plt.subplots(figsize = (7, max(4, 0.5 * len(features) + 1.5)))
    ax.barh(imp.index, imp.values, color = COLOR_CATBOOST, alpha = 0.85)
    ax.set_xlabel("Feature importance (gain %)")
    ax.set_title("CatBoost — feature importance (full dataset, weighted)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "feature_importance.png"), dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    print("  Saved: feature_importance.png")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_model(df, features, results_dir):
    n       = len(df)
    n_cross = int(df[TARGET].sum())
    w_share = df.loc[df[TARGET] == 1, WEIGHT].sum() / df[WEIGHT].sum() * 100
    print(f"Rows : {n:,}  |  Cross-perimeter : {n_cross:,}  ({w_share:.2f} % weighted)")

    print("\n--- Logistic regression (statsmodels) ---")
    or_table, params = fit_logit_table(df, features)
    print(or_table[["label", "OR", "CI_lower", "CI_upper", "p_value", "sig"]]
          .to_string(index = False, float_format = lambda x: f"{x:.4f}"))
    or_table.to_csv(os.path.join(results_dir, "logit_coefficients.csv"), index = False)
    print("  Saved: logit_coefficients.csv")

    params.to_csv(os.path.join(results_dir, "logit_params.csv"), header = ["coef"])
    print("  Saved: logit_params.csv")

    print("\n--- 5-fold cross-validation ---")
    oof, y, w, metrics = run_cv(df, features)

    print("\n--- Saving plots ---")
    plot_or_forest(or_table, results_dir)
    plot_roc(oof, y, w, results_dir)
    plot_calibration(oof, y, w, results_dir)
    plot_feature_importance(df, features, results_dir)

    metrics_df = pd.DataFrame(metrics).T.round(4)
    metrics_df.to_csv(os.path.join(results_dir, "model_metrics.csv"))
    print("  Saved: model_metrics.csv")

    return or_table, metrics_df, params


def execute(context):
    if not HAS_CATBOOST:
        print("[info] CatBoost not found — only logistic regression will run.\n"
              "       Install with: pip install catboost\n")

    results_dir = context.path()

    df_model, features = load_and_prepare(context)
    or_table, metrics_df, params = run_model(df_model, features, results_dir)

    return or_table, metrics_df, params
