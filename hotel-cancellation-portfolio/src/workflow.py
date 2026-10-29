"""Reproducible provisional competition workflow; no external data or hidden labels."""

from pathlib import Path
import argparse, hashlib, json, time, platform, warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from threadpoolctl import threadpool_limits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 835
# These may describe changes AFTER initial booking; exclude in the conservative view.
TIMING_UNCERTAIN = [
    "booking_changes",
    "days_in_waiting_list",
    "required_car_parking_spaces",
    "total_of_special_requests",
]


class BookingFeatures(BaseEstimator, TransformerMixin):
    """Only row-level arithmetic; no statistics learned from validation or test data."""

    def __init__(
        self,
        categorical_ids=True,
        engineered=False,
        conservative=False,
        drop_country=False,
        drop_deposit=False,
    ):
        self.categorical_ids = categorical_ids
        self.engineered = engineered
        self.conservative = conservative
        self.drop_country = drop_country
        self.drop_deposit = drop_deposit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X.copy().drop(columns=["id", "is_canceled"], errors="ignore")
        if self.conservative:
            x = x.drop(columns=TIMING_UNCERTAIN, errors="ignore")
        if self.drop_country:
            x = x.drop(columns="country")
        if self.drop_deposit:
            x = x.drop(columns="deposit_type")
        if self.categorical_ids:
            for c in ["agent", "company"]:
                x[c] = x[c].map(lambda v: "__MISSING__" if pd.isna(v) else str(int(v)))
        for c in x.select_dtypes(include=["object", "string"]).columns:
            x[c] = x[c].fillna("__MISSING__").astype(str)
        if self.engineered:
            x["total_nights"] = x.stays_in_weekend_nights + x.stays_in_week_nights
            # Missing children stays missing; imputation is fitted inside each fold.
            x["total_guests"] = x.adults + x.children + x.babies
            x["has_children"] = np.where(
                x.children.isna(), np.nan, ((x.children + x.babies) > 0).astype(float)
            )
            x["weekend_share"] = x.stays_in_weekend_nights / x.total_nights.replace(
                0, np.nan
            )
        return x


def make_pipeline(
    kind="lr",
    categorical_ids=True,
    engineered=False,
    conservative=False,
    drop_country=False,
    drop_deposit=False,
    C=1.0,
    min_samples_leaf=5,
    max_leaf_nodes=15,
    l2_regularization=1.0,
):
    features = BookingFeatures(
        categorical_ids, engineered, conservative, drop_country, drop_deposit
    )

    def numeric(x):
        return list(x.select_dtypes(include=np.number).columns)

    def categorical(x):
        return list(x.select_dtypes(exclude=np.number).columns)

    nums = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    if kind == "hgb":
        # Fold-fitted rare grouping, <= 255 codes incl. missing; unknown=-1 is treated as missing.
        cats = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            min_frequency=20,
            max_categories=64,
            dtype=np.float64,
        )
    else:
        cats = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=20,
            sparse_output=True,
            dtype=np.float64,
        )
    prep = ColumnTransformer(
        [("num", nums, numeric), ("cat", cats, categorical)],
        sparse_threshold=0 if kind == "hgb" else 1.0,
    )
    if kind == "lr":
        model = LogisticRegression(
            C=C, solver="lbfgs", max_iter=1500, random_state=SEED
        )
    elif kind == "rf":
        model = RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            n_jobs=2,
            random_state=SEED,
        )
    elif kind == "hgb":
        # Explicit categorical mask is set after the preprocessor is fitted (see fit_model).
        model = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            early_stopping=False,
            random_state=SEED,
            categorical_features=None,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("features", features), ("preprocess", prep), ("model", model)])


def fit_model(pipe, x, y):
    # Fit every learned step on this training partition only. A Pipeline is retained for prediction.
    with threadpool_limits(limits=2):
        if isinstance(pipe["model"], HistGradientBoostingClassifier):
            a = pipe["features"].fit_transform(x, y)
            z = pipe["preprocess"].fit_transform(a, y)
            cat_slice = pipe["preprocess"].output_indices_["cat"]
            mask = np.zeros(z.shape[1], dtype=bool)
            mask[cat_slice] = True
            pipe["model"].set_params(categorical_features=mask)
            pipe["model"].fit(z, y)
        else:
            pipe.fit(x, y)
    return pipe


def audit(train, test, out):
    assert train.shape == (83573, 30) and test.shape == (35817, 29)
    assert set(train.is_canceled.unique()) == {0, 1}
    assert list(train.drop(columns="is_canceled").columns) == list(test.columns)
    assert train.id.notna().all() and test.id.notna().all()
    assert train.id.is_unique and test.id.is_unique
    assert not set(train.id) & set(test.id)
    cols = [c for c in test if c != "id"]
    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target_counts": train.is_canceled.value_counts().sort_index().to_dict(),
        "cancellation_rate": float(train.is_canceled.mean()),
        "train_duplicate_full_rows": int(train.duplicated().sum()),
        "train_duplicate_predictors_excluding_id": int(train.duplicated(cols).sum()),
        "test_duplicate_predictors_excluding_id": int(test.duplicated(cols).sum()),
        "test_rows_matching_a_train_predictor_profile": int(
            pd.util.hash_pandas_object(test[cols], index=False)
            .isin(pd.util.hash_pandas_object(train[cols], index=False))
            .sum()
        ),
        "id_unique_train": bool(train.id.is_unique),
        "id_unique_test": bool(test.id.is_unique),
        "id_overlap": 0,
    }
    (out / "audit.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        {
            "dtype": train.dtypes.astype(str),
            "train_missing": train.isna().sum(),
            "train_missing_pct": train.isna().mean() * 100,
            "test_missing": test.isna().sum(),
            "train_unique": train.nunique(),
        }
    ).to_csv(out / "column_audit.csv")
    for col in ["deposit_type", "country"]:
        train.assign(category=train[col].fillna("__MISSING__")).groupby(
            "category"
        ).is_canceled.agg(["size", "mean"]).sort_values("size", ascending=False).to_csv(
            out / f"{col}_association.csv"
        )
    return summary


EXPERIMENTS = [
    (
        "01_provisional_lr_numeric_ids",
        dict(kind="lr", categorical_ids=False),
        "Provisional baseline only; official starter is missing. Numeric IDs test reference.",
    ),
    (
        "02_lr_categorical_ids",
        dict(kind="lr"),
        "Change agent/company to categories; group rare levels using training fold only.",
    ),
    (
        "03_lr_features",
        dict(kind="lr", engineered=True),
        "Test total nights, guests, has children and weekend share.",
    ),
    (
        "04_lr_features_C01",
        dict(kind="lr", engineered=True, C=0.1),
        "Small regularization search: C 1 versus 0.1.",
    ),
    (
        "05_rf_leaf5",
        dict(kind="rf", engineered=True),
        "Nonlinear tree reference; 160 trees and leaf size 5 control cost/overfit.",
    ),
    (
        "06_hgb_leaf15",
        dict(kind="hgb", engineered=True),
        "Boosted trees with native categorical preprocessing; modest 15 leaves.",
    ),
    (
        "07_hgb_leaf31",
        dict(kind="hgb", engineered=True, max_leaf_nodes=31),
        "Small capacity search: 15 versus 31 leaves.",
    ),
    (
        "08_hgb_conservative",
        dict(kind="hgb", engineered=True, conservative=True),
        "Sensitivity: remove four features with uncertain booking-time availability.",
    ),
    (
        "09_hgb_no_country",
        dict(kind="hgb", engineered=True, drop_country=True),
        "Country ablation: measure predictive association, not causation.",
    ),
    (
        "10_hgb_no_deposit",
        dict(kind="hgb", engineered=True, drop_deposit=True),
        "Deposit ablation: check reliance on unusual deposit cancellation association.",
    ),
]


def evaluate(name, params, reason, x, y, folds, out):
    pred = np.full(len(y), np.nan)
    rows = []
    for fold, (tr, va) in enumerate(folds, 1):
        start = time.perf_counter()
        p = make_pipeline(**params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_model(p, x.iloc[tr], y.iloc[tr])
            with threadpool_limits(limits=2):
                scores = p.predict_proba(x.iloc[va])[:, 1]
        pred[va] = scores
        row = {
            "experiment": name,
            "fold": fold,
            "auc": roc_auc_score(y.iloc[va], scores),
            "seconds": time.perf_counter() - start,
            "settings": json.dumps(params, sort_keys=True),
            "reason": reason,
            "warnings": " | ".join(sorted(set(str(w.message) for w in caught))),
        }
        rows.append(row)
        print(name, fold, round(row["auc"], 6), round(row["seconds"], 1), flush=True)
        pd.DataFrame(rows).to_csv(out / f"{name}_folds.csv", index=False)
    assert np.isfinite(pred).all()
    np.save(out.parent / "private" / f"{name}_oof.npy", pred)
    return rows, pred


def run(data_dir, root, team="TeamHH"):
    root = Path(root)
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (root / "private").mkdir(exist_ok=True)
    train = pd.read_csv(Path(data_dir) / "train.csv")
    test = pd.read_csv(Path(data_dir) / "test.csv")
    summary = audit(train, test, out)
    print(json.dumps(summary, indent=2), flush=True)
    x = train.drop(columns="is_canceled")
    y = train.is_canceled
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, y)
    )
    fold_id = np.zeros(len(y), dtype=int)
    for i, (_, v) in enumerate(folds, 1):
        fold_id[v] = i
    np.save(root / "private" / "fold_assignments.npy", fold_id)
    all_rows = []
    oofs = {}
    for name, params, reason in EXPERIMENTS:
        rows, pred = evaluate(name, params, reason, x, y, folds, out)
        all_rows += rows
        oofs[name] = pred
        pd.DataFrame(all_rows).to_csv(out / "fold_scores.csv", index=False)
        tab = (
            pd.DataFrame(all_rows)
            .groupby("experiment", sort=False)
            .agg(
                mean_auc=("auc", "mean"),
                std_auc=("auc", "std"),
                seconds=("seconds", "sum"),
            )
        )
        tab.to_csv(out / "model_comparison.csv")
    # Treat ablations as diagnostics, not additional candidates for leaderboard-style selection.
    candidates = [e[0] for e in EXPERIMENTS[:7]]
    winner = tab.loc[candidates, "mean_auc"].idxmax()
    params = next(p for n, p, r in EXPERIMENTS if n == winner)
    pred = oofs[winner]
    fig, ax = plt.subplots(figsize=(7, 4))
    tab.mean_auc.plot.barh(ax=ax, xerr=tab.std_auc)
    ax.set_xlabel("Mean validation ROC-AUC (error bars: fold standard deviation)")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    fpr, tpr, _ = roc_curve(y, pred)
    ax.plot(fpr, tpr, label=f"OOF AUC {roc_auc_score(y,pred):.4f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "validation_roc.png", dpi=140)
    plt.close(fig)
    observed, estimated = calibration_curve(y, pred, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(estimated, observed, "o-")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(
        xlabel="Average predicted probability", ylabel="Observed cancellation fraction"
    )
    fig.tight_layout()
    fig.savefig(out / "validation_calibration.png", dpi=140)
    plt.close(fig)
    for c in ["deposit_type", "country"]:
        a = pd.read_csv(out / f"{c}_association.csv").head(12)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(a.category, a["mean"])
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel("Observed cancellation fraction")
        fig.tight_layout()
        fig.savefig(out / f"{c}_association.png", dpi=140)
        plt.close(fig)
    # Duplicate sensitivity: keep identical feature profiles in the same fold.
    cols = [c for c in x if c != "id"]
    groups = pd.util.hash_pandas_object(x[cols], index=False)
    gf = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            x, y, groups
        )
    )
    gr, gpred = evaluate(
        "11_winner_grouped_diagnostic",
        params,
        "Sensitivity check only: exact predictor profiles cannot cross train/validation.",
        x,
        y,
        gf,
        out,
    )
    pd.DataFrame(gr).to_csv(out / "grouped_sensitivity.csv", index=False)
    model = fit_model(make_pipeline(**params), x, y)
    with threadpool_limits(limits=2):
        probabilities = model.predict_proba(test)[:, 1]
    submission = pd.DataFrame({"id": test.id.to_numpy(), "is_canceled": probabilities})
    path = root / "private" / f"{team}_submission.csv"
    submission.to_csv(path, index=False)
    saved = pd.read_csv(path)
    assert saved.shape == (35817, 2) and list(saved) == ["id", "is_canceled"]
    assert saved.id.is_unique and saved.id.notna().all() and saved.id.equals(test.id)
    assert set(saved.id) == set(test.id)
    assert (
        np.isfinite(saved.is_canceled).all() and saved.is_canceled.between(0, 1).all()
    )
    sample = Path(data_dir) / "sample_submission.csv"
    if sample.exists():
        s = pd.read_csv(sample)
        assert list(s) == list(saved) and s.id.is_unique and set(s.id) == set(saved.id)
    validation = {
        "file": path.name,
        "rows": len(saved),
        "columns": list(saved),
        "id_order_matches_test": True,
        "all_checks_passed": True,
        "sample_submission_verified": sample.exists(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (root / "private" / "submission_validation.json").write_text(
        json.dumps(validation, indent=2)
    )
    final = {
        "selected_model": winner,
        "settings": params,
        "local_mean_auc": float(tab.loc[winner, "mean_auc"]),
        "local_std_auc": float(tab.loc[winner, "std_auc"]),
        "pooled_oof_auc": roc_auc_score(y, pred),
        "brier_score": brier_score_loss(y, pred),
        "grouped_diagnostic_mean_auc": float(np.mean([r["auc"] for r in gr])),
        "public_kaggle_auc": None,
        "private_kaggle_auc": None,
        "official_baseline_reproduced": False,
        "course_method_compliance_verified": False,
        "memo_rubric_verified": False,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "train_sha256": hashlib.sha256(
            (Path(data_dir) / "train.csv").read_bytes()
        ).hexdigest(),
        "test_sha256": hashlib.sha256(
            (Path(data_dir) / "test.csv").read_bytes()
        ).hexdigest(),
    }
    (out / "results.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2), flush=True)
    return final


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--data-dir", default="data")
    a.add_argument("--root", default=".")
    a.add_argument("--team", default="TeamHH")
    args = a.parse_args()
    run(args.data_dir, args.root, args.team)
