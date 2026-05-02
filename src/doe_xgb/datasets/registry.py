"""Registry of the twelve article-track v1 datasets.

This module is intentionally import-light: no network access and no
pandas / sklearn imports happen on module load. Loaders pull heavy
dependencies on demand.
"""

from __future__ import annotations

from .metadata import DatasetMetadata

REGISTRY: dict[str, DatasetMetadata] = {
    "magic": DatasetMetadata(
        dataset_id="magic",
        display_name="MAGIC Gamma Telescope",
        source_type="uci",
        source_url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
        ),
        openml_id=1120,  # OpenML mirror id (may change; confirm at use time)
        task_type="binary",
        target_column="class",
        target_transform="map:{g:0,h:1}",
        categorical_columns=(),
        numeric_columns=(
            "fLength",
            "fWidth",
            "fSize",
            "fConc",
            "fConc1",
            "fAsym",
            "fM3Long",
            "fM3Trans",
            "fAlpha",
            "fDist",
        ),
        missing_value_policy="keep",
        burden="medium",
        license_note="UCI ML Repository (CC BY 4.0).",
        citation_key="uci_magic",
        include_in_v1=True,
        notes="Dissertation continuity dataset.",
    ),
    "breast_cancer": DatasetMetadata(
        dataset_id="breast_cancer",
        display_name="Breast Cancer Wisconsin (Diagnostic)",
        source_type="sklearn",
        source_url=None,
        task_type="binary",
        target_column="target",
        target_transform=None,
        categorical_columns=(),
        numeric_columns=(),  # filled at load time
        missing_value_policy="keep",
        burden="light",
        license_note="UCI ML Repository (CC BY 4.0); shipped with sklearn.",
        citation_key="sklearn_breast_cancer",
        include_in_v1=True,
        notes="Small classical sanity check; loaded directly from sklearn.",
    ),
    "pima_diabetes": DatasetMetadata(
        dataset_id="pima_diabetes",
        display_name="Pima Indians Diabetes",
        source_type="openml",
        openml_id=37,  # Diabetes (a.k.a. pima) at OpenML
        task_type="binary",
        target_column="class",
        target_transform="binary",
        numeric_columns=(
            "preg",
            "plas",
            "pres",
            "skin",
            "insu",
            "mass",
            "pedi",
            "age",
        ),
        missing_value_policy="impute_median",
        burden="light",
        license_note="OpenML / UCI mirror (CC BY 4.0).",
        citation_key="openml_pima",
        include_in_v1=True,
        notes=(
            "Original PimaIndiansDiabetes (UCI) was retracted; OpenML id 37 "
            "is the standard 'diabetes' mirror."
        ),
    ),
    "spambase": DatasetMetadata(
        dataset_id="spambase",
        display_name="Spambase",
        source_type="uci",
        source_url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        ),
        openml_id=44,
        task_type="binary",
        target_column="is_spam",
        target_transform=None,
        numeric_columns=(),  # 57 numeric columns; populated by loader
        missing_value_policy="keep",
        burden="light",
        license_note="UCI ML Repository (CC BY 4.0).",
        citation_key="uci_spambase",
        include_in_v1=True,
    ),
    "adult": DatasetMetadata(
        dataset_id="adult",
        display_name="Adult / Census Income",
        source_type="uci",
        source_url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        ),
        openml_id=1590,
        task_type="binary",
        target_column="income",
        target_transform="binary>50K",
        categorical_columns=(
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
        ),
        numeric_columns=(
            "age",
            "fnlwgt",
            "education_num",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
        ),
        missing_value_policy="drop_unknown",
        burden="medium",
        license_note="UCI ML Repository (CC BY 4.0).",
        citation_key="uci_adult",
        include_in_v1=True,
        notes="Native categorical handling for CatBoost; one-hot for XGBoost / LightGBM.",
    ),
    "bank_marketing": DatasetMetadata(
        dataset_id="bank_marketing",
        display_name="Bank Marketing",
        source_type="uci",
        source_url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
        ),
        openml_id=1461,
        task_type="binary",
        target_column="y",
        target_transform="binary_yes",
        categorical_columns=(
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ),
        numeric_columns=(
            "age",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
        ),
        missing_value_policy="drop_unknown",
        burden="medium",
        license_note="UCI ML Repository (CC BY 4.0).",
        citation_key="uci_bank",
        include_in_v1=True,
        notes="Use bank-additional-full variant.",
    ),
    "credit_card_default": DatasetMetadata(
        dataset_id="credit_card_default",
        display_name="Default of Credit Card Clients",
        source_type="uci",
        source_url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
            "default%20of%20credit%20card%20clients.xls"
        ),
        openml_id=42477,
        task_type="binary",
        target_column="default_payment_next_month",
        target_transform=None,
        numeric_columns=(),  # 23 features; populated by loader
        missing_value_policy="keep",
        burden="medium",
        license_note="UCI ML Repository (CC BY 4.0).",
        citation_key="uci_credit_default",
        include_in_v1=True,
        notes="XLS source; loader normalizes to CSV.",
    ),
    "german_credit": DatasetMetadata(
        dataset_id="german_credit",
        display_name="German Credit",
        source_type="uci",
        source_url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        ),
        openml_id=31,
        task_type="binary",
        target_column="risk",
        target_transform="binary_bad",
        categorical_columns=(
            "checking_status",
            "credit_history",
            "purpose",
            "savings_status",
            "employment",
            "personal_status",
            "other_parties",
            "property_magnitude",
            "other_payment_plans",
            "housing",
            "job",
            "own_telephone",
            "foreign_worker",
        ),
        numeric_columns=(
            "duration",
            "credit_amount",
            "installment_commitment",
            "residence_since",
            "age",
            "existing_credits",
            "num_dependents",
        ),
        missing_value_policy="keep",
        burden="light",
        license_note="UCI ML Repository (CC BY 4.0).",
        citation_key="uci_german_credit",
        include_in_v1=True,
    ),
    "wine_quality": DatasetMetadata(
        dataset_id="wine_quality",
        display_name="Wine Quality (binarised, red+white)",
        source_type="uci",
        source_url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        ),
        openml_id=287,
        task_type="binary",
        target_column="quality",
        target_transform="binary_quality_ge_6",
        numeric_columns=(
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ),
        missing_value_policy="keep",
        burden="light",
        license_note="UCI ML Repository (CC BY 4.0).",
        citation_key="uci_wine_quality",
        include_in_v1=True,
        notes=(
            "Article v1 binarizes quality at >=6 to keep the headline panel "
            "binary. A multiclass variant is deferred to a follow-up paper."
        ),
    ),
    "dry_bean": DatasetMetadata(
        dataset_id="dry_bean",
        display_name="Dry Bean",
        source_type="uci",
        source_url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip"
        ),
        openml_id=43466,
        task_type="multiclass",
        target_column="Class",
        target_transform="label_encode",
        numeric_columns=(),
        missing_value_policy="keep",
        burden="medium",
        recommended_metrics=(
            "roc_auc_ovr",
            "f1_macro",
            "balanced_accuracy",
        ),
        license_note="UCI ML Repository (CC BY 4.0).",
        citation_key="uci_dry_bean",
        include_in_v1=True,
        notes=(
            "7-way multiclass; included if multiclass support is validated. "
            "Otherwise defer to a v1.1 follow-up."
        ),
    ),
    "mushroom": DatasetMetadata(
        dataset_id="mushroom",
        display_name="Mushroom",
        source_type="uci",
        source_url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
        ),
        openml_id=24,
        task_type="binary",
        target_column="class",
        target_transform="binary_p",
        categorical_columns=(
            "cap-shape",
            "cap-surface",
            "cap-color",
            "bruises",
            "odor",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color",
            "stalk-shape",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring",
            "stalk-color-below-ring",
            "veil-type",
            "veil-color",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
        ),
        numeric_columns=(),
        missing_value_policy="keep",
        burden="light",
        license_note="UCI ML Repository (CC BY 4.0).",
        citation_key="uci_mushroom",
        include_in_v1=True,
        notes=(
            "Categorical-only; native categorical for CatBoost. "
            "One-hot for XGBoost / LightGBM."
        ),
    ),
    "phishing": DatasetMetadata(
        dataset_id="phishing",
        display_name="Phishing Websites (UCI 327)",
        source_type="uci",
        source_url=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
        ),
        openml_id=4534,
        task_type="binary",
        target_column="Result",
        target_transform="map:{-1:0,1:1}",
        numeric_columns=(),
        missing_value_policy="keep",
        burden="medium",
        license_note="UCI ML Repository (CC BY 4.0).",
        citation_key="uci_phishing",
        include_in_v1=True,
        notes=(
            "Engineered features; values in {-1, 0, 1}. ARFF source; loader "
            "writes a normalized CSV."
        ),
    ),
}


V1_INCLUDED: tuple[str, ...] = tuple(
    sorted(d.dataset_id for d in REGISTRY.values() if d.include_in_v1)
)


def list_dataset_ids() -> list[str]:
    return sorted(REGISTRY.keys())


def get_metadata(dataset_id: str) -> DatasetMetadata:
    if dataset_id not in REGISTRY:
        raise KeyError(
            f"unknown dataset_id {dataset_id!r}; choose from {list_dataset_ids()}"
        )
    return REGISTRY[dataset_id]


__all__ = ["REGISTRY", "V1_INCLUDED", "list_dataset_ids", "get_metadata"]
