import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/quality_problem_dataset.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=[
        "problem_text",
        "domain",
        "sub_domain",
        "problem_type"
    ]
)


# ============================================================
# MODEL CREATOR
# ============================================================

def create_model():

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            )
        )
    ])


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_name(text):

    return re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        text.strip()
    ).strip("_").lower()


# ============================================================
# 1. DOMAIN MODEL
# ============================================================

print("\n======================================")
print("TRAINING DOMAIN MODEL")
print("======================================")

X = df["problem_text"]
y = df["domain"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


domain_model = create_model()

domain_model.fit(
    X_train,
    y_train
)


domain_pred = domain_model.predict(X_test)

domain_accuracy = accuracy_score(
    y_test,
    domain_pred
)


print(
    "Domain Accuracy:",
    round(domain_accuracy, 4)
)


joblib.dump(
    domain_model,
    os.path.join(
        MODEL_DIR,
        "domain_model.pkl"
    )
)


# ============================================================
# 2. SUB-DOMAIN MODELS
# ============================================================

print("\n======================================")
print("TRAINING SUB-DOMAIN MODELS")
print("======================================")


subdomain_models = {}

for domain in sorted(df["domain"].unique()):

    print(f"\nTraining sub-domain model: {domain}")

    domain_df = df[
        df["domain"] == domain
    ]

    X = domain_df["problem_text"]

    y = domain_df["sub_domain"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    model = create_model()

    model.fit(
        X_train,
        y_train
    )


    pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        pred
    )


    print(
        f"{domain} Sub-domain Accuracy:",
        round(accuracy, 4)
    )


    filename = (
        "subdomain_"
        + safe_name(domain)
        + ".pkl"
    )


    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            filename
        )
    )


    subdomain_models[domain] = filename


# ============================================================
# 3. PROBLEM TYPE MODELS
# ============================================================

print("\n======================================")
print("TRAINING PROBLEM TYPE MODELS")
print("======================================")


problem_type_models = {}


for sub_domain in sorted(
    df["sub_domain"].unique()
):

    print(
        f"\nTraining problem-type model: "
        f"{sub_domain}"
    )


    subdomain_df = df[
        df["sub_domain"] == sub_domain
    ]


    X = subdomain_df["problem_text"]

    y = subdomain_df["problem_type"]


    # Skip if there is only one problem type
    if y.nunique() < 2:

        print(
            "Skipping because only one "
            "problem type exists."
        )

        continue


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    model = create_model()

    model.fit(
        X_train,
        y_train
    )


    pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        pred
    )


    print(
        f"{sub_domain} Problem Type Accuracy:",
        round(accuracy, 4)
    )


    filename = (
        "problem_type_"
        + safe_name(sub_domain)
        + ".pkl"
    )


    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            filename
        )
    )


    problem_type_models[
        sub_domain
    ] = filename


# ============================================================
# SAVE MODEL MAPPINGS
# ============================================================

joblib.dump(
    subdomain_models,
    os.path.join(
        MODEL_DIR,
        "subdomain_models.pkl"
    )
)


joblib.dump(
    problem_type_models,
    os.path.join(
        MODEL_DIR,
        "problem_type_models.pkl"
    )
)


print("\n======================================")
print("ALL HIERARCHICAL MODELS SAVED")
print("======================================")

print(
    "\nSub-domain models:",
    len(subdomain_models)
)

print(
    "Problem-type models:",
    len(problem_type_models)
)