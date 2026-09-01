import os
import joblib

from src.text_features import (
    extract_keywords,
    extract_affected_group,
    calculate_priority,
    calculate_severity
)

from src.similarity import (
    find_similar_problems
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "models"


# ============================================================
# LOAD DOMAIN MODEL
# ============================================================

domain_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "domain_model.pkl"
    )
)


# ============================================================
# LOAD SUB-DOMAIN MODEL MAPPING
# ============================================================

subdomain_models = joblib.load(
    os.path.join(
        MODEL_DIR,
        "subdomain_models.pkl"
    )
)


# ============================================================
# LOAD PROBLEM-TYPE MODEL MAPPING
# ============================================================

problem_type_models = joblib.load(
    os.path.join(
        MODEL_DIR,
        "problem_type_models.pkl"
    )
)


# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict_problem(problem_text):

    if not problem_text:

        raise ValueError(
            "Problem text cannot be empty."
        )


    problem_text = problem_text.strip()


    if not problem_text:

        raise ValueError(
            "Problem text cannot be empty."
        )


    # ========================================================
    # DOMAIN
    # ========================================================

    domain = domain_model.predict(
        [problem_text]
    )[0]


    domain_probability = max(
        domain_model.predict_proba(
            [problem_text]
        )[0]
    )


    # ========================================================
    # SUB-DOMAIN
    # ========================================================

    subdomain_filename = subdomain_models.get(
        domain
    )


    if not subdomain_filename:

        raise ValueError(
            f"Sub-domain model not found for {domain}"
        )


    subdomain_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            subdomain_filename
        )
    )


    sub_domain = subdomain_model.predict(
        [problem_text]
    )[0]


    subdomain_probability = max(
        subdomain_model.predict_proba(
            [problem_text]
        )[0]
    )


    # ========================================================
    # PROBLEM TYPE
    # ========================================================

    problem_type_filename = problem_type_models.get(
        sub_domain
    )


    if not problem_type_filename:

        raise ValueError(
            f"Problem-type model not found for {sub_domain}"
        )


    problem_type_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            problem_type_filename
        )
    )


    problem_type = problem_type_model.predict(
        [problem_text]
    )[0]


    problem_type_probability = max(
        problem_type_model.predict_proba(
            [problem_text]
        )[0]
    )


    # ========================================================
    # KEYWORDS
    # ========================================================

    keywords = extract_keywords(
        problem_text
    )


    # ========================================================
    # AFFECTED GROUP
    # ========================================================

    affected_group = extract_affected_group(
        problem_text
    )


    # ========================================================
    # PRIORITY
    # ========================================================

    priority = calculate_priority(
        problem_text
    )


    # ========================================================
    # SEVERITY
    # ========================================================

    severity = calculate_severity(
        problem_text
    )


    # ========================================================
    # DUPLICATE / SIMILARITY
    # ========================================================

    similarity_result = find_similar_problems(
        problem_text
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {

        "domain": domain,

        "sub_domain": sub_domain,

        "problem_type": problem_type,

        "priority": priority,

        "severity": severity,

        "affected_group": affected_group,

        "keywords": keywords,

        "duplicate_detection":
            similarity_result,

        "confidence": {

            "domain": round(
                float(domain_probability),
                3
            ),

            "sub_domain": round(
                float(subdomain_probability),
                3
            ),

            "problem_type": round(
                float(problem_type_probability),
                3
            )
        }
    }


    return result
