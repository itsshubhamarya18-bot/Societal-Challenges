import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DATASET
# ============================================================

DATA_PATH = "data/quality_problem_dataset.csv"


# ============================================================
# LOAD DATASET
# ============================================================

if os.path.exists(DATA_PATH):

    df = pd.read_csv(DATA_PATH)

    df["problem_text"] = (
        df["problem_text"]
        .fillna("")
        .astype(str)
    )

else:

    df = pd.DataFrame()


# ============================================================
# CREATE TF-IDF INDEX
# ============================================================

if not df.empty:

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    existing_vectors = vectorizer.fit_transform(
        df["problem_text"]
    )

else:

    vectorizer = None
    existing_vectors = None


# ============================================================
# FIND SIMILAR PROBLEMS
# ============================================================

def find_similar_problems(
    problem_text,
    top_k=3,
    duplicate_threshold=0.75
):

    if vectorizer is None:

        return {
            "is_duplicate": False,
            "similar_problems": []
        }


    # --------------------------------------------------------
    # Convert new problem to TF-IDF
    # --------------------------------------------------------

    new_vector = vectorizer.transform(
        [problem_text]
    )


    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        new_vector,
        existing_vectors
    )[0]


    # --------------------------------------------------------
    # Sort highest similarity first
    # --------------------------------------------------------

    top_indices = similarities.argsort()[::-1]


    similar_problems = []


    for index in top_indices:

        score = float(
            similarities[index]
        )


        # Ignore unrelated problems
        if score < 0.20:
            break


        existing_text = df.iloc[index][
            "problem_text"
        ]


        # Don't return exact same text
        if (
            existing_text.strip().lower()
            == problem_text.strip().lower()
        ):
            continue


        row = df.iloc[index]


        similar_problems.append({

            "problem_id": str(
                row.get(
                    "problem_id",
                    index
                )
            ),

            "problem_text": existing_text,

            "domain": row[
                "domain"
            ],

            "sub_domain": row[
                "sub_domain"
            ],

            "similarity": round(
                score,
                3
            )
        })


        if len(similar_problems) >= top_k:
            break


    # --------------------------------------------------------
    # Duplicate decision
    # --------------------------------------------------------

    is_duplicate = (

        len(similar_problems) > 0

        and similar_problems[0][
            "similarity"
        ] >= duplicate_threshold

    )


    return {

        "is_duplicate": is_duplicate,

        "similar_problems":
            similar_problems

    }