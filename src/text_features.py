import re


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "this",
    "that",
    "there",
    "their",
    "they",
    "our",
    "we",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "a",
    "an",
    "with",
    "from",
    "by",
    "be",
    "been",
    "being",
    "cannot",
    "can",
    "not",
    "do",
    "does",
    "did",
    "due",
    "because",
    "during",
    "season",
    "struggle",
    "provide",
    "need",
    "needs",
    "problem",
    "issue",
    "local",
    "village"
}


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(text, max_keywords=8):

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text.lower()
    )

    keywords = []

    for word in words:

        if word not in STOPWORDS and word not in keywords:

            keywords.append(word)

        if len(keywords) >= max_keywords:
            break

    return keywords


# ============================================================
# AFFECTED GROUP
# ============================================================

AFFECTED_GROUPS = {

    "Farmers": [
        "farmer",
        "farmers",
        "agriculturist",
        "agriculturists"
    ],

    "Students": [
        "student",
        "students",
        "pupil",
        "pupils"
    ],

    "Teachers": [
        "teacher",
        "teachers"
    ],

    "Patients": [
        "patient",
        "patients"
    ],

    "Children": [
        "child",
        "children",
        "kid",
        "kids"
    ],

    "Elderly People": [
        "elderly",
        "old people",
        "senior citizens"
    ],

    "Women": [
        "woman",
        "women",
        "female",
        "females"
    ],

    "Persons with Disabilities": [
        "disabled",
        "disability",
        "disabilities",
        "persons with disabilities"
    ],

    "Rural Residents": [
        "villagers",
        "village residents",
        "rural people",
        "rural residents"
    ]
}


def extract_affected_group(text):

    text_lower = text.lower()

    for group, words in AFFECTED_GROUPS.items():

        for word in words:

            if word in text_lower:
                return group

    return None


# ============================================================
# PRIORITY KEYWORDS
# ============================================================

CRITICAL_WORDS = [

    "death",
    "dead",
    "dying",
    "life threatening",
    "emergency",
    "ambulance",
    "fire",
    "flood",
    "accident",
    "danger",
    "critical",
    "collapse"
]


HIGH_WORDS = [

    "severe",
    "serious",
    "shortage",
    "unavailable",
    "unsafe",
    "disease",
    "hospital",
    "no water",
    "no electricity",
    "lack of water",
    "lack of electricity"
]


# ============================================================
# PRIORITY
# ============================================================

def calculate_priority(text):

    text_lower = text.lower()

    # Critical
    for word in CRITICAL_WORDS:

        if word in text_lower:
            return "Critical"

    # High
    for word in HIGH_WORDS:

        if word in text_lower:
            return "High"

    # Default
    return "Medium"


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(text):

    text_lower = text.lower()

    critical_count = sum(
        word in text_lower
        for word in CRITICAL_WORDS
    )

    high_count = sum(
        word in text_lower
        for word in HIGH_WORDS
    )

    if critical_count >= 1:
        return "Critical"

    if high_count >= 1:
        return "High"

    return "Moderate"