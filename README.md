# Societal Challenge Classifier

A machine learning and NLP-based web application that analyzes citizen-reported societal problems and automatically classifies them into relevant categories.

The system takes a problem description as input and predicts:

- Domain
- Sub-domain
- Problem Type
- Priority
- Severity
- Affected Group
- Important Keywords
- Similar / Duplicate Problems
- Prediction Confidence

## Features

### 1. Hierarchical Problem Classification

The application uses a hierarchical classification approach:

Problem Description
        ↓
Domain
        ↓
Sub-domain
        ↓
Problem Type

This allows the system to classify problems at different levels of detail.

### 2. NLP-Based Classification

The classification models use:

- TF-IDF
- Unigrams and bigrams
- Logistic Regression
- Text preprocessing

### 3. Problem Analysis

In addition to classification, the system extracts useful information from the submitted problem:

- Keywords
- Affected group
- Priority
- Severity

### 4. Similar Problem Detection

The system compares the submitted problem with existing problems in the dataset and identifies similar or potentially duplicate problems.

### 5. Web Interface

The application provides a simple two-page interface:

**Page 1:** Submit a societal problem

**Page 2:** View the analysis and classification results

## Supported Domains

The current dataset contains 10 major domains:

1. Agriculture
2. Education
3. Healthcare
4. Water Management
5. Sanitation
6. Environment
7. Energy
8. Urban Development
9. Accessibility
10. Rural Livelihood

Each domain contains multiple sub-domains and problem types.

## Dataset

The project uses a structured dataset containing:

- Problem ID
- Problem Text
- Domain
- Sub-domain
- Problem Type

The current dataset contains more than 15,000 problem examples.

## Machine Learning Models

The project uses multiple Logistic Regression models.

### Domain Model

A single model predicts the major domain of the problem.

### Sub-domain Models

Separate models are trained for each domain.

For example:

Agriculture → Irrigation, Soil Health, Crop Management, etc.

### Problem Type Models

Separate models are trained for each sub-domain.

For example:

Irrigation → Irrigation Water Shortage, Damaged Irrigation Canal, etc.

This hierarchical approach helps narrow down the classification step by step.

## Project Structure

```text
Societal Challenges/
│
├── app.py
│
├── Procfile
│
├── requirements.txt
│
├── runtime.txt
│
├── README.md
│
├── data/
│   └── quality_problem_dataset.csv
│
├── models/
│   ├── domain_model.pkl
│   ├── subdomain_models.pkl
│   ├── problem_type_models.pkl
│   ├── subdomain_*.pkl
│   └── problem_type_*.pkl
│
├── src/
│   ├── __init__.py
│   ├── predict.py
│   ├── similarity.py
│   ├── text_features.py
│   └── train_models.py
│
└── templates/
    ├── index.html
    └── results.html