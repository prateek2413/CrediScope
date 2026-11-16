# CrediScope – Loan Default Risk Prediction

A machine learning-based loan default risk prediction dashboard built using Python, Scikit-learn, Pandas, and Streamlit.

CrediScope analyzes historical LendingClub loan data and predicts whether a loan is more likely to be fully paid or default. It also provides model evaluation, feature importance analysis, and interactive loan risk predictions.

> **Disclaimer:** This is an academic machine learning research project. It is not a real-world loan approval system and should not be used for actual lending decisions.

## Project Overview

The objective of CrediScope is to explore how machine learning classification algorithms can help identify patterns associated with loan defaults.

The project includes:
- Historical loan dataset inspection and exploratory data analysis.
- Data cleaning and preprocessing.
- Chronological training, validation, and testing.
- Comparison of multiple machine learning models.
- Evaluation using classification metrics.
- Interactive Streamlit dashboard for loan risk prediction.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas, NumPy | Data processing |
| Scikit-learn | Machine learning and preprocessing |
| Matplotlib, Seaborn | Data visualization |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive dashboard |
| Jupyter Notebook | Data analysis and experimentation |

## Dataset

The project uses the **LendingClub Accepted Loans dataset (2007–2018 Q4)**.

- Original dataset: approximately 2.26 million loan records.
- Original features: 151 columns.
- Selected observation period: 2007–2013.
- Final modeling dataset: 227,957 records.
- Selected modeling features: 18.

### Dataset Source

Download the dataset from:

https://www.kaggle.com/datasets/wordsforthewise/lending-club

Download `accepted_2007_to_2018Q4.csv` and place it inside the local `data/` directory.

**Note:** The original dataset is not included in this GitHub repository because of its large file size.

## Data Preparation

The target variable was derived from `loan_status`:

- `0` – Fully Paid
- `1` – Charged Off or Default

Unresolved loan statuses and records that did not meet the selection criteria were excluded.

The final dataset contains:

| Category | Records |
|---|---:|
| Fully Paid | 192,619 |
| Defaults | 35,338 |
| Total | 227,957 |
| Default Rate | 15.50% |

### Chronological Data Split

To preserve the time-based nature of lending data, the dataset was divided chronologically.

| Dataset | Period | Records |
|---|---|---:|
| Training | 2007–2011 | 39,786 |
| Validation | 2012 | 53,367 |
| Testing | 2013 | 134,804 |

## Machine Learning Models

Three classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Evaluation

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision-Recall AUC

### Test Performance

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---:|---:|---:|
| Accuracy | 0.5482 | 0.5613 | **0.6153** |
| Precision | 0.2187 | 0.2004 | **0.2342** |
| Recall | **0.7374** | 0.6064 | 0.6458 |
| F1 Score | 0.3374 | 0.3012 | **0.3437** |
| ROC-AUC | 0.6749 | 0.6048 | **0.6783** |
| PR-AUC | 0.2606 | 0.2157 | **0.2665** |

Random Forest was selected as the primary model based on its overall test performance across the evaluated metrics. Logistic Regression achieved the highest recall.

## Dashboard Features

The Streamlit dashboard provides:

- Project overview and methodology.
- Model performance comparison.
- Confusion matrix analysis.
- Feature importance visualization.
- ROC curve and Precision-Recall curve analysis.
- Interactive loan default risk prediction.
- Floating Home button for convenient navigation.

The dashboard uses the saved Random Forest pipeline, so users do not need to retrain the model to make predictions.

## Project Structure

```text
CrediScope/
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── crediscope_primary_model.joblib
│   ├── decision_tree.joblib
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   └── 02_exploratory_data_analysis.ipynb
│
├── research/
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── model_performance_comparison.png
│   ├── precision_recall_curve.png
│   └── roc_curve.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/prateek2413/CrediScope.git
cd CrediScope
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
python -m streamlit run dashboard/app.py
```

The dashboard will open in your browser, normally at:

`http://localhost:8501`

**Note:** The dashboard uses pre-trained model files included in the repository. Downloading the original dataset and executing the notebooks are not required to run the dashboard.

## Research Visualizations

The `research/` directory contains:
- Model performance comparison.
- Confusion matrices.
- ROC curve.
- Precision-Recall curve.
- Random Forest feature importance.

## Limitations

- The dataset represents historical US LendingClub loans and may not generalize to other lending markets.
- The selected observation period is restricted to 2007–2013.
- Model predictions are based on historical patterns and are not guaranteed outcomes.
- Feature importance indicates model associations, not causal relationships.
- The model's default score is not a calibrated real-world probability of default.

## Future Improvements

- Probability calibration and threshold optimization.
- Explainable AI using SHAP.
- Improved model evaluation and robustness testing.
- Additional datasets and external validation.
- Enhanced dashboard functionality.

## Author

**Prateek Bishnoi**

B.Tech Computer Science and Engineering

GitHub: https://github.com/prateek2413

---

If you find this project useful, feel free to explore the repository and its implementation.
