import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import mlflow
import os


# mlflow.set_tracking_uri("http://localhost:5001")


# load the dataset
dataset = pd.read_csv("train.csv")
numerical_cols = dataset.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = dataset.select_dtypes(include=["object"]).columns.tolist()
categorical_cols.remove("Loan_Status")
categorical_cols.remove("Loan_ID")

# dataset[col] = dataset[col].fillna(dataset[col].median())


## Filling categorical columns with mode
for col in categorical_cols:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

## Filling numerical columns with mean
for col in numerical_cols:
    dataset[col].fillna(dataset[col].median(), inplace=True)

# Take care of outliers
dataset[numerical_cols] = dataset[numerical_cols].apply(
    lambda x: x.clip(*x.quantile([0.05, 0.95]))
)

# Log Transformation & Domain Processing
dataset["LoanAmount"] = np.log(dataset["LoanAmount"]).copy()
dataset["TotalIncome"] = dataset["ApplicantIncome"] + dataset["CoapplicantIncome"]
dataset["TotalIncome"] = np.log(dataset["TotalIncome"]).copy()

# Dropping ApplicantIncome and CoapplicantIncome
dataset = dataset.drop(columns=["ApplicantIncome", "CoapplicantIncome"])

# Label encoding categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])


# Encode the targe columns
dataset["Loan_Status"] = le.fit_transform(dataset["Loan_Status"])

# Train test split
X = dataset.drop(columns=["Loan_ID", "Loan_Status"])
y = dataset.Loan_Status
RANDOM_SEED = 34

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED
)

# RandomForest
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    "n_estimators": [200, 400],
    "max_depth": [10, 20],
    "criterion": ["gini", "entropy"],
    "max_leaf_nodes": [50, 100],
}


grid_forest = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_forest,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
    verbose=0,
)

model_forest = grid_forest.fit(X_train, y_train)


# Logistic Regression
lr = LogisticRegression(random_state=RANDOM_SEED)
param_grid_log = {
    "C": [100, 10, 1.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
}

grid_log = GridSearchCV(
    estimator=lr,
    param_grid=param_grid_log,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
    verbose=0,
)

model_log = grid_log.fit(X_train, y_train)


# Decision Tree
dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
param_grid_dt = {
    "max_depth": [3, 5, 7],
    "criterion": ["gini", "entropy"],
}

grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
    verbose=0,
)

model_tree = grid_dt.fit(X_train, y_train)

mlflow.set_experiment("Loan_prediction")


# Model evaluation metrics
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1_score = metrics.f1_score(actual, pred)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="blue", label="ROC curve area = %0.2f" % auc)
    plt.plot(
        [0, 1],
        [
            0,
            1,
        ],
        "r--",
    )
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("False Positive Rate", size=14)
    plt.ylabel("True Positive Rate", size=14)
    plt.legend(loc="lower right")
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/roc_curve.png")
    # Close plot
    plt.close()
    return (accuracy, f1_score, auc)


def mlflow_logging(model, X, y, name):
    with mlflow.start_run() as run:
        # mlflow.set_tracking_uri("http://localhost:5001")
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        pred = model.predict(X)
        # metrics
        (accuracy, f1_score, auc) = eval_metrics(y, pred)
        # Logging best parameters from gridsearch
        mlflow.log_params(model.best_params_)
        # Logging metrics
        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score)
        mlflow.log_metric("auc", auc)

        # Logging artifacts and model
        mlflow.log_artifact("plots/roc_curve.png")
        mlflow.sklearn.log_model(model, name)
        mlflow.end_run()


mlflow_logging(model_tree, X_test, y_test, "DecisionTree")
mlflow_logging(model_forest, X_test, y_test, "RandomForest")
mlflow_logging(model_log, X_test, y_test, "LogisticRegression")
