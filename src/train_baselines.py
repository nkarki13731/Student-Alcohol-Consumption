
import argparse, os, random
import numpy as np


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, f1_score, confusion_matrix)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


import matplotlib.pyplot as plt
import joblib


from data import load_merged, winsorize_g2
from features import add_engineered_features

def set_seeds(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def ensure_dirs():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)

def plot_residuals(y_true, y_pred, out_path):
    resid = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, resid, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title("Residuals vs Predicted")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def run_regression(df_win, mlflow):
    # Features/target (match your baseline: G2 -> G3)
    X = df_win[["G2"]].values
    y = df_win["G3"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    # Linear Regression
    lin = LinearRegression().fit(X_tr, y_tr)
    y_pred = lin.predict(X_te)
    r2   = r2_score(y_te, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mae  = float(mean_absolute_error(y_te, y_pred))

    cv_r2   = cross_val_score(lin, X_tr, y_tr, scoring="r2", cv=kfold)
    cv_rmse = -cross_val_score(lin, X_tr, y_tr, scoring="neg_root_mean_squared_error", cv=kfold)
    cv_mae  = -cross_val_score(lin, X_tr, y_tr, scoring="neg_mean_absolute_error", cv=kfold)

    mlflow.log_metrics({
        "reg_lin_test_r2": float(r2),
        "reg_lin_test_rmse": rmse,
        "reg_lin_test_mae": mae,
        "reg_lin_cv_r2_mean": float(cv_r2.mean()),
        "reg_lin_cv_r2_std": float(cv_r2.std()),
        "reg_lin_cv_rmse_mean": float(cv_rmse.mean()),
        "reg_lin_cv_mae_mean": float(cv_mae.mean()),
    })
    ensure_dirs()
    plot_residuals(y_te, y_pred, "artifacts/reg_lin_residuals.png")
    mlflow.log_artifact("artifacts/reg_lin_residuals.png")
    joblib.dump(lin, "models/reg_linear.joblib")
    mlflow.log_artifact("models/reg_linear.joblib")

    tree = DecisionTreeRegressor(criterion="squared_error", max_depth=3,
                                 min_samples_split=2, min_samples_leaf=10, random_state=0)
    tree.fit(X_tr, y_tr)
    y_pred = tree.predict(X_te)
    r2   = r2_score(y_te, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mae  = float(mean_absolute_error(y_te, y_pred))

    cv_r2   = cross_val_score(tree, X_tr, y_tr, scoring="r2", cv=kfold)
    cv_rmse = -cross_val_score(tree, X_tr, y_tr, scoring="neg_root_mean_squared_error", cv=kfold)

    mlflow.log_metrics({
        "reg_tree_test_r2": float(r2),
        "reg_tree_test_rmse": rmse,
        "reg_tree_test_mae": mae,
        "reg_tree_cv_r2_mean": float(cv_r2.mean()),
        "reg_tree_cv_rmse_mean": float(cv_rmse.mean()),
    })
    plot_residuals(y_te, y_pred, "artifacts/reg_tree_residuals.png")
    mlflow.log_artifact("artifacts/reg_tree_residuals.png")
    joblib.dump(tree, "models/reg_tree.joblib")
    mlflow.log_artifact("models/reg_tree.joblib")

#
def run_classification(df_with_feats, mlflow):
   
    cat = ["famsup","goout_degree","activities","health_degree","freetime_degree","internet","famrel_degree"]
    # Label: High/Low -> 1/0
    y = (df_with_feats["alc_level"].str.lower() == "high").astype(int)
    X = df_with_feats[cat]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Preprocess: OneHot on categorical
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

    # Logistic Regression + SMOTE
    pipe_lr = Pipeline([
        ("pre", pre),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipe_lr, X_tr, y_tr, cv=cv, n_jobs=-1,
                            scoring={"accuracy": "accuracy", "f1": "f1"})
    mlflow.log_metrics({
        "clf_lr_cv_acc_mean": float(scores["test_accuracy"].mean()),
        "clf_lr_cv_acc_std":  float(scores["test_accuracy"].std()),
        "clf_lr_cv_f1_mean":  float(scores["test_f1"].mean()),
        "clf_lr_cv_f1_std":   float(scores["test_f1"].std()),
    })
    pipe_lr.fit(X_tr, y_tr)
    y_hat = (pipe_lr.predict_proba(X_te)[:,1] >= 0.5).astype(int)
    mlflow.log_metrics({
        "clf_lr_test_acc": float(accuracy_score(y_te, y_hat)),
        "clf_lr_test_f1":  float(f1_score(y_te, y_hat)),
    })
    ensure_dirs()
    plot_confusion(y_te, y_hat, "artifacts/clf_lr_confusion.png")
    mlflow.log_artifact("artifacts/clf_lr_confusion.png")
    joblib.dump(pipe_lr, "models/clf_lr.joblib")
    mlflow.log_artifact("models/clf_lr.joblib")

  
    pipe_dt = Pipeline([
        ("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)])),
        ("smote", SMOTE(random_state=42)),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])
    scores = cross_validate(pipe_dt, X_tr, y_tr, cv=cv, n_jobs=-1,
                            scoring={"accuracy": "accuracy", "f1": "f1"})
    mlflow.log_metrics({
        "clf_dt_cv_acc_mean": float(scores["test_accuracy"].mean()),
        "clf_dt_cv_acc_std":  float(scores["test_accuracy"].std()),
        "clf_dt_cv_f1_mean":  float(scores["test_f1"].mean()),
        "clf_dt_cv_f1_std":   float(scores["test_f1"].std()),
    })
    pipe_dt.fit(X_tr, y_tr)
    y_hat = pipe_dt.predict(X_te)
    mlflow.log_metrics({
        "clf_dt_test_acc": float(accuracy_score(y_te, y_hat)),
        "clf_dt_test_f1":  float(f1_score(y_te, y_hat)),
    })
    plot_confusion(y_te, y_hat, "artifacts/clf_dt_confusion.png")
    mlflow.log_artifact("artifacts/clf_dt_confusion.png")
    joblib.dump(pipe_dt, "models/clf_dt.joblib")
    mlflow.log_artifact("models/clf_dt.joblib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", default="student-mat.csv")
    parser.add_argument("--por", default="student-por.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", type=str, default="midpointreport")
    parser.add_argument("--run-name", type=str, default="baselines")
    args = parser.parse_args()

    set_seeds(args.seed)
    ensure_dirs()

    # Load + engineer
    merged = load_merged(args.mat, args.por)           # from data.py
    merged = add_engineered_features(merged)           # from features.py
    merged_win = winsorize_g2(merged)                  # from data.py

    
    import mlflow
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        # high-level params
        mlflow.log_params({
            "seed": args.seed,
            "mat_path": args.mat,
            "por_path": args.por,
            "n_rows_total": len(merged),
        })

        # tasks
        run_regression(merged_win, mlflow)
        run_classification(merged, mlflow)

if __name__ == "__main__":
    main()
