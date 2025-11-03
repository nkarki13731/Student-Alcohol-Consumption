
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

features = [ "G2"]
target = "G3"


df = mergeddf_win.copy()
df["avg_exam"] = df[["G1","G2"]].mean(axis=1) 

X = df[features].values
y = df[target].values

#  Train / Test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True
)

# Fit model 
model = LinearRegression()
model.fit(X_train, y_train)

# 5-fold cross-validation on full data 
kfold = KFold(n_splits=5, shuffle=True, random_state= 0)
cv_r2   = cross_val_score(model, X_train, y_train, scoring="r2", cv=kfold)
cv_rmse = -cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kfold)
cv_mae  = -cross_val_score(model, X_train, y_train, scoring="neg_mean_absolute_error", cv=kfold)

#  Evaluate on test set 
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("Test metrics")
print(f"R^2 (accuracy): {r2:.4f}")
print(f"RMSE:           {rmse:.4f}")
print(f"MAE:            {mae:.4f}")

print("\n5-fold CV (mean ± std)")
print(f"R^2:  {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"RMSE: {cv_rmse.mean():.4f}")
print(f"MAE:  {cv_mae.mean():.4f}")


print("\nCoefficients:")
for name, coef in zip(features, model.coef_):
    print(f"{name:>9}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")


results = {
    "model": model,
    "features": features,
    "test_metrics": {"r2": r2, "rmse": rmse, "mae": mae},
    "cv_metrics": {"r2_mean": cv_r2.mean(), "r2_std": cv_r2.std(),
                   "rmse_mean": cv_rmse.mean(), "mae_mean": cv_mae.mean()},
    "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test
}


from sklearn.tree import DecisionTreeRegressor



features =  ["G2"]
target = "G3"


X = df[features].values
y = df[target].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=0
)

# --- 3) Fit model ---
regr = DecisionTreeRegressor(
    criterion="squared_error",  
    max_depth=3,             
    min_samples_split=2,
    min_samples_leaf=10,
    random_state=0             
)
regr.fit(X_train, y_train)

# --- 4) Evaluate on test set ---
y_pred = regr.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("Test metrics (regression)")
print(f"R^2:   {r2:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAE:   {mae:.4f}")

# --- 5) 5-fold cross-validation on full data ---
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
cv_r2   = cross_val_score(regr, X_train, y_train, scoring="r2", cv=kfold)
cv_rmse = -cross_val_score(regr, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kfold)
cv_mae  = -cross_val_score(regr, X_train, y_train, scoring="neg_mean_absolute_error", cv=kfold)

print("\n5-fold CV (mean ± std)")
print(f"R^2:  {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"RMSE: {cv_rmse.mean():.4f}")
print(f"MAE:  {cv_mae.mean():.4f}")


print("\nFeature importances:")
for name, imp in zip(features, regr.feature_importances_):
    print(f"{name:>9}: {imp:.4f}")


results_reg = {
    "model": regr,
    "features": features,
    "test_metrics": {"r2": r2, "rmse": rmse, "mae": mae},
    "cv_metrics": {"r2_mean": cv_r2.mean(), "r2_std": cv_r2.std(),
                   "rmse_mean": cv_rmse.mean(), "mae_mean": cv_mae.mean()},
    "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test
}


import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


df = mergeddf.copy()
cat_feats = ["famsup","goout_degree","activities","health_degree","freetime_degree","internet","famrel_degree"]
df["alc_level_bin"] = (df["alc_level"].str.lower() == "high").astype(int)
X = df[cat_feats]; y = df["alc_level_bin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats)]
)

pipe = Pipeline(steps=[
    ("pre", pre),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42))
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(
    pipe, X_train, y_train, cv=cv, n_jobs=-1,
    scoring={"accuracy":"accuracy","f1":"f1"}
)
print(f"CV Accuracy: {cv_scores['test_accuracy'].mean():.3f} ± {cv_scores['test_accuracy'].std():.3f}")
print(f"CV F1:       {cv_scores['test_f1'].mean():.3f} ± {cv_scores['test_f1'].std():.3f}")


pipe.fit(X_train, y_train)

proba  = pipe.predict_proba(X_test)[:, 1]   
pred05 = (proba >= 0.5).astype(int)

print("\nTEST Accuracy (0.5):", accuracy_score(y_test, pred05))
print("TEST F1 (0.5):      ", f1_score(y_test, pred05))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred05))
print("\nClassification report:\n", classification_report(y_test, pred05, digits=3))




import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


df = mergeddf.copy()
cat_feats = ["famsup","goout_degree","activities","health_degree","freetime_degree","internet","famrel_degree"]

0
df["alc_level_bin"] = (df["alc_level"].str.lower() == "high").astype(int)
X = df[cat_feats]
y = df["alc_level_bin"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ----- Pipeline: OneHot -> SMOTE (train only) -> Decision Tree -----
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)])
pipe = Pipeline([
    ("pre", pre),
    ("smote", SMOTE(random_state=42)),           
    ("clf", DecisionTreeClassifier(random_state=42))
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(
    pipe, X_train, y_train, cv=cv, n_jobs=-1,
    scoring={"accuracy":"accuracy","f1":"f1"}
)
print(f"CV Accuracy: {cv_scores['test_accuracy'].mean():.3f} ± {cv_scores['test_accuracy'].std():.3f}")
print(f"CV F1:       {cv_scores['test_f1'].mean():.3f} ± {cv_scores['test_f1'].std():.3f}")


pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("\nTEST Accuracy:", accuracy_score(y_test, y_pred))
print("TEST F1:      ", f1_score(y_test, y_pred))
print("\nClassification report:\n",
      classification_report(y_test, y_pred, target_names=["Low","High"]))


cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print("Confusion matrix (counts):\n", cm)



