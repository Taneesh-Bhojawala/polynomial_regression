import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression

#Checking how the data looks and also if there are any missing data and also checking the range the data is in.
train = pd.read_csv("data/BT2024053_train_var1.csv")

print("Training Data format: ", train.shape)
print("\nMissing values:\n", train.isnull().sum())
print("\nTraining Statistics:\n", train.describe())

#Separating the features and label
features = ["x1", "x2", "x3", "x4", "x5", "x6"]
X = train[features]
y = train["y"]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

model = LinearRegression()

scoring = {"mse": "neg_mean_squared_error", "r2": "r2"}

results = cross_validate(model, X, y, cv=kfold, scoring=scoring, return_train_score=True)

#sklearn uses negative mse to calculate it, this is done so that it properly gets the lower is better rule that is used for error, so we need to convert is back to positive
train_mse = -results["train_mse"]
validation_mse = -results["test_mse"]

train_r2 = results["train_r2"]
validation_r2 = results["test_r2"]

#Display results for each fold
print("LINEAR REGRESSION BASELINE - 5-FOLD CV")

for i in range(5):
    print(f"\nFold {i + 1}")
    print(f"  Train MSE:       {train_mse[i]:.6f}")
    print(f"  Validation MSE:  {validation_mse[i]:.6f}")
    print(f"  Train R²:        {train_r2[i]:.6f}")
    print(f"  Validation R²:   {validation_r2[i]:.6f}")

#Saving fold results separately
fold_results = pd.DataFrame({
    "fold": [1, 2, 3, 4, 5],
    "train_mse": train_mse,
    "validation_mse": validation_mse,
    "train_r2": train_r2,
    "validation_r2": validation_r2
})

fold_results.to_csv("results/var1/baseline_cv_folds.csv", index=False)

#saving the summary of each model we train to common csv so that later we can look at all the data and decide
summary = pd.DataFrame({
    "problem": ["var1"],
    "model": ["Linear Regression"],
    "degree": [1],
    "alpha": [None],
    "mean_train_mse": [train_mse.mean()],
    "std_train_mse": [train_mse.std()],
    "mean_validation_mse": [validation_mse.mean()],
    "std_validation_mse": [validation_mse.std()],
    "mean_train_r2": [train_r2.mean()],
    "std_train_r2": [train_r2.std()],
    "mean_validation_r2": [validation_r2.mean()],
    "std_validation_r2": [validation_r2.std()]
})

summary.to_csv("results/var1/cv_results.csv", index=False)

print("\nResults saved:")
print("results/var1/baseline_cv_folds.csv")
print("results/var1/cv_results.csv")