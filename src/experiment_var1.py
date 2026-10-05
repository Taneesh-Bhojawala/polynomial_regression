import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

train = pd.read_csv("data/BT2024053_train_var1.csv")

print("Training Data format:", train.shape)
print("\nMissing values:\n", train.isnull().sum())
print("\nTraining Statistics:\n", train.describe())

features = ["x1", "x2", "x3", "x4", "x5", "x6"]

X = train[features]
y = train["y"]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {"mse": "neg_mean_squared_error", "r2": "r2"}

all_results = []
all_fold_results = []

model = LinearRegression()

results = cross_validate(model, X, y, cv=kfold, scoring=scoring, return_train_score=True)

#sklearn returns negative MSE because its scoring rule is
#designed so that higher scores are always considered better.
train_mse = -results["train_mse"]
validation_mse = -results["test_mse"]

train_r2 = results["train_r2"]
validation_r2 = results["test_r2"]

print("\nLINEAR REGRESSION BASELINE - 5-FOLD CV")

for i in range(5):
    print(f"\nFold {i + 1}")
    print(f"  Train MSE:       {train_mse[i]:.6f}")
    print(f"  Validation MSE:  {validation_mse[i]:.6f}")
    print(f"  Train R²:        {train_r2[i]:.6f}")
    print(f"  Validation R²:   {validation_r2[i]:.6f}")

    #Save individual fold result
    all_fold_results.append({
        "problem": "var1",
        "model": "Linear Regression",
        "degree": 1,
        "fold": i + 1,
        "selected_alpha": None,
        "selected_l1_ratio": None,
        "train_mse": train_mse[i],
        "validation_mse": validation_mse[i],
        "train_r2": train_r2[i],
        "validation_r2": validation_r2[i]
    })

#Save baseline summary to master results list
all_results.append({
    "problem": "var1",
    "model": "Linear Regression",
    "degree": 1,

    "mean_train_mse": train_mse.mean(),
    "std_train_mse": train_mse.std(),

    "mean_validation_mse": validation_mse.mean(),
    "std_validation_mse": validation_mse.std(),

    "mean_train_r2": train_r2.mean(),
    "std_train_r2": train_r2.std(),

    "mean_validation_r2": validation_r2.mean(),
    "std_validation_r2": validation_r2.std()
})

#Alpha values that RidgeCV will test
ridge_alphas = np.logspace(-6, 6, 25)

for degree in range(1, 11):

    print("\n" + "-" * 60)
    print(f"DEGREE {degree}\n")

    models = {
        #normal Polynomial Regression
        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression()),

        #Polynomial + Ridge
        "Ridge": make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), RidgeCV(alphas=ridge_alphas, cv=3, scoring="neg_mean_squared_error")),

        #Polynomial + Lasso
        "Lasso": make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), LassoCV(alphas=30, cv=3, max_iter=5000, tol=1e-3)),

        # Polynomial + Elastic Net
        "Elastic Net": make_pipeline(PolynomialFeatures(degree=degree,include_bias=False), StandardScaler(), ElasticNetCV(alphas=30, cv=3, l1_ratio=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99], max_iter=5000, tol=1e-3))
    }

    #Run each model for this particular degree
    for model_name, model in models.items():

        results = cross_validate(model, X, y, cv=kfold, scoring=scoring, return_train_score=True, return_estimator=True, n_jobs=-2)

        train_mse = -results["train_mse"]
        validation_mse = -results["test_mse"]

        train_r2 = results["train_r2"]
        validation_r2 = results["test_r2"]

        #Get selected regularization parameters from each fold
        selected_alphas = []
        selected_l1_ratios = []

        for estimator in results["estimator"]:
            
            if model_name == "Ridge":
                alpha = estimator.named_steps["ridgecv"].alpha_
                selected_alphas.append(alpha)
                selected_l1_ratios.append(None)

            elif model_name == "Lasso":
                alpha = estimator.named_steps["lassocv"].alpha_
                selected_alphas.append(alpha)
                selected_l1_ratios.append(None)

            elif model_name == "Elastic Net":
                alpha = estimator.named_steps["elasticnetcv"].alpha_
                l1_ratio = estimator.named_steps["elasticnetcv"].l1_ratio_

                selected_alphas.append(alpha)
                selected_l1_ratios.append(l1_ratio)

            else:
                selected_alphas.append(None)
                selected_l1_ratios.append(None)
                
        #Print summary
        print(f"\n{model_name}")
        print(f"  Mean Train MSE:       {train_mse.mean():.6f}")
        print(f"  Mean Validation MSE:  {validation_mse.mean():.6f}")
        print(f"  Mean Train R²:        {train_r2.mean():.6f}")
        print(f"  Mean Validation R²:   {validation_r2.mean():.6f}")

        #Save individual fold results
        for i in range(5):

            all_fold_results.append({
                "problem": "var1",
                "model": model_name,
                "degree": degree,
                "fold": i + 1,
                "selected_alpha": selected_alphas[i],
                "selected_l1_ratio": selected_l1_ratios[i],
                "train_mse": train_mse[i],
                "validation_mse": validation_mse[i],
                "train_r2": train_r2[i],
                "validation_r2": validation_r2[i]
            })

        #Save summary result
        all_results.append({
            "problem": "var1",
            "model": model_name,
            "degree": degree,

            "mean_train_mse": train_mse.mean(),
            "std_train_mse": train_mse.std(),

            "mean_validation_mse": validation_mse.mean(),
            "std_validation_mse": validation_mse.std(),

            "mean_train_r2": train_r2.mean(),
            "std_train_r2": train_r2.std(),

            "mean_validation_r2": validation_r2.mean(),
            "std_validation_r2": validation_r2.std()
        })

#Save the results to the file

# Master summary
cv_results = pd.DataFrame(all_results)
cv_results.to_csv("results/var1/cv_results.csv",index=False)

# Detailed fold results
cv_fold_results = pd.DataFrame(all_fold_results)
cv_fold_results.to_csv("results/var1/cv_fold_results.csv", index=False)

print("\nALL CROSS-VALIDATION RESULTS SAVED")

print("\nMaster results:")
print("results/var1/cv_results.csv")

print("\nFold-level results:")
print("results/var1/cv_fold_results.csv")

print(f"\nTotal model configurations tested: {len(cv_results)}")
print(f"Total fold results stored: {len(cv_fold_results)}")