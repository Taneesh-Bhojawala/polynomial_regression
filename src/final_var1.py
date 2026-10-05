import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

# FINAL MODEL FOR VAR1
# Selected through cross-validation:
# Lasso + Polynomial Degree 5

train = pd.read_csv("data/BT2024053_train_var1.csv")
features = ["x1", "x2", "x3", "x4", "x5", "x6"]

X_train = train[features]
y_train = train["y"]

model = make_pipeline(PolynomialFeatures(degree=5, include_bias=False), StandardScaler(), LassoCV(alphas=30, cv=3, max_iter=5000, tol=1e-3))


# TRAIN ON ALL 1000 TRAINING SAMPLES

model.fit(X_train, y_train)

poly = model.named_steps["polynomialfeatures"]
scaler = model.named_steps["standardscaler"]
lasso = model.named_steps["lassocv"]

final_alpha = lasso.alpha_

print("\n" + "-" * 60)
print("FINAL MODEL - VAR1")

print("Model: Lasso")
print("Polynomial Degree: 5")
print(f"Final Alpha: {final_alpha:.10f}")
print(f"Training Samples: {len(X_train)}")

#Get the Polynomial Terms and Coefficients

terms = poly.get_feature_names_out(features)

#Lasso coefficients are based on standardized polynomial features.
#Convert them back to coefficients for the original polynomial terms.
coefficients = lasso.coef_ / scaler.scale_

intercept = (lasso.intercept_- sum((lasso.coef_ * scaler.mean_) / scaler.scale_))

coefficient_table = pd.DataFrame({"term": terms,"coefficient": coefficients})

coefficient_table["selected"] = (coefficient_table["coefficient"].abs() > 1e-12)

#Save the polynomial coefficients to file
coefficient_table.to_csv("predictions/var1_coefficients.csv", index=False)

#Load the Test Data
test = pd.read_csv("data/BT2024053_test_var1.csv")

X_test = test[features]

#Get predictions from model
predictions = model.predict(X_test)

#Save predictions
prediction_table = pd.DataFrame({"y": predictions})

prediction_table.to_csv("predictions/BT2024053_pred_var1.csv", index=False)

#Saving the final model information based on the entire dataset
final_model_info = pd.DataFrame({
    "model": ["Lasso"],
    "degree": [5],
    "alpha": [final_alpha],
    "intercept": [intercept],
    "training_samples": [len(X_train)],
    "test_predictions": [len(predictions)]
})

final_model_info.to_csv("predictions/var1_final_model.csv", index=False)

print("\nFiles saved:")
print("predictions/BT2024053_pred_var1.csv")
print("predictions/var1_coefficients.csv")
print("predictions/var1_final_model.csv")

print(f"\nNumber of predictions: {len(predictions)}")
print(f"Non-zero polynomial terms: {coefficient_table['selected'].sum()}")