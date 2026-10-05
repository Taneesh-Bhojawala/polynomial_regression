import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

# FINAL MODEL FOR VAR2
# Selected through cross-validation:
# Ridge + Polynomial Degree 9

train = pd.read_csv("data/BT2024053_train_var2.csv")
features = ["x1", "x2", "x3"]

X_train = train[features]
y_train = train["y"]

ridge_alphas = np.logspace(-6, 6, 25)

model = make_pipeline(PolynomialFeatures(degree=9, include_bias=False), StandardScaler(), RidgeCV(alphas=ridge_alphas, cv=3, scoring="neg_mean_squared_error"))


# TRAIN ON ALL 1000 TRAINING SAMPLES

model.fit(X_train, y_train)

poly = model.named_steps["polynomialfeatures"]
scaler = model.named_steps["standardscaler"]
ridge = model.named_steps["ridgecv"]

final_alpha = ridge.alpha_

print("\n" + "-" * 60)
print("FINAL MODEL - VAR2")

print("Model: Ridge")
print("Polynomial Degree: 9")
print(f"Final Alpha: {final_alpha:.10f}")
print(f"Training Samples: {len(X_train)}")

#Get the Polynomial Terms and Coefficients

terms = poly.get_feature_names_out(features)

#Ridge coefficients are based on standardized polynomial features.
#Convert them back to coefficients for the original polynomial terms.
coefficients = ridge.coef_ / scaler.scale_

intercept = (ridge.intercept_- sum((ridge.coef_ * scaler.mean_) / scaler.scale_))

coefficient_table = pd.DataFrame({"term": terms,"coefficient": coefficients})

#Save the polynomial coefficients to file
coefficient_table.to_csv("predictions/var2_coefficients.csv", index=False)

#Load the Test Data
test = pd.read_csv("data/BT2024053_test_var2.csv")

X_test = test[features]

#Get predictions from model
predictions = model.predict(X_test)

#Save predictions
prediction_table = pd.DataFrame({"y": predictions})

prediction_table.to_csv("predictions/BT2024053_pred_var2.csv", index=False)

#Saving the final model information based on the entire dataset
final_model_info = pd.DataFrame({
    "model": ["Ridge"],
    "degree": [9],
    "alpha": [final_alpha],
    "intercept": [intercept],
    "training_samples": [len(X_train)],
    "test_predictions": [len(predictions)]
})

final_model_info.to_csv("predictions/var2_final_model.csv", index=False)

print("\nFiles saved:")
print("predictions/BT2024053_pred_var2.csv")
print("predictions/var2_coefficients.csv")
print("predictions/var2_final_model.csv")

print(f"\nNumber of predictions: {len(predictions)}")