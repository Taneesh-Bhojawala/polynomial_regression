# Polynomial Regression and Regularization

This assignment implements polynomial regression for two regression problems and compares:

- Linear Regression
- Polynomial Regression
- Polynomial + Ridge
- Polynomial + Lasso
- Polynomial + Elastic Net

The goal is to study model complexity, overfitting, and the effect of regularization, then select a final model for each problem using cross-validation.

## 1. Project Structure

```text
polynomial_regression/
├── data/
│   ├── BT2024053_train_var1.csv
│   ├── BT2024053_test_var1.csv
│   ├── BT2024053_train_var2.csv
│   └── BT2024053_test_var2.csv
├── src/
│   ├── experiment_var1.py
│   ├── analyze_var1.py
│   ├── final_var1.py
│   ├── experiment_var2.py
│   ├── analyze_var2.py
│   └── final_var2.py
├── results/
│   ├── var1/
│   └── var2/
├── predictions/
├── requirements.txt
├── .gitignore
└── README.md
```

## 2. Dataset

Each problem contains 1000 training samples and 1000 test samples.

### Var1

- Features: `x1, x2, x3, x4, x5, x6`
- Target: `y`
- Polynomial degrees tested: 1 to 10

### Var2

- Features: `x1, x2, x3`
- Target: `y`
- Polynomial degrees tested: 1 to 20

The test files contain only input features and are reserved for the final prediction stage.

## 3. Requirements and Setup

Python 3.9+ is recommended. Required libraries are listed in `requirements.txt`.

From the project root:

```powershell
python -m pip install -r requirements.txt
```

A virtual environment can be used if required:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 4. Important: Run from the Project Root

The programs use relative paths such as `data/...`, `results/...`, and `predictions/...`.

Therefore, run commands from the main `polynomial_regression` directory:

```powershell
cd path\to\polynomial_regression
```

Do not run the programs from inside `src/`.

## 5. Workflow

The complete workflow is:

```text
Training data
     |
     v
5-fold outer cross-validation
     |
     v
Compare polynomial degrees and model types
     |
     v
Analyze CV results
     |
     v
Select final model
     |
     v
Train selected model on all 1000 training samples
     |
     v
Predict the 1000 test samples
```

The test set is not used for model or hyperparameter selection.

## 6. Step 1 - Run Experiments

### Var1

```powershell
python src/experiment_var1.py
```

Tests degrees 1-10 using Linear Regression, Polynomial Regression, Ridge, Lasso and Elastic Net.

### Var2

```powershell
python src/experiment_var2.py
```

Tests degrees 1-20 using the same model families.

### Multi-core / Parallel Execution

The outer 5-fold cross-validation uses:

```python
n_jobs=-2
```

This runs the independent outer folds in parallel using all available CPU cores except one, reducing execution time. The inner CV performed by the regularized models is a separate process and is not necessarily using all CPU cores.

## 7. Cross-Validation Methodology

The experiments use:

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

For each outer fold:

- 80% of the training data is used for fitting and hyperparameter selection.
- 20% is kept as the outer validation fold.
- Ridge, Lasso and Elastic Net use an additional 3-fold inner CV to select their regularization parameters.

Conceptually:

```text
Outer 5-fold CV
     |
     +-- Outer training data
     |       |
     |       +-- Inner 3-fold CV for hyperparameter selection
     |       |
     |       +-- Fit using selected hyperparameters
     |
     +-- Outer validation data
             |
             +-- Evaluate MSE and R²
```

This keeps the outer validation fold separate from hyperparameter selection.

The final programs use the same 3-fold CV procedure to select the final regularization parameter on all 1000 training samples.

## 8. Polynomial Features

The code uses:

```python
PolynomialFeatures(degree=degree, include_bias=False)
```

All powers and interaction terms up to the selected degree are generated automatically. Interaction terms are not manually chosen.

`include_bias=False` excludes the constant column because the regression estimator handles the intercept.

The number of terms grows rapidly with degree:

- Var1, degree 10: 8007 non-constant terms
- Var2, degree 20: 1770 non-constant terms

This rapid growth is one reason regularization is useful for higher-degree models.

## 9. Standardization

Ridge, Lasso and Elastic Net use the pipeline:

```text
PolynomialFeatures
      ->
StandardScaler
      ->
Regularized Regression
```

Scaling is performed after polynomial features are generated. Because the scaler is inside the pipeline, it is fitted separately for each CV split, preventing data leakage.

## 10. Models and Hyperparameters

### Linear Regression

Used as the baseline.

### Polynomial Regression

Uses polynomial features followed by ordinary Linear Regression without regularization.

### Ridge

Ridge uses L2 regularization. The experiment explicitly tests 25 alpha values:

```python
ridge_alphas = np.logspace(-6, 6, 25)
```

So the search range is `10^-6` to `10^6` with 25 logarithmically spaced values.

### Lasso

Lasso uses L1 regularization:

```python
LassoCV(alphas=30, cv=3, max_iter=5000, tol=1e-3)
```

Here, `alphas=30` means that scikit-learn automatically generates 30 alpha values along its regularization path. It does **not** mean `alpha = 30`.

Lasso can set some coefficients to zero, which can also provide feature selection.

### Elastic Net

Elastic Net combines L1 and L2 regularization.

The experiment tests:

```python
l1_ratio = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
```

For each `l1_ratio`, 30 alpha values are generated automatically, giving:

```text
30 alpha values × 6 l1_ratio values = 180 alpha/l1_ratio combinations
```

The best combination is selected by inner cross-validation.

### `alphas=30` vs `n_alphas`

The current code uses:

```python
LassoCV(alphas=30, ...)
ElasticNetCV(alphas=30, ...)
```

The integer tells scikit-learn how many alpha values to generate automatically. Older versions of scikit-learn used a separate `n_alphas` parameter for this purpose. `n_alphas` is not used in this project.

## 11. Number of Configurations Tested

The values 41 and 81 count only the basic degree/model combinations:

### Var1

```text
1 baseline + (10 degrees × 4 model types) = 41
```

### Var2

```text
1 baseline + (20 degrees × 4 model types) = 81
```

These are **not** the total number of candidate model/hyperparameter combinations.

For each degree, the regularized models additionally search:

```text
Ridge       = 25 alpha values
Lasso       = 30 alpha values
Elastic Net = 30 × 6 = 180 alpha/l1_ratio combinations
```

Therefore, conceptually, each degree has:

```text
1 + 25 + 30 + 180 = 236 candidate combinations
```

This gives approximately:

```text
Var1: 10 × 236 + 1 = 2361 candidate combinations
Var2: 20 × 236 + 1 = 4721 candidate combinations
```

These numbers describe the size of the search space, not the number of completely independent full model fits. `LassoCV` and `ElasticNetCV` use efficient regularization-path computations, and the candidates are evaluated within the nested cross-validation procedure.

## 12. Evaluation Metrics

### Mean Squared Error (MSE)

Measures the average squared prediction error. Lower is better.

Scikit-learn returns the metric as `neg_mean_squared_error`, so the code multiplies it by `-1` before reporting MSE.

### R² Score

Measures the proportion of target variance explained by the model. Higher is better.

The results also store the standard deviation across the five outer folds to help assess stability.

## 13. Step 2 - Analyze Results

Run the analysis after the corresponding experiment has completed.

```powershell
python src/analyze_var1.py
python src/analyze_var2.py
```

The analysis programs:

- compare training and validation MSE across polynomial degrees
- generate model-comparison plots
- find the best configuration for each model type
- find configurations within 1% of the best validation MSE
- save the final recommended configuration

The combined validation-MSE plot uses a logarithmic y-axis because high-degree unregularized polynomial models can produce very large errors.

## 14. Model Selection Rule

The primary criterion is mean validation MSE from the outer 5-fold CV.

The selection procedure is:

1. Find the lowest mean validation MSE.
2. Find all configurations within 1% of that value.
3. If multiple configurations remain, prefer the lower polynomial degree.
4. If still tied, prefer the smaller absolute train-validation MSE gap.

The gap is:

```text
|mean validation MSE - mean training MSE|
```

The gap is used only as a diagnostic/tie-breaker, not as the primary selection criterion.

## 15. Results Files

Each variable has a separate results directory.

Typical files include:

```text
results/var1/
├── cv_results.csv
├── cv_fold_results.csv
├── best_per_model.csv
├── close_models.csv
├── final_model_selection.csv
└── *.png
```

The Var2 directory contains the corresponding files.

`cv_results.csv` stores summary metrics for each basic model-degree configuration.

`cv_fold_results.csv` stores fold-level metrics and the selected `alpha` / `l1_ratio` for regularized models.

The CSV files are overwritten when the experiment or analysis programs are run again.

## 16. Step 3 - Final Models and Predictions

The selected final models are:

```text
Var1: Lasso + Polynomial Degree 5
Var2: Ridge + Polynomial Degree 9
```

Run:

```powershell
python src/final_var1.py
python src/final_var2.py
```

The final programs:

1. Load all 1000 training samples.
2. Generate the selected polynomial features.
3. Standardize the polynomial features.
4. Use 3-fold CV to select the final regularization parameter.
5. Fit the selected model on all training data.
6. Predict the 1000 test samples.
7. Save the required prediction CSVs.

The test data is used only after the final model has been fixed.

The final regularization parameter is selected again using the complete training dataset. Alpha values from the outer folds are not averaged and used as the final alpha.

## 17. Final Output Files

The required submission files are:

```text
predictions/BT2024053_pred_var1.csv
predictions/BT2024053_pred_var2.csv
```

Each contains one column:

```text
y
```

Additional files are generated for reference:

```text
predictions/var1_coefficients.csv
predictions/var1_final_model.csv
predictions/var2_coefficients.csv
predictions/var2_final_model.csv
```

The coefficient files contain polynomial terms and their coefficients after converting back from standardized feature space. The final-model files store information such as model type, degree, final alpha, intercept, training sample count, and prediction count.

## 18. Complete Execution Order

From the project root:

```powershell
python -m pip install -r requirements.txt

python src/experiment_var1.py
python src/analyze_var1.py
python src/final_var1.py

python src/experiment_var2.py
python src/analyze_var2.py
python src/final_var2.py
```

Var1 and Var2 are independent, so either can be processed first.

## 19. Important Notes

- Do not use the test files for model selection, degree selection, or hyperparameter tuning.
- Do not average alpha values selected in different outer folds.
- The final programs intentionally hardcode the selected model and degree for simplicity and reproducibility.
- Do not manually scale the CSV data; scaling is handled inside the model pipeline.
- Always run the programs from the project root.

