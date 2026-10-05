import pandas as pd
import matplotlib.pyplot as plt

#Load the results
results = pd.read_csv("results/var2/cv_results.csv")
print("Total model configurations:", len(results))

results["absolute_mse_gap"] = (results["mean_validation_mse"] - results["mean_train_mse"]).abs()

#Plot Training and Validation MSE vs Degree
for model in results["model"].unique():

    model_results = results[results["model"] == model]

    plt.figure(figsize=(10, 6))

    plt.plot(model_results["degree"], model_results["mean_train_mse"], marker="o", label="Training MSE")
    plt.plot(model_results["degree"], model_results["mean_validation_mse"], marker="o", label="Validation MSE")

    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean MSE")
    plt.title(f"Training and Validation MSE vs Degree- {model}")
    plt.legend()
    plt.grid(True)

    filename = (model.lower().replace(" ", "_")+ "_train_and_validation_mse_vs_degree.png")

    plt.savefig(f"results/var2/{filename}", dpi=300, bbox_inches="tight")
    plt.show()

#Plot Validation MSE for all models (fixed scale)
plt.figure(figsize=(12, 7))

line_styles = {
    "Linear Regression": {"marker": "x", "linestyle": ":"},
    "Polynomial Regression": {"marker": "o", "linestyle": "-"},
    "Ridge": {"marker": "s", "linestyle": "--"},
    "Lasso": {"marker": "^", "linestyle": "-."},
    "Elastic Net": {"marker": "d", "linestyle": ":"}
}

for model in results["model"].unique():

    model_results = results[results["model"] == model]
    style = line_styles.get(model, {"marker": "o", "linestyle": "-"})

    plt.plot(
        model_results["degree"],
        model_results["mean_validation_mse"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=2,
        alpha=0.8,
        label=model
    )

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Validation MSE (Log Scale)")
plt.title("Validation MSE vs Polynomial Degree (All Models)")

#Set Y-axis to logarithmic so the unpenalized model doesn't crush the scale
plt.yscale("log")
plt.legend()
# Add finer grid lines for the log scale
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.savefig("results/var2/validation_mse_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

#Best Configuration for each model type
print("\n" + "-" * 70)
print("BEST CONFIGURATION FOR EACH MODEL TYPE")

best_per_model = []

for model in results["model"].unique():

    model_results = results[results["model"] == model]

    best = model_results.loc[model_results["mean_validation_mse"].idxmin()]
    best_per_model.append(best)

    print(f"\n{model}")
    print(f"  Degree:              {int(best['degree'])}")
    print(f"  Training MSE:        {best['mean_train_mse']:.6f}")
    print(f"  Validation MSE:      {best['mean_validation_mse']:.6f}")
    print(f"  Validation R²:       {best['mean_validation_r2']:.6f}")
    print(f"  Train-Validation Gap: {best['absolute_mse_gap']:.6f}")

best_per_model = pd.DataFrame(best_per_model)

#Finding absolute best model among the ones trained
absolute_best = results.loc[results["mean_validation_mse"].idxmin()]
best_mse = absolute_best["mean_validation_mse"]

#Models within 1% of the best validation MSE
close_threshold = 0.01
close_models = results[results["mean_validation_mse"]<= best_mse * (1 + close_threshold)].copy()

#Select final model according to rule:
#Rules that im setting are that if they are within the above defined threshold, then follow:
# 1. Prefer low mean validation mse
# 2. Prefer lower degree
# 3. If still tied, prefer smaller MSE gap

if len(close_models) == 1:
    final_model = absolute_best
else:
    close_models = close_models.sort_values(by=["degree","absolute_mse_gap"])
    final_model = close_models.iloc[0]

#Display the results
print("\n" + "-" * 70)
print("OVERALL BEST BY VALIDATION MSE")

print(f"Model:          {absolute_best['model']}")
print(f"Degree:         {int(absolute_best['degree'])}")
print(f"Validation MSE: {absolute_best['mean_validation_mse']:.6f}")
print(f"Validation R²:  {absolute_best['mean_validation_r2']:.6f}")


print("\n" + "-" * 70)
print("MODELS WITHIN 1% OF BEST VALIDATION MSE")

print(
    close_models[
        [
            "model",
            "degree",
            "mean_validation_mse",
            "mean_train_mse",
            "absolute_mse_gap",
            "mean_validation_r2"
        ]
    ].to_string(index=False)
)


print("\n" + "-" * 70)
print("FINAL RECOMMENDED MODEL")

print(f"Model:              {final_model['model']}")
print(f"Degree:             {int(final_model['degree'])}")
print(f"Training MSE:       {final_model['mean_train_mse']:.6f}")
print(f"Validation MSE:     {final_model['mean_validation_mse']:.6f}")
print(f"Validation R²:      {final_model['mean_validation_r2']:.6f}")
print(f"Train-Validation Gap: {final_model['absolute_mse_gap']:.6f}")

#Save to csv
best_per_model.to_csv("results/var2/best_per_model.csv", index=False)

close_models.to_csv("results/var2/close_models.csv", index=False)

final_selection = pd.DataFrame([final_model])
final_selection.to_csv("results/var2/final_model_selection.csv",index=False)

print("\nAnalysis results saved:")
print("results/var2/best_per_model.csv")
print("results/var2/close_models.csv")
print("results/var2/final_model_selection.csv")