import matplotlib.pyplot as plt
import os

# Example metrics (replace with real values if needed)
metrics = {
    "Baseline (TF-IDF + LR)": {"accuracy": 0.88, "f1": 0.87},
    "DistilBERT": {"accuracy": 0.93, "f1": 0.92},
}

models = list(metrics.keys())
accuracy = [metrics[m]["accuracy"] for m in models]
f1_scores = [metrics[m]["f1"] for m in models]

# Output directory
output_dir = os.path.join("outputs", "plots")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "model_comparison.png")

# Plot
x = range(len(models))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([i - width/2 for i in x], accuracy, width, label="Accuracy")
plt.bar([i + width/2 for i in x], f1_scores, width, label="F1-score")

plt.xticks(x, models, rotation=10)
plt.ylim(0.7, 1.0)
plt.ylabel("Score")
plt.title("Baseline vs DistilBERT Performance")
plt.legend()
plt.tight_layout()

plt.savefig(output_path)
print(f"Saved comparison plot to: {output_path}")
