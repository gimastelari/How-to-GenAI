import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load modeled electrician data
# -----------------------------
df = pd.read_csv("data/clean/software_engineer_ai_exposure_model.csv")

# Create figures directory if needed
import os
os.makedirs("figures", exist_ok=True)

# -----------------------------
# 1. Task Count by Category
# -----------------------------
category_counts = df["category"].value_counts()

plt.figure()
category_counts.plot(kind="bar")
plt.title("Software Engineer Tasks by Category")
plt.ylabel("Number of Tasks")
plt.xlabel("Task Category")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/software_engineer_task_count_by_category.png")
plt.close()

# -----------------------------
# 2. Average AI Exposure by Category
# -----------------------------
avg_ai = df.groupby("category")["ai_exposure"].mean()

plt.figure()
avg_ai.plot(kind="bar")
plt.title("Average AI Exposure by Task Category")
plt.ylabel("AI Exposure Score")
plt.xlabel("Task Category")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/software_engineer_ai_exposure_by_category.png")
plt.close()

# -----------------------------
# 3. Task Contribution to AI Exposure
# -----------------------------
df_sorted = df.sort_values("contribution", ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(df_sorted["Task"], df_sorted["contribution"])
plt.title("Task-Level Contribution to AI Exposure (Software Engineers)")
plt.xlabel("Contribution to AI Exposure Index")
plt.tight_layout()
plt.savefig("figures/software_engineer_task_contributions.png")
plt.close()

# -----------------------------
# 4. Weight vs AI Exposure
# -----------------------------
plt.figure()
plt.scatter(df["weight"], df["ai_exposure"])
plt.title("Task Weight vs AI Exposure")
plt.xlabel("Task Weight")
plt.ylabel("AI Exposure Score")
plt.tight_layout()
plt.savefig("figures/software_engineer_weight_vs_ai_exposure.png")
plt.close()

print("All software engineer visualizations saved to figures/")