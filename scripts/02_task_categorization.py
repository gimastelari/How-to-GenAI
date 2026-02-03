import pandas as pd

# -----------------------------
# Load clean electrician tasks
# -----------------------------
tasks = pd.read_csv("data/clean/electrician_tasks.csv")

# -----------------------------
# Assign task categories
# -----------------------------
category_map = {
    0: "Planning & Design",
    1: "Physical Execution",
    2: "Physical Execution",
    3: "Physical Execution",
    4: "Physical Execution",
    5: "Physical Execution",
    6: "Administration & Compliance",
    7: "Planning & Design",
    8: "Physical Execution",
    9: "Diagnostics & Troubleshooting",
    10: "Diagnostics & Troubleshooting",
    11: "Diagnostics & Troubleshooting",
    12: "Physical Execution",
    13: "Physical Execution",
    14: "Physical Execution",
    15: "Physical Execution",
    16: "Planning & Design",
    17: "Administration & Compliance",
    18: "Physical Execution",
    19: "Physical Execution",
    20: "Emergency / Irregular Tasks"
}

tasks["category"] = tasks.index.map(category_map)

# -----------------------------
# Assign raw task importance
# (1 = rare, 5 = core daily task)
# -----------------------------
raw_weight_map = {
    "Physical Execution": 5,
    "Diagnostics & Troubleshooting": 4,
    "Planning & Design": 3,
    "Administration & Compliance": 2,
    "Emergency / Irregular Tasks": 1
}

tasks["raw_weight"] = tasks["category"].map(raw_weight_map)

# Normalize weights
tasks["weight"] = tasks["raw_weight"] / tasks["raw_weight"].sum()

# -----------------------------
# Assign AI exposure scores
# -----------------------------
ai_exposure_map = {
    "Physical Execution": 0.05,
    "Diagnostics & Troubleshooting": 0.30,
    "Planning & Design": 0.50,
    "Administration & Compliance": 0.70,
    "Emergency / Irregular Tasks": 0.10
}

tasks["ai_exposure"] = tasks["category"].map(ai_exposure_map)

# -----------------------------
# Compute AI Exposure Index
# -----------------------------
tasks["contribution"] = tasks["weight"] * tasks["ai_exposure"]
AI_exposure_index = tasks["contribution"].sum()

# -----------------------------
# Output results
# -----------------------------
print("\nElectrician AI Exposure Index:", round(AI_exposure_index, 3))
print("\nTask-level breakdown:")
print(tasks[["Task", "category", "weight", "ai_exposure", "contribution"]])

# -----------------------------
# Save final modeled dataset
# -----------------------------
tasks.to_csv("data/clean/electrician_ai_exposure_model.csv", index=False)

print("\nSaved model output to data/clean/electrician_ai_exposure_model.csv")