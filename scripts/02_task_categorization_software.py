import pandas as pd

# -----------------------------
# Load clean electrician tasks
# -----------------------------
tasks = pd.read_csv("data/clean/software_engineer_tasks.csv")

# -----------------------------
# Assign task categories
# -----------------------------
category_map = {
    0: "System Design",
    1: "Debugging & Testing",
    2: "System Design",
    3: "Maintenance & Refactoring",
    4: "Collaboration & Communication",
    5: "System Design",
    6: "Coding & Implementation",
    7: "Coding & Implementation",
    8: "System Design",
    9: "System Design",  # ✅ CHANGE (was collaboration)
    10: "System Design",
    11: "Deployment & DevOps",
    12: "Deployment & DevOps",
    13: "System Design",  # ✅ CHANGE (was collaboration)
    14: "System Design",  # ✅ CHANGE (was collaboration)
    15: "System Design",
    16: "Collaboration & Communication"
}

tasks["category"] = tasks.index.map(category_map)

# -----------------------------
# Assign raw task importance
# (1 = rare, 5 = core daily task)
# -----------------------------
raw_weight_map = {
    "Coding & Implementation": 5,
    "Debugging & Testing": 5,
    "System Design": 4,
    "Maintenance & Refactoring": 4,
    "Deployment & DevOps": 3,
    "Collaboration & Communication": 3,
    "Research & Analysis": 2
}

tasks["raw_weight"] = tasks["category"].map(raw_weight_map)

# Normalize weights
tasks["weight"] = tasks["raw_weight"] / tasks["raw_weight"].sum()

# -----------------------------
# Assign AI exposure scores
# -----------------------------
ai_exposure_map = {
    "Coding & Implementation": 0.85,
    "Debugging & Testing": 0.75,
    "System Design": 0.60,
    "Maintenance & Refactoring": 0.80,
    "Deployment & DevOps": 0.50,
    "Collaboration & Communication": 0.30,
    "Research & Analysis": 0.40
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
print("\nSoftware Engineer AI Exposure Index:", round(AI_exposure_index, 3))
print("\nTask-level breakdown:")
print(tasks[["Task", "category", "weight", "ai_exposure", "contribution"]])

# -----------------------------
# Save final modeled dataset
# -----------------------------
tasks.to_csv(
    "data/clean/software_engineer_ai_exposure_model.csv",
    index=False,
    quoting=1,
    columns=["Task", "category", "weight", "ai_exposure", "contribution"]
)

print("\nSaved model output to data/clean/software_engineer_ai_exposure_model.csv")