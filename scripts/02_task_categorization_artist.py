import pandas as pd

# -----------------------------
# Load clean artist tasks
# -----------------------------
tasks = pd.read_csv("data/clean/artist_tasks.csv")

print("Columns in dataset:", tasks.columns)

# -----------------------------
# CREATE CATEGORY COLUMN ✅
# -----------------------------
def categorize(task):
    task = str(task).lower()

    if any(word in task for word in [
        "paint", "draw", "sketch", "sculpt", "illustrate",
        "create", "design", "produce"
    ]):
        return "Creative Production"

    elif any(word in task for word in [
        "develop", "concept", "plan", "research", "visualize"
    ]):
        return "Creative Direction"

    elif any(word in task for word in [
        "edit", "revise", "adjust", "modify", "refine"
    ]):
        return "Editing & Refinement"

    elif any(word in task for word in [
        "communicate", "collaborate", "present", "exhibit"
    ]):
        return "Communication"

    else:
        return "Other"

# 🔥 THIS LINE IS THE MOST IMPORTANT ONE
tasks["category"] = tasks["Task"].apply(categorize)

print("\nCategory distribution:")
print(tasks["category"].value_counts())

# -----------------------------
# Assign raw weights
# -----------------------------
raw_weight_map = {
    "Creative Production": 5,
    "Creative Direction": 4,
    "Editing & Refinement": 3,
    "Communication": 2,
    "Other": 1
}

tasks["raw_weight"] = tasks["category"].map(raw_weight_map)

# Normalize weights
tasks["weight"] = tasks["raw_weight"] / tasks["raw_weight"].sum()

# -----------------------------
# Assign AI exposure
# -----------------------------
ai_exposure_map = {
    "Creative Production": 0.85,
    "Creative Direction": 0.60,
    "Editing & Refinement": 0.75,
    "Communication": 0.40,
    "Other": 0.50
}

tasks["ai_exposure"] = tasks["category"].map(ai_exposure_map)

# -----------------------------
# Compute contribution
# -----------------------------
tasks["contribution"] = tasks["weight"] * tasks["ai_exposure"]

AI_exposure_index = tasks["contribution"].sum()

# -----------------------------
# Output
# -----------------------------
print("\nArtist AI Exposure Index:", round(AI_exposure_index, 3))

# -----------------------------
# Save
# -----------------------------
tasks.to_csv("data/clean/artist_ai_exposure_model.csv", index=False)

print("\n✅ Saved model output to data/clean/artist_ai_exposure_model.csv")