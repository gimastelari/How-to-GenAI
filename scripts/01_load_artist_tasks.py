import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
FILE_PATH = "data/raw/onet_task_statements.xlsx"
SOC_CODE = "27-1013.00"  # Fine Artists
OUTPUT_PATH = "data/clean/artist_tasks.csv"

# -----------------------------
# Load O*NET Task Statements
# -----------------------------
print("Loading O*NET task statements...")

tasks = pd.read_excel(FILE_PATH)

print("Dataset loaded.")
print("Shape:", tasks.shape)
print("Columns:")
for col in tasks.columns:
    print(" -", col)

# -----------------------------
# Filter to Fine Artists
# -----------------------------
if "O*NET-SOC Code" not in tasks.columns:
    raise KeyError("Column 'O*NET-SOC Code' not found.")

if "Task" not in tasks.columns:
    raise KeyError("Column 'Task' not found.")

artist_tasks = tasks[tasks["O*NET-SOC Code"] == SOC_CODE]

print("\nFiltered to Fine Artists")
print("Number of tasks:", artist_tasks.shape[0])

print("\nFine Artist Tasks:")
for i, task in enumerate(artist_tasks["Task"], start=1):
    print(f"{i}. {task}")

# -----------------------------
# Save Clean Dataset
# -----------------------------
artist_tasks.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
