import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
FILE_PATH = "data/raw/onet_task_statements.xlsx"
SOC_CODE = "15-1252.00"  # Software Developers
OUTPUT_PATH = "data/clean/software_engineer_tasks.csv"

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
# Filter to Electricians
# -----------------------------
if "O*NET-SOC Code" not in tasks.columns:
    raise KeyError("Column 'O*NET-SOC Code' not found.")

if "Task" not in tasks.columns:
    raise KeyError("Column 'Task' not found.")

electrician_tasks = tasks[tasks["O*NET-SOC Code"] == SOC_CODE]

print("\nFiltered to Software Engineers")
print("Number of tasks:", electrician_tasks.shape[0])

print("\nSoftware Engineer Tasks:")
for i, task in enumerate(electrician_tasks["Task"], start=1):
    print(f"{i}. {task}")

# -----------------------------
# Save Clean Dataset
# -----------------------------
electrician_tasks.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
