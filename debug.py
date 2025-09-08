import os

file_path = os.path.join("Data", "synthetic_dataset.xlsx")
abs_path = os.path.abspath(file_path)

print("Looking for file at:", abs_path)
print("Exists:", os.path.exists(abs_path))
