import pandas as pd
import numpy as np
import random
import os
import pickle

# === Load Data ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "Data", "Dataset.csv")
excel_path = os.path.join(base_dir, "Data", "synthetic_dataset.xlsx")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(" Loaded Dataset.csv")
elif os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    print(" Loaded synthetic_dataset.xlsx")
else:
    raise FileNotFoundError(" No dataset found in /Data folder")

# === Q-Learning ===
actions = ['Add_VM', 'Remove_VM', 'Do_Nothing']
q_table = {}
alpha, gamma, epsilon = 0.1, 0.9, 0.1
cpu_bins = np.linspace(0, 100, 6)
mem_bins = np.linspace(0, 100, 6)

def discretize(value, bins):
    return np.digitize([value], bins)[0]

def calculate_reward(cpu, mem, vms):
    reward = 0
    if cpu > 80 or mem > 75:
        reward -= 1  # Lower penalty
    elif 50 < cpu < 70 and 40 < mem < 65:
        reward += 2  # Increase positive reward
    else:
        reward -= 0.5

    if vms > 15:
        reward -= 0.5
    return reward


for i in range(len(df) - 1):
    row = df.iloc[i]
    next_row = df.iloc[i + 1]
    state = (
        discretize(row['CPU Utilization (%)'], cpu_bins),
        discretize(row['Memory Utilization (%)'], mem_bins)
    )
    if state not in q_table:
        q_table[state] = {a: 0 for a in actions}
    action = random.choice(actions) if random.uniform(0, 1) < epsilon else max(q_table[state], key=q_table[state].get)
    reward = calculate_reward(row['CPU Utilization (%)'], row['Memory Utilization (%)'], row['Number of Active VMs'])
    next_state = (
        discretize(next_row['CPU Utilization (%)'], cpu_bins),
        discretize(next_row['Memory Utilization (%)'], mem_bins)
    )
    if next_state not in q_table:
        q_table[next_state] = {a: 0 for a in actions}
    old_value = q_table[state][action]
    next_max = max(q_table[next_state].values())
    q_table[state][action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

# Save
q_table_path = os.path.join(base_dir, "RL_Agent", "q_table.pkl")
with open(q_table_path, "wb") as f:
    pickle.dump(q_table, f)

print(" Q-Learning completed and Q-table saved.")
new_q_table_path = os.path.join(base_dir, "RL_Agent", "new_q_table.pkl")
with open(new_q_table_path, "wb") as f:
    pickle.dump(q_table, f)

print(" New Q-table saved as new_q_table.pkl")

