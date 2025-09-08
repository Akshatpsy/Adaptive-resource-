# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from io import BytesIO
from Optimization import spider_monkey
from tensorflow.keras.models import load_model

# App layout
st.set_page_config(page_title="Adaptive Provisioning", layout="wide")

# Base paths
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "Data", "synthetic_dataset.xlsx")

# Sidebar controls
st.sidebar.header("Simulation Controls")
pop_size = st.sidebar.slider("Spider Monkey Population", 5, 100, 10)
iterations = st.sidebar.slider("Iterations", 10, 200, 30)
rl_method = st.sidebar.radio("Select RL Technique", ("Q-learning", "DQN"))

# Load dataset
df = pd.read_excel(dataset_path)

# --- Scenario Builder Sliders ---
st.sidebar.markdown("###  Stress Test Simulation")
cpu_spike = st.sidebar.slider("CPU Spike (%)", 0, 100, 0, help="Adds a fixed % spike to all CPU values")
mem_boost = st.sidebar.slider("Memory Boost Multiplier", 1.0, 2.0, 1.0, step=0.1, help="Multiplies memory utilization")

# Apply the stress simulation
df['CPU Utilization (%)'] = np.clip(df['CPU Utilization (%)'] + cpu_spike, 0, 100)
df['Memory Utilization (%)'] = np.clip(df['Memory Utilization (%)'] * mem_boost, 0, 100)


# RL agent paths
q_learning_path = os.path.join(base_dir, "RL_Agent", "q_table.pkl")
dqn_model_path = os.path.join(base_dir, "RL_Agent", "models", "dqn_model.h5")

# Load Q-table or model
q_table = {}
model = None

if rl_method == "Q-learning":
    if os.path.exists(q_learning_path):
        with open(q_learning_path, "rb") as f:
            q_table = pickle.load(f)
elif rl_method == "DQN":
    if os.path.exists(dqn_model_path):
        st.success(" DQN model loaded!")
        model = load_model(dqn_model_path)
        q_table = {}  # DQN uses neural network, not Q-table
    else:
        st.warning(" DQN model not found. Please train it first.")
        q_table = {}

# Header
st.title("Adaptive Resource Provisioning using RL + SMO")
st.markdown("A dashboard to visualize **VM optimization** using Spider Monkey Optimization and RL techniques.")

# Run optimization
with st.spinner("Running hybrid optimization..."):
    result_df, fitness_history = spider_monkey.run_optimization(df, q_table, pop_size, iterations)

# Metrics
st.markdown("### Optimization Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Min Penalty", f"{min(fitness_history):.2f}")
col2.metric("Max Penalty", f"{max(fitness_history):.2f}")
col3.metric("Final VM Recommendation", int(result_df['Optimized VM Count'].iloc[-1]))

# VM Trend Chart
st.subheader("Optimized VM Allocation Over Iterations")
fig_vm, ax_vm = plt.subplots(figsize=(8, 3))
ax_vm.plot(result_df["Optimized VM Count"], color="teal", linewidth=2)
ax_vm.set_xlabel("Iteration")
ax_vm.set_ylabel("VM Count")
st.pyplot(fig_vm)

# Heatmap View
st.subheader("Q-Table Heatmap Visualization")
view_option = st.radio("Select View Mode:", ["Single Q-Table", "Compare Two Q-Tables"], horizontal=True)

def generate_heatmap_matrix(q_table):
    state_action_map = {}
    for (cpu_bin, mem_bin), actions in q_table.items():
        state_index = (cpu_bin, mem_bin)
        if state_index not in state_action_map:
            state_action_map[state_index] = list(actions.values())
    states = list(state_action_map.keys())
    if not states:
        return np.zeros((1, 1)), [(0, 0)]
    action_count = len(state_action_map[states[0]])
    matrix = np.zeros((len(states), action_count))
    for i, state in enumerate(states):
        matrix[i] = state_action_map[state]
    return matrix, states

def plot_q_heatmap(matrix, title, ax):
    sns.heatmap(matrix, ax=ax, cmap="YlOrRd", cbar=True,
                xticklabels=action_labels if matrix.shape[1] > 1 else False,
                yticklabels=[f"State {i}" for i in range(len(matrix))])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Actions")
    ax.set_ylabel("States")

if rl_method == "DQN":
    st.info(" DQN does not use a Q-table. It uses a neural network for decision-making.")
else:
    action_labels = list(q_table[next(iter(q_table))].keys()) if q_table else []

    if view_option == "Single Q-Table":
        matrix, _ = generate_heatmap_matrix(q_table)
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_q_heatmap(matrix, f"{rl_method} Q-Table Heatmap", ax)
        st.pyplot(fig)
    else:
        new_q_table_path = os.path.join(base_dir, "RL_Agent", "new_q_table.pkl")
        if os.path.exists(new_q_table_path):
            with open(new_q_table_path, "rb") as f:
                q_table2 = pickle.load(f)
        else:
            st.warning("new_q_table.pkl not found. Showing original Q-table twice.")
            q_table2 = q_table
        matrix1, _ = generate_heatmap_matrix(q_table)
        matrix2, _ = generate_heatmap_matrix(q_table2)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_q_heatmap(matrix1, "Original Q-Table", axes[0])
        plot_q_heatmap(matrix2, "Updated Q-Table", axes[1])
        st.pyplot(fig)

# Cost Chart
st.subheader("Cost Over Iterations")
fig_cost, ax_cost = plt.subplots(figsize=(8, 3))
ax_cost.plot(result_df["Cost"], color="purple", linewidth=2)
ax_cost.set_xlabel("Iteration")
ax_cost.set_ylabel("Cost")
st.pyplot(fig_cost)

# Energy Chart
st.subheader("Energy Consumption Over Iterations")
fig_energy, ax_energy = plt.subplots(figsize=(8, 3))
ax_energy.plot(result_df["Energy"], color="green", linewidth=2)
ax_energy.set_xlabel("Iteration")
ax_energy.set_ylabel("Energy")
st.pyplot(fig_energy)

# SLA Violations
st.subheader("SLA Violations (CPU > 75%)")
sla_threshold = 75
fig_sla, ax_sla = plt.subplots(figsize=(8, 3))
ax_sla.plot(df["CPU Utilization (%)"], label="CPU Utilization", color="orange")
ax_sla.axhline(sla_threshold, color="red", linestyle="--", label="SLA Threshold")
ax_sla.set_title("SLA Violation Monitoring")
ax_sla.legend()
st.pyplot(fig_sla)

# Raw Dataset
st.subheader("Raw Dataset Preview")
st.dataframe(df.head(50), use_container_width=True)

# Summary Stats
st.subheader("Summary Statistics")
summary_stats = {
    "Average VM Count": [round(result_df["Optimized VM Count"].mean(), 2)],
    "Average Cost": [round(result_df["Cost"].mean(), 2)],
    "Average Energy": [round(result_df["Energy"].mean(), 2)],
}
summary_df = pd.DataFrame(summary_stats)
st.table(summary_df)

# Excel Export
st.markdown("### Export Optimized Results")
buffer = BytesIO()
result_df.to_excel(buffer, index=False, engine='openpyxl')
buffer.seek(0)
st.download_button(
    label="Download Optimized Results",
    data=buffer,
    file_name="optimized_vm_allocation.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
