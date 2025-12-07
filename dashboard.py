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

from aws_scaling import update_aws_scaling_capacity  # mock AWS simulator

# -------------------------------
# Initialize session state values
# -------------------------------
if "runtime_result_df" not in st.session_state:
    st.session_state["runtime_result_df"] = None

if "runtime_recommended_vms" not in st.session_state:
    st.session_state["runtime_recommended_vms"] = None

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

# --- Stress test simulation for dataset ---
st.sidebar.markdown("### Stress Test Simulation")
cpu_spike = st.sidebar.slider("CPU Spike (%)", 0, 100, 0)
mem_boost = st.sidebar.slider("Memory Boost Multiplier", 1.0, 2.0, 1.0, step=0.1)

# Load dataset
df = pd.read_excel(dataset_path)

# Apply dataset modification
df["CPU Utilization (%)"] = np.clip(df["CPU Utilization (%)"] + cpu_spike, 0, 100)
df["Memory Utilization (%)"] = np.clip(df["Memory Utilization (%)"] * mem_boost, 0, 100)

# Load RL agent
q_learning_path = os.path.join(base_dir, "RL_Agent", "q_table.pkl")
dqn_model_path = os.path.join(base_dir, "RL_Agent", "models", "dqn_model.h5")

q_table = {}
model = None

if rl_method == "Q-learning":
    if os.path.exists(q_learning_path):
        with open(q_learning_path, "rb") as f:
            q_table = pickle.load(f)
elif rl_method == "DQN":
    if os.path.exists(dqn_model_path):
        st.success("DQN model loaded!")
        model = load_model(dqn_model_path)
    else:
        st.warning("DQN model not found.")

# -------------------------------
# TITLE
# -------------------------------
st.title("Adaptive Resource Provisioning using RL + SMO")
st.markdown("A dynamic cloud resource optimization dashboard using **Reinforcement Learning** and **Spider Monkey Optimization**.")

# -------------------------------
# 1️ RUNTIME SCENARIO TESTER
# -------------------------------
st.markdown("---")
st.header(" Runtime Scenario Tester (What-if Analysis)")

scenario = st.selectbox(
    "Choose a predefined scenario",
    ["Custom", "CPU Spike", "Memory Bottleneck", "Network Congestion", "Underutilized Cluster"],
)

# DEFAULT INPUT VALUES
cpu_default = 60
mem_default = 50
disk_default = 100.0
net_default = 50.0
active_default = 10
workload_default = "Mixed"
demand_default = "Medium"

# Scenario presets
if scenario == "CPU Spike":
    cpu_default, mem_default = 90, 50
    workload_default, demand_default = "CPU-intensive", "High"

elif scenario == "Memory Bottleneck":
    cpu_default, mem_default = 40, 90
    workload_default, demand_default = "Memory-intensive", "High"

elif scenario == "Network Congestion":
    cpu_default, mem_default = 50, 60
    net_default = 10.0
    workload_default, demand_default = "I/O-intensive", "Medium"

elif scenario == "Underutilized Cluster":
    cpu_default, mem_default = 20, 25
    active_default = 30
    demand_default = "Low"

# Input Sliders
colA, colB = st.columns(2)

with colA:
    cpu_input = st.slider("CPU Utilization (%)", 0, 100, cpu_default)
    mem_input = st.slider("Memory Utilization (%)", 0, 100, mem_default)
    disk_io_input = st.slider("Disk I/O (MB/s)", 0.0, 200.0, disk_default)
    net_bw_input = st.slider("Network Bandwidth (Mbps)", 0.0, 200.0, net_default)

with colB:
    active_vms_input = st.slider("Current Active VMs", 1, 100, active_default)

    workload_options = ["CPU-intensive", "Memory-intensive", "I/O-intensive", "Mixed"]
    workload_input = st.selectbox("Workload Type", workload_options, index=workload_options.index(workload_default))

    demand_options = ["Low", "Medium", "High", "Very High"]
    resource_demand_input = st.selectbox("Resource Demand Level", demand_options, index=demand_options.index(demand_default))

run_runtime = st.button(" Run Runtime Allocation")

# When the runtime button is clicked
if run_runtime:
    base_row = df.iloc[-1].copy()

    # Update only existing columns
    mapping = {
        "CPU Utilization (%)": cpu_input,
        "Memory Utilization (%)": mem_input,
        "Disk I/O (MB/s)": disk_io_input,
        "Network Bandwidth (Mbps)": net_bw_input,
        "Number of Active VMs": active_vms_input,
        "Workload Type": workload_input,
        "Resource Demand": resource_demand_input,
    }

    for key,val in mapping.items():
        if key in df.columns:
            base_row[key] = val

    runtime_df = pd.DataFrame([base_row])

    # small pop + iterations for speed
    runtime_pop, runtime_iter = 5, 5
    with st.spinner("Optimizing..."):
        runtime_result_df, runtime_fitness = spider_monkey.run_optimization(
            runtime_df, q_table, runtime_pop, runtime_iter
        )

    # Save results to session state
    st.session_state["runtime_result_df"] = runtime_result_df
    st.session_state["runtime_recommended_vms"] = int(runtime_result_df["Optimized VM Count"].iloc[-1])

    st.success("Runtime optimization completed!")

# -------------------------------
# DISPLAY RUNTIME RESULTS
# -------------------------------

runtime_result_df = st.session_state["runtime_result_df"]
recommended_vms = st.session_state["runtime_recommended_vms"]

if runtime_result_df is not None:
    st.metric("Recommended VMs (Runtime)", recommended_vms)

    # AWS SIMULATION BUTTON (works now!)
    if st.button(" Simulate AWS Auto Scaling"):
        update_aws_scaling_capacity(recommended_vms)
        st.success(f"Simulated: AWS Auto Scaling would set capacity to {recommended_vms} instances.")

    # Runtime VM Trend
    st.subheader("Runtime VM Allocation Trend")
    st.line_chart(runtime_result_df["Optimized VM Count"])

    # Cost/Energy
    ccols = [c for c in ["Cost", "Energy"] if c in runtime_result_df.columns]
    if ccols:
        st.subheader("Runtime Cost & Energy")
        st.line_chart(runtime_result_df[ccols])

# ----------------------------------------------------
# 2️ OFFLINE DATASET-BASED OPTIMIZATION
# ----------------------------------------------------
st.markdown("---")
st.header(" Offline Optimization on Workload Dataset")

with st.spinner("Running optimization..."):
    result_df, fitness_history = spider_monkey.run_optimization(df, q_table, pop_size, iterations)

# Summary metrics
col1, col2, col3 = st.columns(3)
col1.metric("Min Penalty", f"{min(fitness_history):.2f}")
col2.metric("Max Penalty", f"{max(fitness_history):.2f}")
col3.metric("Final VM Recommendation", int(result_df["Optimized VM Count"].iloc[-1]))

# VM Trend
st.subheader("Optimized VM Allocation Over Iterations")
st.line_chart(result_df["Optimized VM Count"])

# ----------------------------------------------------
# Q-TABLE VISUALIZATION
# ----------------------------------------------------
st.subheader("Q-Table Heatmap")

view_option = st.radio("Select View Mode", ["Single Q-Table", "Compare Two Q-Tables"], horizontal=True)

def generate_heatmap_matrix(q_table):
    if not q_table:
        return np.zeros((1,1)), []
    states = list(q_table.keys())
    action_labels = list(list(q_table.values())[0].keys())
    matrix = np.array([list(q_table[s].values()) for s in states])
    return matrix, action_labels

if rl_method == "Q-learning":
    matrix, action_labels = generate_heatmap_matrix(q_table)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(matrix, ax=ax, cmap="YlOrRd")
    st.pyplot(fig)
else:
    st.info("DQN does not use a Q-table.")

# ----------------------------------------------------
# COST & ENERGY
# ----------------------------------------------------
st.subheader("Cost Over Iterations")
st.line_chart(result_df["Cost"])

st.subheader("Energy Consumption Over Iterations")
st.line_chart(result_df["Energy"])

# SLA
st.subheader("SLA Violation Check (CPU > 75%)")
fig, ax = plt.subplots(figsize=(8,3))
ax.plot(df["CPU Utilization (%)"], color="orange")
ax.axhline(75, color="red", linestyle="--")
st.pyplot(fig)

# Raw Data
st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

# Summary Table
st.subheader("Summary Statistics")
summary = pd.DataFrame({
    "Average VM Count": [round(result_df["Optimized VM Count"].mean(), 2)],
    "Average Cost": [round(result_df["Cost"].mean(), 2)],
    "Average Energy": [round(result_df["Energy"].mean(), 2)],
})
st.table(summary)

# Export
st.markdown("### Export Optimized Results")
buf = BytesIO()
result_df.to_excel(buf, index=False)
buf.seek(0)
st.download_button("Download Results", buf, "optimized_vm_allocation.xlsx")
