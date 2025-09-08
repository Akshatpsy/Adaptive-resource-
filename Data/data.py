import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create /Data directory if not exists
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "Data")
os.makedirs(data_dir, exist_ok=True)

# Parameters
num_rows = 10000
start_time = datetime(2024, 1, 1, 0, 0, 0)

# Generate synthetic data
timestamps = [start_time + timedelta(hours=i) for i in range(num_rows)]
cpu_util = np.random.uniform(20, 90, num_rows).round(1)
mem_util = np.random.uniform(30, 80, num_rows).round(1)
disk_io = np.random.uniform(50, 200, num_rows).round(1)
network_bw = np.random.uniform(20, 100, num_rows).round(1)
active_vms = np.random.randint(5, 20, num_rows)
workload_types = ["CPU-intensive", "Memory-intensive", "I/O-intensive"]
resource_map = {"CPU-intensive": "High", "Memory-intensive": "Medium", "I/O-intensive": "Low"}
workload = np.random.choice(workload_types, num_rows)
resource_demand = [resource_map[w] for w in workload]

df = pd.DataFrame({
    "Timestamp": timestamps,
    "CPU Utilization (%)": cpu_util,
    "Memory Utilization (%)": mem_util,
    "Disk I/O (MB/s)": disk_io,
    "Network Bandwidth (Mbps)": network_bw,
    "Number of Active VMs": active_vms,
    "Workload Type": workload,
    "Resource Demand": resource_demand
})

# Save both formats
excel_path = os.path.join(data_dir, "synthetic_dataset.xlsx")
csv_path = os.path.join(data_dir, "synthetic_dataset.csv")
df.to_excel(excel_path, index=False)
df.to_csv(csv_path, index=False)

print(f" Dataset saved:\n - {excel_path}\n - {csv_path}")
