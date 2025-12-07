# aws_scaling.py  (mock version – no real AWS calls)

def update_aws_scaling_capacity(desired_capacity: int):
    """
    Mock function: In real deployment this will call AWS Auto Scaling API.
    For our project, we just simulate / log the scaling decision.
    """
    # Safety limits: 1 se 20 ke beech hi capacity rakho
    desired_capacity = max(1, min(desired_capacity, 20))

    # Yeh sirf console / terminal pe print karega (Streamlit run karte time dikhega)
    print(f"[SIMULATION] Would update AWS Auto Scaling to {desired_capacity} instances.")

    # Optional: dict return kar dete hain, agar future me use karna ho
    return {
        "status": "simulated",
        "desired_capacity": desired_capacity
    }
