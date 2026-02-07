import subprocess
import time
import sys
import os

import urllib.request

def run_dashboard():
    print("Starting Streamlit Dashboard...")

    # Fetch public IP for LocalTunnel password
    try:
        public_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()
        print(f"\n IMPORTANT: Your Public IP is: {public_ip}")
        print(f" (Copy this IP for the 'Tunnel Password' if asked)\n")
    except Exception as e:
        print(f"Could not fetch public IP: {e}")

    # Start Streamlit in the background
    # Using sys.executable to ensure we use the current python environment
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port", "8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for Streamlit to initialize
    print("Waiting for Streamlit to initialize...")
    time.sleep(5)
    
    print("Starting LocalTunnel...")
    # Start LocalTunnel
    # cmd /c lt ... is safer on Windows if lt is a bat file
    lt_process = subprocess.Popen(
        ["cmd", "/c", "lt", "--port", "8501"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("Dashboard is running.")
    print(f" -> Local Link: http://localhost:8501")
    print(" -> Public Link: Check the output below (requires IP password above)")
    print("Press Ctrl+C to stop.")

    try:
        # Read output from localtunnel to find the URL
        while True:
            output = lt_process.stdout.readline()
            if output:
                print(f"LocalTunnel: {output.strip()}")
            
            # Also check streamlit output just in case
            st_out = streamlit_process.stdout.readline()
            if st_out:
                # Filter out some noisy streamlit logs if needed, or just print
                pass # print(f"Streamlit: {st_out.strip()}")

            if lt_process.poll() is not None:
                print("LocalTunnel process ended unexpectedly.")
                break
            if streamlit_process.poll() is not None:
                print("Streamlit process ended unexpectedly.")
                print(streamlit_process.stderr.read())
                break
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping processes...")
        streamlit_process.terminate()
        lt_process.terminate()

if __name__ == "__main__":
    run_dashboard()
