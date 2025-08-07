# steps/serve_bentoml.py
import subprocess
import sys
import argparse
import time
import requests# For health checking the server

def start_bentoml_server(
    # CORRECTED: The argument is a bento_tag, not a model_tag
    bento_tag: str,
    port: int = 3000,
    reload: bool = True,
    foreground: bool = False,
    health_endpoint: str = "/healthz",
    max_retries: int = 10,
):
    """
    Starts a BentoML server for a given Bento.
    
    Args:
        bento_tag (str): The Bento tag to serve (e.g., 'customer_churn_prediction_service:j3kv2vdtxw2wclb3').
        port (int): Port to serve on.
        reload (bool): Enable hot-reloading for development.
        foreground (bool): Run in foreground for debugging.
        health_endpoint (str): Health check path.
        max_retries (int): Number of health check attempts.
    """
    # CORRECTED: The command serves the bento_tag
    cmd = [sys.executable, "-m", "bentoml", "serve", bento_tag, "--port", str(port)]
    if reload:
        cmd.append("--reload")
    
    print(f"Starting BentoML server for Bento '{bento_tag}' on port {port}...")

    if not foreground:
        print(" (in background/detached mode)")
        # Use CREATE_NEW_PROCESS_GROUP to allow the process to live beyond the parent script
        # DETACHED_PROCESS is not needed here as it's a Windows-specific flag
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        print(" (in foreground/debug mode)")
        creationflags = 0

    try:
        # Start the subprocess
        process = subprocess.Popen(
            cmd,
            creationflags=creationflags,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False # Important for security and portability
        )

        if foreground:
            print("Server running in foreground. Press Ctrl+C to stop.")
            # Communicate to print stdout/stderr live
            stdout, stderr = process.communicate()
            if stdout:
                print("STDOUT:", stdout.decode())
            if stderr:
                print("STDERR:", stderr.decode())
            return process

        # Poll the health endpoint for background process
        url = f"http://localhost:{port}{health_endpoint}"
        print("Waiting for server to become healthy...")
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    print(f"✅ Server started successfully! Endpoint: http://localhost:{port}")
                    print(f"Process ID: {process.pid} (To stop, run: taskkill /PID {process.pid} /F)")
                    return process
            except requests.exceptions.RequestException:
                print(f"Health check attempt {attempt+1}/{max_retries} failed. Retrying in 3s...")
                time.sleep(3)
        
        print("❌ Failed to start server after retries.")
        stdout, stderr = process.communicate(timeout=5)
        if stdout:
            print("Server stdout:", stdout.decode())
        if stderr:
            print("Server stderr:", stderr.decode())
        
        process.terminate()
        return None

    except Exception as e:
        print(f"An error occurred while trying to start the server: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a BentoML server.")
    # CORRECTED: Help text now refers to a Bento tag
    parser.add_argument("bento_tag", type=str, help="The Bento tag to be served (e.g., 'customer_churn_prediction_service:latest')")
    parser.add_argument("--port", type=int, default=3000, help="Port to serve on")
    parser.add_argument("--no-reload", action="store_true", help="Disable hot-reloading")
    parser.add_argument("--foreground", action="store_true", help="Run in foreground for debugging")
    parser.add_argument("--health-endpoint", type=str, default="/healthz", help="Health check endpoint")
    
    args = parser.parse_args()
    start_bentoml_server(
        args.bento_tag,
        args.port,
        reload=not args.no_reload,
        foreground=args.foreground,
        health_endpoint=args.health_endpoint,
    )