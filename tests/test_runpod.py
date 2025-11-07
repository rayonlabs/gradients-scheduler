#!/usr/bin/env python3
"""Test RunPod serverless endpoint for model merging."""

import os
import time
from dotenv import load_dotenv
import runpod

# Load environment variables
load_dotenv()

# Get config from .env
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

# Validate config
if not all([RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, HF_TOKEN, HF_USERNAME]):
    print("❌ Missing required environment variables in .env")
    print(f"  RUNPOD_API_KEY: {'✓' if RUNPOD_API_KEY else '✗'}")
    print(f"  RUNPOD_ENDPOINT_ID: {'✓' if RUNPOD_ENDPOINT_ID else '✗'}")
    print(f"  HF_TOKEN: {'✓' if HF_TOKEN else '✗'}")
    print(f"  HF_USERNAME: {'✓' if HF_USERNAME else '✗'}")
    exit(1)

# Set RunPod API key
runpod.api_key = RUNPOD_API_KEY

print("=" * 60)
print("RunPod Serverless Merge Test (Traditional Queue-Based)")
print("=" * 60)
print(f"Endpoint ID: {RUNPOD_ENDPOINT_ID}")
print(f"HF Username: {HF_USERNAME}")
print()

# Test job input
job_input = {
    "lora_model_id": "samoline/dc2283b8-97bf-4312-bf4e-4de72ffa2bb0",
    "base_model_id": "Maykeye/TinyLLama-v0",
    "anonimize": True,
    "hf_token": HF_TOKEN,
    "hf_username": HF_USERNAME,
}

print("Submitting job to RunPod...")
print(f"  LoRA: {job_input['lora_model_id']}")
print(f"  Base: {job_input['base_model_id']}")
print()

try:
    # Initialize endpoint
    endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)

    # Submit job
    print("Submitting job (may take 30s+ on cold start to spin up worker)...")
    job = endpoint.run({"input": job_input})

    print(f"✓ Job submitted successfully!")
    print(f"  Job ID: {job.job_id}")
    print()

    # Poll for completion
    print("Waiting for job to complete...")
    start_time = time.time()

    while True:
        status = job.status()
        elapsed = time.time() - start_time

        print(f"  Status: {status} (elapsed: {elapsed:.0f}s)")

        if status in ["COMPLETED", "FAILED"]:
            break

        if elapsed > 1800:  # 30 min timeout
            print("❌ Job timed out after 30 minutes")
            exit(1)

        time.sleep(10)

    # Get result
    print()
    output = job.output()

    if status == "COMPLETED" and output.get("status") == "success":
        print("✓ Job completed successfully!")
        print(f"  Merged model: {output['model_repo_id']}")
    else:
        print("❌ Job failed!")
        print(f"  Error: {output.get('error', 'Unknown error')}")
        print(f"  Full output: {output}")
        exit(1)

except Exception as e:
    print(f"❌ Error: {e}")

    # Print response details if available
    if hasattr(e, 'response') and e.response is not None:
        print(f"\nResponse Status: {e.response.status_code}")
        print(f"Response Headers: {dict(e.response.headers)}")
        print(f"Response Body: {e.response.text}")

    import traceback
    traceback.print_exc()
    exit(1)
