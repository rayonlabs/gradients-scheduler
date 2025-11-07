#!/usr/bin/env python3
"""Test RunPod handler locally without Docker."""

import os
import sys
from dotenv import load_dotenv

# Add src to path so we can import the handler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gradients_worker.runpod_handler import handler

# Load environment variables
load_dotenv()

# Get config from .env
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

# Validate config
if not all([HF_TOKEN, HF_USERNAME]):
    print("❌ Missing required environment variables in .env")
    print(f"  HF_TOKEN: {'✓' if HF_TOKEN else '✗'}")
    print(f"  HF_USERNAME: {'✓' if HF_USERNAME else '✗'}")
    exit(1)

print("=" * 60)
print("Local RunPod Handler Test")
print("=" * 60)
print(f"HF Username: {HF_USERNAME}")
print()

# Test job input (same format as RunPod serverless)
job = {
    "input": {
        "lora_model_id": "samoline/dc2283b8-97bf-4312-bf4e-4de72ffa2bb0",
        "base_model_id": "Maykeye/TinyLLama-v0",
        "anonimize": True,
        "hf_token": HF_TOKEN,
        "hf_username": HF_USERNAME,
    }
}

print("Running handler locally...")
print(f"  LoRA: {job['input']['lora_model_id']}")
print(f"  Base: {job['input']['base_model_id']}")
print()

try:
    # Call the handler directly
    result = handler(job)

    if result.get("status") == "success":
        print("✓ Handler completed successfully!")
        print(f"  Merged model: {result['model_repo_id']}")
    else:
        print("❌ Handler failed!")
        print(f"  Error: {result.get('error', 'Unknown error')}")
        exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
