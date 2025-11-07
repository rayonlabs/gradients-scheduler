#!/usr/bin/env python3
"""Check RunPod endpoint health and worker status."""

import os
from dotenv import load_dotenv
import runpod

# Load environment variables
load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
    print("❌ Missing RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID in .env")
    exit(1)

# Set API key
runpod.api_key = RUNPOD_API_KEY

print("=" * 60)
print("RunPod Endpoint Health Check")
print("=" * 60)
print(f"Endpoint ID: {RUNPOD_ENDPOINT_ID}")
print()

# Initialize endpoint
endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)

# Get health status
health = endpoint.health()

print("Workers:")
print(f"  Ready: {health['workers']['ready']}")
print(f"  Idle: {health['workers']['idle']}")
print(f"  Running: {health['workers']['running']}")
print(f"  Throttled: {health['workers']['throttled']}")
print(f"  Initializing: {health['workers']['initializing']}")
print(f"  Unhealthy: {health['workers']['unhealthy']}")
print()

print("Jobs:")
print(f"  In Queue: {health['jobs']['inQueue']}")
print(f"  In Progress: {health['jobs']['inProgress']}")
print(f"  Completed: {health['jobs']['completed']}")
print(f"  Failed: {health['jobs']['failed']}")
print(f"  Retried: {health['jobs']['retried']}")
print()

# Overall status
if health['workers']['ready'] > 0 and health['workers']['unhealthy'] == 0:
    print("✅ Endpoint is healthy and ready")
else:
    print("⚠️ Endpoint may have issues - check worker status")
