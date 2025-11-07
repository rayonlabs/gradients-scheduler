# Tests

This directory contains test scripts for the RunPod serverless integration.

## Test Files

- **`test_runpod.py`**: ✅ Test the RunPod serverless endpoint using the RunPod SDK. **Main integration test.**
- **`test_runpod_local.py`**: ✅ Test the handler function directly by importing it locally (uses local GPU/CPU).
- **`test_runpod_health.py`**: ✅ Quick health check of the RunPod endpoint (shows worker status and job stats).

## Running Tests

Make sure you have a `.env` file in the root directory with:
```bash
RUNPOD_API_KEY=your_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
HF_TOKEN=your_hf_token
HF_USERNAME=your_hf_username
```

Run tests from the tests directory:
```bash
cd tests
python test_runpod.py
```

## Notes

- All tests load credentials from `.env` file (never hardcoded)
- Test model IDs (like `samoline/dc2283b8-...`) are just test fixtures
- `.http` file for manual HTTP testing is gitignored
