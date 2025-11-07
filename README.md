# Gradients Scheduler

A continuous worker service for training and evaluating models using the Gradients API.

## Installation

Clone the repository

Create a virtual environment:

```bash
python -m venv scheduler_venv
source scheduler_venv/bin/activate
```

Install the package:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Configuration
Create a .env file in the root directory with the following required variables (example in example.env):

```
GRADIENTS_API_KEY=your_api_key_here
GRADIENTS_API_URL=https://api.gradients.io
WANDB_ENTITY=your_wandb_entity_here
WANDB_API_KEY=your_wandb_api_key_here
CHECK_INTERVAL=600  # Optional, defaults to 600 seconds
HF_USERNAME=your_huggingface_username
HF_TOKEN=your_huggingface_token
S3_COMPATIBLE_ENDPOINT=your_s3_endpoint_url
S3_COMPATIBLE_ACCESS_KEY=your_s3_access_key
S3_COMPATIBLE_SECRET_KEY=your_s3_secret_key
S3_BUCKET_NAME=your_bucket_name

# Model loading configuration (optional)
USE_CPU_FOR_MODELS=false  # Set to true to use CPU instead of GPU for model merging and tokenizer updates

# RunPod configuration (optional, for offloading model merging to RunPod serverless)
USE_RUNPOD_FOR_MERGE=false  # Set to true to use RunPod for model merging
RUNPOD_API_KEY=your_runpod_api_key  # Required if USE_RUNPOD_FOR_MERGE=true
RUNPOD_ENDPOINT_ID=your_endpoint_id  # Required if USE_RUNPOD_FOR_MERGE=true
RUNPOD_TIMEOUT=10000  # Optional, defaults to 10000 seconds (~3 hour)
RUNPOD_MAX_RETRIES=3  # Optional, defaults to 3 retry attempts

# Delay settings (all optional)
MIN_HOURS_BETWEEN_RUNS=6
MAX_HOURS_BETWEEN_RUNS=8
MIN_DAYS_BETWEEN_RUNS=2
MAX_DAYS_BETWEEN_RUNS=3
```

## RunPod Integration for Large Model Merging

For large models (30B-70B+) that exceed local GPU capacity, you can offload model merging to RunPod serverless GPUs.

### Why Use RunPod?

- **Handle any model size**: Merge 70B, 405B+ models without local GPU constraints
- **Pay per use**: Only charged for actual merge time (~$1-3 per merge)
- **Auto-scaling**: Handles multiple merges in parallel
- **No idle costs**: No need to keep GPUs running 24/7

### Setup Steps

#### 1. Build and Push Docker Image

```bash
# Build the RunPod container
docker build -f Dockerfile.runpod -t your-dockerhub-username/gradients-merger:latest .

# Push to Docker Hub (or RunPod registry)
docker push your-dockerhub-username/gradients-merger:latest
```

#### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Container Image**: `your-dockerhub-username/gradients-merger:latest`
   - **GPU Type**: H200
   - **GPUs per Worker**: 3
   - **Max Workers**: 2 (depending on your needs)
   - **Execution Timeout**: 1800 seconds (0.5 hour)
   - **Active Workers**: 0 (serverless, auto-scales)
4. Copy your endpoint ID

#### 3. Configure Environment Variables

Add to your `.env` file:

```bash
USE_RUNPOD_FOR_MERGE=true
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
RUNPOD_TIMEOUT=10000
RUNPOD_MAX_RETRIES=3
```

### How It Works

1. **Scheduler submits merge job** to RunPod endpoint with model IDs
2. **RunPod spins up worker** with 3x H200 GPUs (cold start ~10-30s)
3. **Worker downloads models**, merges LoRA with base model using all 3 GPUs, uploads to HuggingFace Hub
4. **Worker returns** the merged model repo ID
5. **Billing stops** automatically when worker completes
6. **Retry mechanism**: Automatically retries up to 3 times on failure with exponential backoff

### Troubleshooting

**Job timeout**: Increase `RUNPOD_TIMEOUT` in `.env` for very large models

**Out of memory**: Ensure RunPod endpoint is configured with 4x H100 80GB GPUs

**Cold start delays**: First merge may take longer (~30s) as RunPod spins up workers

**Authentication errors**: Verify `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` are correct

Running the Worker
To run the worker:

```bash
python -m gradients_worker.main
```

Configuration Files
The main configuration is in `config.yaml`. You need to create your own `config.yaml` file (not tracked by git) based on the provided `example-config.yaml`.

```yaml
########################################################
# Example task configuration #1
########################################################
# Multiple tasks can run in parallel, their names appear in the logs
name-of-the-task:
  # If enabled is false, the task will not be run
  enabled: true
  wandb_project: "wandb-project-name"
  # Time interval between the end of the previous run and the start of the next run
  run_intervals:
    min_days: 0
    max_days: 0
    min_hours: 0
    max_hours: 0
  # Datasets are downloaded, merged, shuffled, split into chunks, and uploaded to MinIO for each gradients task
  # Final dataset will default to instruction, output
  datasets:
    # Dataset #1, name is the huggingface dataset identifier, fields are the column names in the dataset
    - name: "yahma/alpaca-cleaned"
      field_instruction: "instruction"
      field_input: "input" # Optional, can be left empty
      field_output: "output"
    # Dataset #2 ...
    - name: "databricks/databricks-dolly-15k"
      field_instruction: "instruction"
      field_input: "context" # Optional, can be left empty
      field_output: "response"
    # etc
  # HuggingFace model identifier to be finetuned, it will be downloaded, verified, and merged with its tokenizer
  model_repo: "unsloth/Meta-Llama-3.1-8B"
  # If specified, the model tokenizer will be updated to the specified HuggingFace repository
  # tokenizer_id:
  # Time to complete the task, in hours
  hours_to_complete: 8
  # Number of samples to use for each gradients training job
  samples_per_training: 150_000
  # Size of the final test dataset, in percentage of the total dataset, this is never shared with gradients
  final_test_size: 0.05
  # Random seed for shuffling the dataset
  random_seed: 45
```

## Run

We advise that you create a screen session in which the scheduler can run indefinitely (until stopped):

```bash
cd gradients-scheduler
screen -S gradients-scheduler -L -Logfile ./gradients-scheduler.log -a
source .venv/bin/activate
python -m gradients_worker.main
```

Then to exit the screen session type, ctr+a then ctrl+d
To observe the logs continuously from outside the screen session, run:

```
tail -n 1000 -f gradients-scheduler.log
```

# Evaluation

The gradients scheduler will store all gradients-related metrics in your specified Wandb project.
