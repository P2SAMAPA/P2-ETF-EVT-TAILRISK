"""
Upload results to Hugging Face dataset repository.
"""

import json
import os
from datetime import datetime
from huggingface_hub import HfApi, upload_file
import config

def push_daily_result(payload: dict):
    """
    Saves payload as JSON and uploads to HF dataset repo.
    Filename format: evt_tailrisk_YYYY-MM-DD.json
    """
    filename = f"evt_tailrisk_{config.TODAY}.json"
    
    # Save locally
    with open(filename, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    
    print(f"Saved local file: {filename}")
    
    # Upload to Hugging Face
    if config.HF_TOKEN:
        api = HfApi(token=config.HF_TOKEN)
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=config.HF_OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.HF_OUTPUT_REPO}/{filename}")
    else:
        print("HF_TOKEN not set. Skipping upload.")
    
    # Clean up local file (optional)
    # os.remove(filename)
