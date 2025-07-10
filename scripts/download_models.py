import os
from huggingface_hub import snapshot_download

def download_models():
    """
    Downloads the required models from Hugging Face Hub to a local directory.
    """
    model_repo = "ResembleAI/chatterbox"
    local_dir = "models"

    print("Downloading models...")
    try:
        # Using snapshot_download to fetch all model files at once.
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
        )
        print("All models downloaded successfully.")
    except Exception as e:
        print(f"Error downloading models: {e}")
        exit(1)

if __name__ == "__main__":
    download_models()