import os
from huggingface_hub import hf_hub_download

def download_models():
    """
    Downloads the required models from Hugging Face Hub to a local directory.
    """
    model_repo = "ResembleAI/chatterbox"
    local_dir = "models"

    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    files_to_download = [
        "config.json",
        "s3gen.safetensors",
        "t3_cfg.safetensors",
        "tokenizer.json",
        "vocab.txt"
    ]

    print("Downloading models...")
    for file in files_to_download:
        print(f"Downloading {file}...")
        try:
            hf_hub_download(
                repo_id=model_repo,
                filename=file,
                local_dir=local_dir,
                local_dir_use_symlinks=False # Use False to copy files directly
            )
            print(f"Successfully downloaded {file}")
        except Exception as e:
            print(f"Error downloading {file}: {e}")
            # Decide if you want to exit on error or just log it
            # For a build script, it's probably best to exit
            exit(1)

    print("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()