import argparse
import os
import shutil

def clone_voice(file_path: str):
    """
    Copies the given audio file to the voices/ directory.
    """
    voices_dir = "voices/"
    if not os.path.exists(voices_dir):
        os.makedirs(voices_dir)

    if not os.path.isfile(file_path):
        print(f"Error: Input file '{file_path}' not found.")
        return

    try:
        shutil.copy(file_path, voices_dir)
        print(f"Successfully copied '{file_path}' to '{voices_dir}'")
    except Exception as e:
        print(f"Error copying file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone a new voice by copying an audio file.")
    parser.add_argument("file_path", type=str, help="Path to the audio file to be cloned.")
    args = parser.parse_args()

    clone_voice(args.file_path)