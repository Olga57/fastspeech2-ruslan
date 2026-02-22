import os
import shutil
import subprocess
import sys


REPO_URL = "https://github.com/Olga57/fastspeech2-ruslan.git"
REPO_DIR = "fastspeech2-ruslan"

CKPT_PATH = os.path.join(REPO_DIR, "FastSpeech2_Weight", "checkpoint_49.pth")
VOCAB_PATH = os.path.join(REPO_DIR, "FastSpeech2_Weight", "vocab.json")


def run(cmd):
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    # Очистка старой папки, если она есть
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)

    print("Cloning repository...")
    # Git LFS (если доступно)
    try:
        run(["git", "lfs", "install"])
    except Exception as e:
        print(f"Warning: git lfs install failed: {e}")

    run(["git", "clone", REPO_URL])

    if os.path.exists(CKPT_PATH) and os.path.exists(VOCAB_PATH):
        print(f"OK: checkpoint found: {CKPT_PATH}")
        print(f"OK: vocab found: {VOCAB_PATH}")
    else:
        print("Warning: checkpoint/vocab not found after clone.")
        print(f"Expected: {CKPT_PATH}")
        print(f"Expected: {VOCAB_PATH}")

    print("Installing dependencies...")
    # Пин версии huggingface_hub для совместимости со speechbrain
    run([sys.executable, "-m", "pip", "install", "-q",
         "torchaudio", "librosa", "matplotlib", "textgrid", "speechbrain", "huggingface_hub==0.19.0"])

    print("Done.")
    print("If you are running this in Colab/Jupyter and you get import conflicts, restart runtime manually.")


if __name__ == "__main__":
    main()
