import os
import gdown

def download_weights():
    # Создаем папки
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    # ВСТАВЬ СЮДА СВОИ ID ФАЙЛОВ ИЗ GOOGLE ДИСКА ВМЕСТО МОИХ СЛОВ!
    files = {
        "checkpoints/checkpoint_49.pth": "ВСТАВЬ_ID_ДЛЯ_CHECKPOINT",
        "artifacts/vocab.json": "ВСТАВЬ_ID_ДЛЯ_VOCAB",
        "artifacts/stats.json": "ВСТАВЬ_ID_ДЛЯ_STATS",
    }
    
    for path, file_id in files.items():
        if not os.path.exists(path):
            print(f"Скачиваем {path}...")
            gdown.download(id=file_id, output=path, quiet=False)
        else:
            print(f"Файл {path} уже существует.")

if __name__ == "__main__":
    download_weights()
