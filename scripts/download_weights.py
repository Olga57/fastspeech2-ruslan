import os
import gdown
import shutil

def download_weights():
    # Твоя ссылка на папку
    folder_url = 'https://drive.google.com/drive/folders/17ZeLKw0o0C8erGcs1dX0OwLHWh03zhjO?usp=sharing'
    temp_dir = 'temp_weights'
    
    # Создаем нужные директории в проекте
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    print("📥 Скачиваем папку с весами из Google Drive...")
    # Скачиваем всё содержимое папки
    gitfs https://github.com/user/repository.git /path/to/mountpoint
    gdown.download_folder(url=folder_url, output=temp_dir, quiet=False, use_cookies=False)
    
    print("🔄 Распределяем файлы по папкам...")
    
    # Ищем файлы во временной папке и переносим их
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            src_path = os.path.join(root, file)
            
            if file.endswith('.pth'):
                # Переименовываем чекпоинт, чтобы код синтеза его нашел
                dst_path = os.path.join("checkpoints", "checkpoint_49.pth")
                shutil.move(src_path, dst_path)
                print(f"✅ Перемещено: {file} -> {dst_path}")
                
            elif file.endswith('.json'):
                # Переносим словари и статистику в artifacts
                dst_path = os.path.join("artifacts", file)
                shutil.move(src_path, dst_path)
                print(f"✅ Перемещено: {file} -> {dst_path}")
                
    # Удаляем временную папку за собой
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("🎉 Скачивание завершено! Все веса на месте.")

if __name__ == "__main__":
    download_weights()
