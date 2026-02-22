import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

def prepare_mfa_data(raw_dir, extract_dir, mfa_input_dir):
    raw_dir = Path(raw_dir)
    wav_dir = Path(extract_dir) / "RUSLAN"
    mfa_input_dir = Path(mfa_input_dir)

    if mfa_input_dir.exists():
        shutil.rmtree(mfa_input_dir)
    mfa_input_dir.mkdir(parents=True, exist_ok=True)

    csv_candidates = list(raw_dir.glob("*metadata*.csv"))
    if not csv_candidates:
        raise FileNotFoundError("CSV файл с метаданными не найден!")

    csv_path = csv_candidates[0]
    print(f"📖 Читаем метаданные из: {csv_path}")

    # Создаем .lab файлы
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("✍️ Генерируем .lab файлы...")
    for line in tqdm(lines):
        parts = line.strip().split('|')
        if len(parts) < 2: continue

        filename_base = parts[0].strip()
        text = parts[1].strip().replace('+', '')

        wav_path = wav_dir / f"{filename_base}.wav"
        if not wav_path.exists():
            wav_path = wav_dir / f"{filename_base.split('_')[0]}.wav"

        if wav_path.exists():
            lab_path = wav_path.with_suffix('.lab')
            with open(lab_path, "w", encoding="utf-8") as lf:
                lf.write(text)
            count += 1

    print(f"✅ Создано {count} .lab файлов.")

    # Конвертация аудио в 16kHz для MFA
    wav_files = list(wav_dir.glob("*.wav"))
    lab_files = list(wav_dir.glob("*.lab"))
    print(f"🎶 Найдено {len(wav_files)} WAV и {len(lab_files)} LAB. Начинаем конвертацию...")

    for lab in tqdm(lab_files, desc="Копирование LAB"):
        shutil.copy(lab, mfa_input_dir / lab.name)

    error_count = 0
    for wav in tqdm(wav_files, desc="Конвертация WAV (16kHz)"):
        out_path = mfa_input_dir / wav.name
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(wav),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out_path)
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            error_count += 1

    print(f"\n🎉 Подготовка завершена. Файлы для MFA: {mfa_input_dir}")
    if error_count > 0:
        print(f"⚠️ Ошибок конвертации: {error_count}")

if __name__ == "__main__":
    prepare_mfa_data(
        raw_dir="/content/RUSLAN_RAW",
        extract_dir="/content/RUSLAN_EXTRACTED",
        mfa_input_dir="/content/mfa_data_16k"
    )
