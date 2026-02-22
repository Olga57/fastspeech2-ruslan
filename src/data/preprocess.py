import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import json
import textgrid

def extract_features(wav_dir, tg_dir, out_root, sr=22050, n_fft=1024, hop_len=256, n_mels=80, ref_db=20):
    wav_dir = Path(wav_dir)
    tg_dir = Path(tg_dir)
    out_root = Path(out_root)
    
    feat_dir = out_root / "features"
    stats_dir = out_root / "stats"
    feat_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    tg_files = list(tg_dir.glob("*.TextGrid"))
    print(f"🔍 Найдено TextGrid файлов: {len(tg_files)}")

    pitch_list, energy_list, phonemes = [], [], set()
    count, errors = 0, 0

    for tg_path in tqdm(tg_files, desc="Извлечение признаков"):
        base_name = tg_path.stem
        wav_path = wav_dir / f"{base_name}.wav"

        if not wav_path.exists():
            wav_path = wav_dir / f"{base_name.split('_')[0]}.wav"
        if not wav_path.exists():
            errors += 1
            continue

        try:
            y, _ = librosa.load(wav_path, sr=sr)

            # Mel-spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - ref_db + 100) / 100
            mel_t = mel_db.T 

            # Energy & Pitch
            energy = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_len)[0]
            f0, _, _ = librosa.pyin(y, fmin=50, fmax=1100, sr=sr, frame_length=n_fft, hop_length=hop_len)
            f0 = np.nan_to_num(f0)

            # TextGrid parsing
            tg = textgrid.TextGrid.fromFile(str(tg_path))
            phones_tier = tg[1] if len(tg) > 1 else tg[0]
            
            phone_ids, durations = [], []
            for interval in phones_tier:
                ph = interval.mark
                if ph == "": continue
                phonemes.add(ph)

                dur_sec = interval.maxTime - interval.minTime
                dur_frames = int(dur_sec * sr / hop_len + 0.5)

                phone_ids.append(ph)
                durations.append(dur_frames)

            # Выравнивание длин (Aligning lengths)
            min_len = min(mel_t.shape[0], len(energy), len(f0), sum(durations))
            mel_t = mel_t[:min_len]
            energy = energy[:min_len]
            f0 = f0[:min_len]

            current_dur_sum = sum(durations)
            if current_dur_sum != min_len:
                diff = min_len - current_dur_sum
                if len(durations) > 0:
                    durations[-1] += diff
                    if durations[-1] < 0: continue

            if len(phone_ids) == 0: continue

            if len(f0[f0 > 0]) > 0:
                pitch_list.extend(f0[f0 > 0])
            energy_list.extend(energy)

            # Сохранение (Saving)
            np.save(feat_dir / f"{base_name}-mel.npy", mel_t)
            np.save(feat_dir / f"{base_name}-pitch.npy", f0)
            np.save(feat_dir / f"{base_name}-energy.npy", energy)
            np.save(feat_dir / f"{base_name}-duration.npy", np.array(durations))
            np.save(feat_dir / f"{base_name}-text_temp.npy", np.array(phone_ids))
            count += 1

        except Exception as e:
            print(f"Ошибка на {base_name}: {e}")
            errors += 1

    print(f"\n✅ Обработано {count} файлов. Ошибок: {errors}")

    # Создание словаря (Vocab mapping)
    special_syms = ["<pad>", "<unk>", "sil", "sp"]
    valid_phonemes = sorted(list(phonemes - set(special_syms)))
    vocab_dict = {ph: i for i, ph in enumerate(special_syms + valid_phonemes)}

    with open(out_root / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, indent=2)

    # Конвертация фонем в ID
    temp_files = list(feat_dir.glob("*-text_temp.npy"))
    for f in tqdm(temp_files, desc="Конвертация в ID"):
        ph_arr = np.load(f)
        id_arr = [vocab_dict.get(p, vocab_dict["<unk>"]) for p in ph_arr]
        np.save(f.with_name(f.name.replace("text_temp", "ids")), np.array(id_arr))
        f.unlink()

    # Статистика
    p_arr = np.array(pitch_list) if pitch_list else np.array([0., 1.])
    e_arr = np.array(energy_list) if energy_list else np.array([0., 1.])
    simple_stats = [float(p_arr.min()), float(p_arr.max()), float(e_arr.min()), float(e_arr.max())]

    with open(stats_dir / "stats.json", "w") as f:
        json.dump(simple_stats, f)
    print("💾 Все фичи и статистика сохранены!")

if __name__ == "__main__":
    # Пример вызова
    extract_features(
        wav_dir="/content/RUSLAN_EXTRACTED/RUSLAN",
        tg_dir="/content/mfa_output",
        out_root="/content/fs2_ruslan/artifacts"
    )
