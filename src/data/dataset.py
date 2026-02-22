import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
from pathlib import Path

class RuslanTTSDataset(Dataset):
    def __init__(self, data_dir, stats_path):
        self.data_dir = Path(data_dir)
        self.mel_files = sorted(list(self.data_dir.glob("*-mel.npy")))

        self.stats = []
        if Path(stats_path).exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                # Ожидается формат [p_min, p_max, e_min, e_max] или подобный из твоих скриптов
                self.stats = stats

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel_path = self.mel_files[idx]
        base_name = mel_path.name.replace("-mel.npy", "")

        pitch_path = self.data_dir / f"{base_name}-pitch.npy"
        energy_path = self.data_dir / f"{base_name}-energy.npy"
        duration_path = self.data_dir / f"{base_name}-duration.npy"
        text_path = self.data_dir / f"{base_name}-ids.npy"

        mel = np.load(mel_path)
        pitch = np.load(pitch_path)
        energy = np.load(energy_path)
        duration = np.load(duration_path)
        text = np.load(text_path)

        return {
            "id": base_name,
            "text": torch.from_numpy(text).long(),
            "mel": torch.from_numpy(mel).float(),
            "pitch": torch.from_numpy(pitch).float(),
            "energy": torch.from_numpy(energy).float(),
            "duration": torch.from_numpy(duration).long()
        }

def collate_fn_tts(batch):
    batch.sort(key=lambda x: len(x["text"]), reverse=True)
    
    ids = [x["id"] for x in batch]
    text_lens = [len(x["text"]) for x in batch]
    mel_lens = [x["mel"].shape[0] for x in batch]

    texts = [x["text"] for x in batch]
    mels = [x["mel"] for x in batch]
    pitches = [x["pitch"] for x in batch]
    energies = [x["energy"] for x in batch]
    durations = [x["duration"] for x in batch]

    text_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    mel_padded = pad_sequence(mels, batch_first=True, padding_value=0)
    pitch_padded = pad_sequence(pitches, batch_first=True, padding_value=0)
    energy_padded = pad_sequence(energies, batch_first=True, padding_value=0)
    dur_padded = pad_sequence(durations, batch_first=True, padding_value=0)

    return {
        "id": ids,
        "text": text_padded,
        "text_lens": torch.tensor(text_lens),
        "mel": mel_padded,
        "mel_lens": torch.tensor(mel_lens),
        "pitch": pitch_padded,
        "energy": energy_padded,
        "duration": dur_padded
    }
