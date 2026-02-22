import torch
import json
import numpy as np
import os
from pathlib import Path
from src.model.fastspeech2 import FastSpeech2

class Synthesizer:
    def __init__(self, ckpt_path, vocab_path, dict_path, config, device='cuda'):
        self.device = device
        self.config = config

        # Загрузка словаря токенов
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        # Загрузка словаря MFA (текст -> фонемы)
        self.word_to_phones = {}
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        self.word_to_phones[parts[0].lower()] = parts[1:]
        else:
            print(f"⚠️ Словарь {dict_path} не найден! Преобразование текста может не сработать.")

        # Инициализация модели
        vocab_size = len(self.vocab) + 1
        self.model = FastSpeech2(self.config, vocab_size=vocab_size).to(self.device)

        print(f"🔄 Загрузка весов: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        print("✅ Модель FastSpeech2 готова!")

    def text_to_sequence(self, text):
        text = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        words = text.split()
        sequence = []
        for w in words:
            if w in self.word_to_phones:
                for p in self.word_to_phones[w]:
                    sequence.append(self.vocab.get(p, self.vocab.get("<UNK>", 0)))
                if "<SIL>" in self.vocab:
                    sequence.append(self.vocab["<SIL>"])
            else:
                print(f"⚠️ Слово '{w}' не найдено в словаре (пропускаем)")

        if not sequence:
            return None
        return torch.tensor(sequence).long().unsqueeze(0).to(self.device)

    def synthesize(self, text, vocoder, speed=1.0, pitch=1.0, energy=1.0):
        src = self.text_to_sequence(text)
        if src is None:
            print("❌ Ошибка: пустая последовательность.")
            return None

        src_len = torch.tensor([src.shape[1]]).to(self.device)

        with torch.no_grad():
            output = self.model(
                src_seq=src, 
                src_lens=src_len,
                d_control=speed, 
                p_control=pitch, 
                e_control=energy
            )
            
            mel_postnet = output["mel_postnet"]
            mel_vocoder = mel_postnet.transpose(1, 2)
            audio = vocoder(mel_vocoder)
            return audio.squeeze().cpu().numpy(), mel_postnet.squeeze().cpu().numpy()

def load_vocoder(device='cuda'):
    print("🔄 Загрузка вокодера HiFi-GAN...")
    vocoder_data = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
    vocoder = vocoder_data[0] if isinstance(vocoder_data, tuple) else vocoder_data
    return vocoder.to(device).eval()
