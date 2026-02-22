import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pathlib import Path
from torch.utils.data import DataLoader

# Импортируем наши модули из папки src!
from src.data.dataset import RuslanTTSDataset, collate_fn_tts
from src.model.fastspeech2 import FastSpeech2
from src.utils.tools import plot_spectrogram_to_numpy

CONFIG = {
    "batch_size": 16,
    "lr": 0.001,
    "epochs": 50,
    "log_step": 10,
    "visualize_step": 200,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "stats_path": "artifacts/stats.json",
    "data_dir": "artifacts/features",
    "vocab_path": "artifacts/vocab.json",
    "checkpoint_dir": "checkpoints"
}

# Конфиг модели (тот же, что и при инференсе)
model_config = {
    "transformer": {
        "encoder_layer": 4, "encoder_head": 2, "encoder_hidden": 256,
        "decoder_layer": 4, "decoder_head": 2, "decoder_hidden": 256,
        "fft_conv1d_filter_size": 1024, "fft_conv1d_kernel_size": [9, 1], "dropout": 0.1
    },
    "audio": { "n_mels": 80 }
}

def train():
    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    wandb.init(project="fs2_ruslan_fixed", config=CONFIG)

    print("📦 Инициализация датасета...")
    train_dataset = RuslanTTSDataset(CONFIG['data_dir'], CONFIG['stats_path'])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn_tts, 
        num_workers=2
    )

    with open(CONFIG['vocab_path']) as f:
        vocab = json.load(f)
    vocab_size = len(vocab) + 1

    print("🧠 Инициализация модели...")
    model = FastSpeech2(model_config, vocab_size=vocab_size).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    print("🚀 Начинаем обучение...")
    global_step = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0

        for i, batch in enumerate(train_loader):
            # Распаковываем батч и кидаем на GPU/CPU
            texts = batch["text"].to(CONFIG['device'])
            mels = batch["mel"].to(CONFIG['device']).transpose(1, 2)
            pitches = batch["pitch"].to(CONFIG['device'])
            energies = batch["energy"].to(CONFIG['device'])
            durations = batch["duration"].to(CONFIG['device'])
            text_lens = batch["text_lens"].to(CONFIG['device'])
            mel_lens = batch["mel_lens"].to(CONFIG['device'])

            optimizer.zero_grad()
            
            # Forward pass
            output = model(
                src_seq=texts, src_lens=text_lens, 
                mel_target=mels, mel_lens=mel_lens,
                d_target=durations, p_target=pitches, e_target=energies
            )

            # Считаем лоссы
            loss_mel = l1_loss(output['mel_output'], mels)
            loss_postnet = l1_loss(output['mel_postnet'], mels)
            loss_dur = mse_loss(output['log_duration'], torch.log(durations.float() + 1))
            loss_pitch = mse_loss(output['pitch'], pitches)
            loss_energy = mse_loss(output['energy'], energies)

            total_loss = loss_mel + loss_postnet + loss_dur + loss_pitch + loss_energy

            # Backward pass
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            epoch_loss += total_loss.item()

            # Логирование в Weights & Biases
            if global_step % CONFIG['log_step'] == 0:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/loss_mel": loss_mel.item(),
                    "train/loss_dur": loss_dur.item(),
                    "train/loss_pitch": loss_pitch.item(),
                }, step=global_step)
                print(f"Epoch {epoch} | Step {global_step} | Loss: {total_loss.item():.4f}")

            # Визуализация спектрограмм
            if global_step % CONFIG['visualize_step'] == 0:
                model.eval()
                with torch.no_grad():
                    tgt = mels[0].cpu().numpy()
                    pred = output['mel_postnet'][0].cpu().numpy()
                    img = plot_spectrogram_to_numpy(tgt.T, pred.T) # Используем нашу утилиту
                    wandb.log({"val/spectrogram": wandb.Image(img, caption=f"Step {global_step}")}, step=global_step)
                    print("🖼️ Vis sent to WandB")
                model.train()
        
        scheduler.step()
        print(f"✅ Epoch {epoch} finished. Avg Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Сохраняем чекпоинты каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            save_path = f"{CONFIG['checkpoint_dir']}/checkpoint_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Модель сохранена: {save_path}")

    wandb.finish()

if __name__ == "__main__":
    train()
