import argparse
import json
import os
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_deps():
    # На всякий случай (как у тебя в ноуте): переустановка совместимых версий
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speechbrain", "huggingface_hub==0.19.0"], check=False)


def patch_torchaudio():
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    return torchaudio


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2)
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=kernel_size[1], padding=(kernel_size[1] - 1) // 2)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.w_2(F.relu(self.w_1(x)))
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_k ** -0.5

    def forward(self, q, k, v, mask=None):
        residual_q = q
        b_sz, len_q, _ = q.size()
        b_sz, len_k, _ = k.size()
        b_sz, len_v, _ = v.size()

        q = self.w_qs(q).view(b_sz, len_q, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)
        k = self.w_ks(k).view(b_sz, len_k, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k)
        v = self.w_vs(v).view(b_sz, len_v, self.n_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = F.softmax(attn, dim=2)
        out = torch.bmm(attn, v).view(self.n_head, b_sz, len_q, self.d_v).permute(1, 2, 0, 3).contiguous().view(b_sz, len_q, -1)
        out = self.dropout(self.fc(out))
        return self.layer_norm(out + residual_q), attn


class FFTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, kernel_size, dropout)

    def forward(self, x, mask=None):
        x, _ = self.slf_attn(x, x, x, mask)
        x = self.pos_ffn(x)
        return x, None


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_head, d_k, d_v, d_inner, kernel_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.layer_stack = nn.ModuleList(
            [FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout) for _ in range(n_layers)]
        )

    def forward(self, src_seq):
        x = self.embedding(src_seq)
        for layer in self.layer_stack:
            x, _ = layer(x)
        return x


class VariancePredictor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.ReLU(), nn.LayerNorm(d_model), nn.Dropout(0.1),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.ReLU(), nn.LayerNorm(d_model), nn.Dropout(0.1),
        )
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        o = x.transpose(1, 2)
        for layer in self.conv:
            if isinstance(layer, nn.LayerNorm):
                o = layer(o.transpose(1, 2)).transpose(1, 2)
            else:
                o = layer(o)
        return self.linear(o.transpose(1, 2)).squeeze(-1)


class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, 4, 2, 128, 128, 1024, [9, 1], 0.1)
        self.decoder = Encoder(vocab_size, d_model, 4, 2, 128, 128, 1024, [9, 1], 0.1)
        self.mel_linear = nn.Linear(d_model, 80)
        self.duration_predictor = VariancePredictor(d_model)
        self.pitch_predictor = VariancePredictor(d_model)
        self.energy_predictor = VariancePredictor(d_model)
        self.pitch_embed = nn.Conv1d(1, d_model, 3, padding=1)
        self.energy_embed = nn.Conv1d(1, d_model, 3, padding=1)
        self.postnet = nn.Sequential(
            nn.Conv1d(80, 512, 5, padding=2), nn.Tanh(),
            nn.Conv1d(512, 512, 5, padding=2), nn.Tanh(),
            nn.Conv1d(512, 80, 5, padding=2),
        )

    def forward(self, src_seq, p_c=1.0, e_c=1.0, d_c=1.0):
        enc_out = self.encoder(src_seq)

        log_dur = self.duration_predictor(enc_out)
        dur = torch.clamp(torch.exp(log_dur) - 1, min=0)

        d_mult = torch.round(dur * (1.0 / d_c)).int()
        d_mult = torch.clamp(d_mult, min=5)

        output = []
        for i in range(enc_out.size(0)):
            output.append(torch.repeat_interleave(enc_out[i], d_mult[i], dim=0))
        x_adapt = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)

        pitch = self.pitch_predictor(enc_out)
        energy = self.energy_predictor(enc_out)

        p_ex_list, e_ex_list = [], []
        for i in range(pitch.size(0)):
            p_ex_list.append(torch.repeat_interleave(pitch[i], d_mult[i], dim=0))
            e_ex_list.append(torch.repeat_interleave(energy[i], d_mult[i], dim=0))

        p_ex = torch.nn.utils.rnn.pad_sequence(p_ex_list, batch_first=True).unsqueeze(-1)
        e_ex = torch.nn.utils.rnn.pad_sequence(e_ex_list, batch_first=True).unsqueeze(-1)

        x_adapt = (
            x_adapt
            + self.pitch_embed((p_ex * p_c).transpose(1, 2)).transpose(1, 2)
            + self.energy_embed((e_ex * e_c).transpose(1, 2)).transpose(1, 2)
        )

        dec_out = x_adapt
        for layer in self.decoder.layer_stack:
            dec_out, _ = layer(dec_out)

        mel = self.mel_linear(dec_out)
        mel_t = mel.transpose(1, 2)
        refined = mel_t + self.postnet(mel_t)
        return refined


def build_text_seq(text, vocab):
    # Если vocab не кириллический — транслит как у тебя
    if "а" not in vocab:
        m = {
            "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh", "з": "z",
            "и": "i", "й": "j", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
            "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh",
            "щ": "sch", "ы": "y", "э": "e", "ю": "yu", "я": "ya", " ": " "
        }
        text = "".join([m.get(c, c) for c in text.lower()])

    seq = [vocab[c] for c in text.lower() if c in vocab]
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Текстура поверхности определяется структурой")
    parser.add_argument("--speed", type=float, default=1.0, help="d_c in your code; lower -> slower, higher -> faster")
    parser.add_argument("--ckpt", type=str, default="fastspeech2-ruslan/FastSpeech2_Weight/checkpoint_49.pth")
    parser.add_argument("--vocab", type=str, default="fastspeech2-ruslan/FastSpeech2_Weight/vocab.json")
    parser.add_argument("--out", type=str, default="demo_output.wav")
    args = parser.parse_args()

    ensure_deps()
    torchaudio = patch_torchaudio()

    from speechbrain.inference.vocoders import HIFIGAN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"Vocab not found: {args.vocab}")

    with open(args.vocab, encoding="utf-8") as f:
        vocab = json.load(f)

    model = FastSpeech2(len(vocab) + 1, d_model=256).to(device)

    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    new_state = {}
    for k, v in state.items():
        new_state[k] = v
        if k.startswith("encoder."):
            new_state[k.replace("encoder.", "decoder.")] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()

    hifi = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech",
        savedir="tmp_vocoder",
        run_opts={"device": device},
    )

    seq = build_text_seq(args.text, vocab)
    if not seq:
        raise ValueError("Empty token sequence after vocab filtering. Check vocab.json and input text.")

    src = torch.tensor(seq).long().unsqueeze(0).to(device)

    with torch.no_grad():
        mel = model(src, d_c=args.speed)
        wav = hifi.decode_batch(mel)  # expected tensor
        wav = wav.detach().cpu()

    # Приводим к (1, T)
    if wav.dim() == 3:
        # (B, 1, T) or (B, T, 1) - попробуем самые частые варианты
        if wav.shape[1] == 1:
            wav = wav[0]
        elif wav.shape[2] == 1:
            wav = wav[0].transpose(0, 1)
        else:
            wav = wav[0]
    elif wav.dim() == 2:
        wav = wav[0].unsqueeze(0)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)

    wav = torch.clamp(wav, -1.0, 1.0)
    torchaudio.save(args.out, wav, 22050)
    print(f"Saved: {args.out}")

    # Если запускаешь внутри ноутбука — попробуем проиграть
    try:
        import IPython.display as ipd
        display(ipd.Audio(wav.squeeze(0).numpy(), rate=22050))
    except Exception:
        pass


if __name__ == "__main__":
    main()
