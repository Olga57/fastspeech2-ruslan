# @title ⚙️ Шаг 2: Инициализация FastSpeech 2 и генерация аудио
import os
import sys
import json
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import IPython.display as ipd
import ipywidgets as widgets
from IPython.display import display, clear_output

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("⚠️ ВНИМАНИЕ: Вы используете CPU. Для работы вокодера NVIDIA требуется GPU.")

print("⚙️ [1/4] Инициализация архитектуры модели...")

# --- 1. АРХИТЕКТУРА (ОРИГИНАЛЬНАЯ, ИЗ ВАШЕГО ДЕМО) ---
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
        d_mult = torch.clamp(d_mult, min=2) # Минимум 2 фрейма на фонему

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


# --- 2. ЗАГРУЗКА ВЕСОВ И СЛОВАРЯ ---
print("📂 [2/4] Загрузка чекпоинта и словаря...")
ckpt_path = "fastspeech2-ruslan/FastSpeech2_Weight/checkpoint_49.pth"
vocab_path = "fastspeech2-ruslan/FastSpeech2_Weight/vocab.json"

if not os.path.exists(ckpt_path): ckpt_path = "FastSpeech2_Weight/checkpoint_49.pth"
if not os.path.exists(vocab_path): vocab_path = "FastSpeech2_Weight/vocab.json"

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
vocab_size = len(vocab) + 1

model = FastSpeech2(vocab_size=vocab_size, d_model=256).to(device)
state = torch.load(ckpt_path, map_location=device)

# ИСПРАВЛЕННАЯ ЗАГРУЗКА (без случайной перезаписи декодера)
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
model.load_state_dict(state, strict=False)
model.eval()

print("🔌 [3/4] Загрузка вокодера HiFi-GAN (NVIDIA)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vocoder_data = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan', trust_repo=True, verbose=False)
    vocoder = vocoder_data[0] if isinstance(vocoder_data, tuple) else vocoder_data
    vocoder = vocoder.to(device).eval()

# --- 3. ФОНЕТИЧЕСКАЯ ТРАНСКРИПЦИЯ (АДАПТЕР) ---
HARDCODED_DICT = {
    "текстура": ["tʲ", "e", "k", "s̪", "t̪", "u", "r", "a"],
    "поверхности": ["p", "a", "vʲ", "e", "r", "x", "n̪", "a", "sʲ", "tʲ", "i"],
    "определяется": ["a", "p", "rʲ", "i", "dʲ", "e", "ʎ", "a", "j", "i", "t̪s̪", "a"],
    "кристаллической": ["k", "rʲ", "i", "s̪", "t̪", "a", "ʎ", "i", "tɕ", "i", "s̪", "k", "a", "j"],
    "структурой": ["s̪", "t̪", "r", "u", "k", "t̪", "u", "r", "a", "j"],
    "материала": ["m", "a", "tʲ", "e", "rʲ", "i", "a", "ɫ", "a"],
    "синтез": ["sʲ", "i", "n̪", "tʲ", "e", "z̪"],
    "речи": ["rʲ", "e", "tɕ", "i"],
    "является": ["j", "i", "v", "ʎ", "a", "j", "i", "t̪s̪", "a"],
    "одной": ["a", "d̪", "n̪", "o", "j"],
    "из": ["i", "z̪"],
    "наиболее": ["n̪", "a", "i", "b", "o", "ʎ", "i", "j", "e"],
    "сложных": ["s̪", "ɫ", "o", "ʐ", "n̪", "ɨ", "x"],
    "задач": ["z̪", "a", "d̪", "a", "tɕ"],
    "обработки": ["a", "b", "r", "a", "b", "o", "t̪", "k", "i"],
    "естественного": ["j", "i", "sʲ", "tʲ", "e", "s̪", "tʲ", "vʲ", "i", "n̪", "a", "v", "a"],
    "языка": ["j", "i", "z̪", "ɨ", "k", "a"],
    "трансформер": ["t̪", "r", "a", "n̪", "s̪", "f", "o", "r", "mʲ", "i", "r"],
    "это": ["e", "t̪", "a"],
    "архитектура": ["a", "r", "x", "i", "tʲ", "i", "k", "t̪", "u", "r", "a"],
    "глубокого": ["ɡ", "ɫ", "u", "b", "o", "k", "a", "v", "a"],
    "обучения": ["a", "b", "u", "tɕ", "e", "ɲ", "i", "j", "a"],
    "основанная": ["a", "s̪", "n̪", "o", "v", "a", "n̪", "a", "j", "a"],
    "на": ["n̪", "a"],
    "механизме": ["mʲ", "i", "x", "a", "ɲ", "i", "z̪", "mʲ", "e"],
    "внимания": ["v", "ɲ", "i", "m", "a", "ɲ", "i", "j", "a"]
}

FALLBACK_MAP = {
    'а': ['a'], 'б': ['b'], 'в': ['v'], 'г': ['ɡ'], 'д': ['d̪'], 'е': ['e'], 'ё': ['o'],
    'ж': ['ʐ'], 'з': ['z̪'], 'и': ['i'], 'й': ['j'], 'к': ['k'], 'л': ['ɫ'], 'м': ['m'],
    'н': ['n̪'], 'о': ['o'], 'п': ['p'], 'р': ['r'], 'с': ['s̪'], 'т': ['t̪'], 'у': ['u'],
    'ф': ['f'], 'х': ['x'], 'ц': ['t̪s̪'], 'ч': ['tɕ'], 'ш': ['ʂ'], 'щ': ['ɕː'],
    'ъ': [], 'ы': ['ɨ'], 'ь': [], 'э': ['e'], 'ю': ['u'], 'я': ['a']
}

def synthesize_audio(text, speed=1.0, pitch=1.0, energy=1.0):
    text = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
    words = text.split()
    sequence = []
    
    for w in words:
        if w in HARDCODED_DICT:
            for p in HARDCODED_DICT[w]:
                sequence.append(vocab.get(p, vocab.get("<UNK>", 87)))
        else:
            for char in w:
                phones = FALLBACK_MAP.get(char, [])
                for p in phones:
                    sequence.append(vocab.get(p, vocab.get("<UNK>", 87)))
        
        if "<SIL>" in vocab:
            sequence.append(vocab["<SIL>"])
            
    if not sequence:
        return None
        
    src = torch.tensor(sequence).long().unsqueeze(0).to(device)

    with torch.no_grad():
        mel_postnet = model(src, d_c=speed, p_c=pitch, e_c=energy)
        mel_vocoder = mel_postnet.transpose(1, 2)
        audio = vocoder(mel_vocoder)
        return audio.squeeze().cpu().numpy()

# --- 4. ИНТЕРФЕЙС УПРАВЛЕНИЯ ---
print("\n🎧 [4/4] ИНТЕРФЕЙС ГЕНЕРАЦИИ ГОТОВ\n")

phrases = [
    "Текстура поверхности определяется кристаллической структурой материала",
    "Синтез речи является одной из наиболее сложных задач обработки естественного языка",
    "Трансформер это архитектура глубокого обучения основанная на механизме внимания",
    "--- ВВЕСТИ СВОЙ ТЕКСТ ---"
]

dropdown = widgets.Dropdown(options=phrases, value=phrases[0], description='Фраза:', layout=widgets.Layout(width='90%'))
text_input = widgets.Text(value='', placeholder='Введите текст...', description='Свой текст:', layout=widgets.Layout(width='90%'), disabled=True)

speed_slider = widgets.FloatSlider(value=1.0, min=0.5, max=2.0, step=0.1, description='Скорость:')
pitch_slider = widgets.FloatSlider(value=1.0, min=0.5, max=2.0, step=0.1, description='Тон:')
energy_slider = widgets.FloatSlider(value=1.0, min=0.5, max=2.0, step=0.1, description='Громкость:')

button = widgets.Button(description='Синтезировать 🎙️', button_style='success')
output_audio = widgets.Output()

def on_dropdown_change(change):
    text_input.disabled = (change['new'] != phrases[3])
dropdown.observe(on_dropdown_change, names='value')

def on_button_clicked(b):
    with output_audio:
        clear_output(wait=True)
        text = text_input.value.strip() if dropdown.value == phrases[3] else dropdown.value
        if not text:
            print(" Введите текст!")
            return
            
        print(f"⏳ Генерирую речь...")
        try:
            audio_wav = synthesize_audio(text, speed=speed_slider.value, pitch=pitch_slider.value, energy=energy_slider.value)
            
            if audio_wav is not None:
                display(ipd.Audio(audio_wav, rate=22050, autoplay=True)) 
            else:
                print(" Ошибка генерации.")
        except Exception as e:
            print(f" Ошибка: {e}")

button.on_click(on_button_clicked)

ui_box = widgets.VBox([dropdown, text_input, widgets.HBox([speed_slider, pitch_slider, energy_slider]), button])
display(ui_box)
display(output_audio)
