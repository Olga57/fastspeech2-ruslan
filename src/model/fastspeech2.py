import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def get_non_pad_mask(seq, pad_idx=0):
    assert seq.dim() == 2
    return seq.ne(pad_idx).unsqueeze(-1).float()

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx=0):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)
    return padding_mask.unsqueeze(1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=10000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.softmax(attn)
        return torch.matmul(attn, v), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).transpose(1, 2)

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=kernel_size[0], padding=(kernel_size[0]-1)//2)
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=kernel_size[1], padding=(kernel_size[1]-1)//2)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_1(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.w_2(output)
        output = self.dropout(output)
        output = output.transpose(1, 2)
        return output

class FFTBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, fft_kernel_size, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, fft_kernel_size, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, enc_input, mask=None):
        output, attn = self.slf_attn(enc_input, enc_input, enc_input, mask=mask)
        output = self.layer_norm_1(enc_input + output)
        output_ffn = self.pos_ffn(output)
        output = self.layer_norm_2(output + output_ffn)
        return output, attn

class Encoder(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = config["transformer"]["encoder_hidden"] // n_head
        d_v = config["transformer"]["encoder_hidden"] // n_head
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["fft_conv1d_filter_size"]
        kernel_size = config["transformer"]["fft_conv1d_kernel_size"]
        dropout = config["transformer"]["dropout"]

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layer_stack = nn.ModuleList([
            FFTBlock(d_model, d_inner, n_head, d_k, d_v, kernel_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src_seq, mask):
        x = self.embedding(src_seq)
        x = self.pos_encoding(x)
        enc_slf_attn_list = []
        for enc_layer in self.layer_stack:
            x, attn = enc_layer(x, mask=mask)
            enc_slf_attn_list.append(attn)
        return x, enc_slf_attn_list

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = config["transformer"]["decoder_hidden"] // n_head
        d_v = config["transformer"]["decoder_hidden"] // n_head
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["fft_conv1d_filter_size"]
        kernel_size = config["transformer"]["fft_conv1d_kernel_size"]
        dropout = config["transformer"]["dropout"]

        self.pos_encoding = PositionalEncoding(d_model)
        self.layer_stack = nn.ModuleList([
            FFTBlock(d_model, d_inner, n_head, d_k, d_v, kernel_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, enc_seq, mask):
        x = self.pos_encoding(enc_seq)
        dec_slf_attn_list = []
        for dec_layer in self.layer_stack:
            x, attn = dec_layer(x, mask=mask)
            dec_slf_attn_list.append(attn)
        return x, dec_slf_attn_list

class VariancePredictor(nn.Module):
    def __init__(self, d_model, filter_size, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, filter_size, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.ln1 = nn.LayerNorm(filter_size)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.ln2 = nn.LayerNorm(filter_size)
        self.dropout2 = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(filter_size, 1)

    def forward(self, x, mask=None):
        out = x.transpose(1, 2)
        out = F.relu(self.conv1(out))
        out = self.dropout1(self.ln1(out.transpose(1, 2)))
        out = F.relu(self.conv2(out.transpose(1, 2)))
        out = self.dropout2(self.ln2(out.transpose(1, 2)))
        out = self.linear_layer(out).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out

class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, durations, mel_max_length=None):
        output = []
        for batch_i in range(x.size(0)):
            content = x[batch_i]
            dur = durations[batch_i]
            expanded = torch.repeat_interleave(content, dur, dim=0)
            output.append(expanded)
        output = pad_sequence(output, batch_first=True, padding_value=0.0)
        if mel_max_length is not None:
            if output.size(1) < mel_max_length:
                pad_size = mel_max_length - output.size(1)
                output = F.pad(output, (0, 0, 0, pad_size))
            elif output.size(1) > mel_max_length:
                output = output[:, :mel_max_length, :]
        return output

class VarianceAdaptor(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config["transformer"]["encoder_hidden"]
        self.duration_predictor = VariancePredictor(d_model, 256, 3, 0.1)
        self.pitch_predictor = VariancePredictor(d_model, 256, 3, 0.1)
        self.energy_predictor = VariancePredictor(d_model, 256, 3, 0.1)
        self.length_regulator = LengthRegulator()
        self.pitch_embedding = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.energy_embedding = nn.Conv1d(1, d_model, kernel_size=3, padding=1)

    def forward(self, x, src_mask, mel_mask=None, d_target=None, p_target=None, e_target=None, max_len=None, d_control=1.0, p_control=1.0, e_control=1.0):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        
        if d_target is not None:
            duration_rounded = d_target
        else:
            duration_rounded = torch.clamp(torch.round((torch.exp(log_duration_prediction) - 1) * d_control), min=0).long()

        x_adapted = self.length_regulator(x, duration_rounded, max_len)

        pitch_prediction = self.pitch_predictor(x_adapted, mel_mask)
        pitch_to_embed = p_target if p_target is not None else pitch_prediction * p_control
        x_adapted = x_adapted + self.pitch_embedding(pitch_to_embed.unsqueeze(1)).transpose(1, 2)

        energy_prediction = self.energy_predictor(x_adapted, mel_mask)
        energy_to_embed = e_target if e_target is not None else energy_prediction * e_control
        x_adapted = x_adapted + self.energy_embedding(energy_to_embed.unsqueeze(1)).transpose(1, 2)

        return x_adapted, log_duration_prediction, pitch_prediction, energy_prediction, duration_rounded

class PostNet(nn.Module):
    def __init__(self, n_mel_channels=80, postnet_embedding_dim=512, postnet_kernel_size=5, postnet_n_convolutions=5):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(n_mel_channels, postnet_embedding_dim, kernel_size=postnet_kernel_size, stride=1, padding=(postnet_kernel_size - 1) // 2),
                nn.BatchNorm1d(postnet_embedding_dim))
        )
        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(postnet_embedding_dim, postnet_embedding_dim, kernel_size=postnet_kernel_size, stride=1, padding=(postnet_kernel_size - 1) // 2),
                    nn.BatchNorm1d(postnet_embedding_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5))
            )
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(postnet_embedding_dim, n_mel_channels, kernel_size=postnet_kernel_size, stride=1, padding=(postnet_kernel_size - 1) // 2),
                nn.BatchNorm1d(n_mel_channels))
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        x = self.convolutions[-1](x)
        return x.transpose(1, 2)

class FastSpeech2(nn.Module):
    def __init__(self, config, vocab_size):
        super(FastSpeech2, self).__init__()
        self.encoder = Encoder(config, vocab_size)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = Decoder(config)
        self.mel_linear = nn.Linear(config["transformer"]["decoder_hidden"], config["audio"]["n_mels"])
        self.postnet = PostNet(n_mel_channels=config["audio"]["n_mels"])

    def forward(self, src_seq, src_lens, mel_target=None, mel_lens=None, d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None, d_control=1.0, p_control=1.0, e_control=1.0):
        src_attn_mask = get_attn_key_pad_mask(src_seq, src_seq)
        src_non_pad_mask = get_non_pad_mask(src_seq)
        src_predictor_mask = src_seq.eq(0)

        enc_output, _ = self.encoder(src_seq, src_attn_mask)
        enc_output = enc_output * src_non_pad_mask

        if mel_lens is not None:
            mel_mask_tensor = torch.arange(mel_target.size(1), device=src_seq.device)[None, :] < mel_lens[:, None]
            mel_mask = ~mel_mask_tensor
            mel_non_pad_mask = mel_mask_tensor.unsqueeze(-1).float()
        else:
            mel_mask = None
            mel_non_pad_mask = None

        variance_output, log_duration_prediction, pitch_prediction, energy_prediction, durations_rounded = self.variance_adaptor(
            enc_output, src_mask=src_predictor_mask, mel_mask=mel_mask,
            d_target=d_target, p_target=p_target, e_target=e_target,
            max_len=mel_target.size(1) if mel_target is not None else None,
            d_control=d_control, p_control=p_control, e_control=e_control
        )

        if mel_target is None:
            dec_mask = None
            dec_non_pad_mask = 1.0
        else:
            dec_mask = mel_mask.unsqueeze(1).expand(-1, variance_output.size(1), -1).unsqueeze(1)
            dec_non_pad_mask = mel_non_pad_mask

        dec_output, _ = self.decoder(variance_output, dec_mask)
        mel_output = self.mel_linear(dec_output)
        mel_output_postnet = mel_output + self.postnet(mel_output)

        if mel_non_pad_mask is not None:
             mel_output = mel_output * mel_non_pad_mask
             mel_output_postnet = mel_output_postnet * mel_non_pad_mask

        if mel_target is None:
            return {"mel_postnet": mel_output_postnet}

        return {
            "mel_output": mel_output,
            "mel_postnet": mel_output_postnet,
            "log_duration": log_duration_prediction,
            "pitch": pitch_prediction,
            "energy": energy_prediction,
            "src_mask": src_attn_mask,
            "mel_mask": mel_mask,
            "dec_output": dec_output
        }

class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, inputs, targets):
        mel_target = targets["mel"]
        mel_loss = self.l1_loss(inputs["mel_output"], mel_target)
        postnet_mel_loss = self.l1_loss(inputs["mel_postnet"], mel_target)

        dur_target = torch.log(targets["duration"].float() + 1)
        dur_loss = self.mse_loss(inputs["log_duration"], dur_target)

        pitch_loss = self.mse_loss(inputs["pitch"], targets["pitch"])
        energy_loss = self.mse_loss(inputs["energy"], targets["energy"])

        total_loss = mel_loss + postnet_mel_loss + dur_loss + pitch_loss + energy_loss
        return total_loss, {
            "mel": mel_loss.item(),
            "postnet": postnet_mel_loss.item(),
            "dur": dur_loss.item(),
            "pitch": pitch_loss.item(),
            "energy": energy_loss.item()
        }
