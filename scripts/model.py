import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def subsequent_mask(seq_len:int, device=None):
    """
    Создаёт маску “subsequent” для decoder‑части трансформера
    
    Маска имеет форму [seq_len, seq_len] и содержит `-inf` в верхнем треугольнике
    (включая диагональ), чтобы запрещать обращение к будущим токенам

    Args:
        seq_len (int): длина целевой последовательности
        device: устройство для тензора. Если None – используется CPU

    Returns:
        torch.Tensor: float‑маска с -inf в закрытых позициях и 0 в открытых
    """
    attn_shape = (seq_len, seq_len)
    mask_bool = torch.triu(torch.ones(attn_shape, device=device), diagonal=1).bool()
    mask_float = torch.full(attn_shape, float('-inf'), device=device)
    mask_float[~mask_bool] = 0.0
    return mask_float


class LearnablePositionalEncoding(nn.Module):
    """
    Позиционное кодирование с обучаемыми эмбеддингами

    Вместо фиксированных синусоидальных векторов используется
    learnable embedding, который добавляется к токен‑эмбеддингам
    """

    def __init__(self, embed_dim:int, max_len:int, dropout:float, batch_first:bool, padding_idx:int):
        """
        Args:
            embed_dim (int): размерность эмбеддингов
            max_len (int): максимальная длина последовательности
            dropout (float): коэффициент dropout для позиционного кода
            batch_first (bool): если True – входы в виде [B, T, D]
            padding_idx (int): индекс токена‑паддинга (не учитывается)
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(max_len, embed_dim, padding_idx=padding_idx)
        self._max_len = max_len
        self._batch_first = batch_first

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Добавляет позиционное кодирование к входу
        Args:
            x (torch.Tensor): [seq_len, batch, d_model] или [batch, seq_len, d_model]
        Returns:
            torch.Tensor: с добавленным позиционным вектором и dropout
        """
        seq_len = x.size(1) if self._batch_first else x.size(0)
        if seq_len > self._max_len:
            raise ValueError(
                f'Длина последовательности {seq_len} превышает максимальную длину позиционного кодирования {self._max_len}'
            )
        pos_indices = torch.arange(seq_len, device=x.device).unsqueeze(1)
        pos_embeddings = self.pos_embedding(pos_indices)  # (seq_len, 1, d_model)
        if self._batch_first:
            pos_embeddings = pos_embeddings.permute(1, 0, 2)  # [B,T,D]
        return x + pos_embeddings


class TransformerModel(nn.Module):
    """
    Seq2Seq трансформер для генерации текста

    Поддерживает задачу перевода/генерации с использованием
    learnable positional encodings и возможностью задания температуры
    """

    def __init__(self,
                 source_vocab_size:int, target_vocab_size:int, embed_dim:int, d_model:int, nhead:int, num_encoder_layers:int,
                 num_decoder_layers:int, dim_fc_hidden:int, dropout:int=0.1, max_len=1000, batch_first=True, padding_idx=0):
        """
        Args:
            source_vocab_size (int): размер словаря источника
            target_vocab_size (int): размер словаря целевого языка
            embed_dim (int): размерность токен‑эмбеддингов
            d_model (int): глубина модели трансформера
            nhead (int): число голов в multi‑head attention
            num_encoder_layers (int): количество слоёв encoder‑части
            num_decoder_layers (int): количество слоёв decoder‑части
            dim_fc_hidden (int): размер скрытого слоя feedforward
            dropout (float): коэффициент dropout
            max_len (int): максимальная длина последовательности
            batch_first (bool): формат входов [B,T,D] или [T,B,D]
            padding_idx (int): индекс токена‑паддинга
        """
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        # Встроенные эмбеддинги для source и target
        self.source_embedding = nn.Embedding(source_vocab_size, embed_dim, padding_idx=padding_idx)
        self.target_embedding = nn.Embedding(target_vocab_size, embed_dim, padding_idx=padding_idx)

        # Обучаемые позиционное кодирование
        self.pos_encoding_encoder = LearnablePositionalEncoding(embed_dim, max_len=max_len,
            dropout=dropout, batch_first=batch_first, padding_idx=padding_idx)
        self.pos_encoding_decoder = LearnablePositionalEncoding(embed_dim, max_len=max_len,
            dropout=dropout, batch_first=batch_first, padding_idx=padding_idx)

        # Перевод из embed_dim в d_model
        self.embed_to_model_projection = nn.Linear(embed_dim, d_model)

        # Ядро трансформера
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,\
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_fc_hidden, dropout=dropout, batch_first=batch_first)

        # Финальный классификатор
        self.classifier = nn.Linear(d_model, target_vocab_size)

    def forward(self, source: torch.Tensor, target: torch.Tensor,
                temperature: float, apply_softmax: bool = True):
        """
        Прямой ход модели
        Args:
            source (torch.Tensor): [B,T] – индексы исходных токенов
            target (torch.Tensor): [B,T] – индексы целевых токенов (с BOS)
            temperature (float): температура для softmax
            apply_softmax (bool): если True, применяем softmax к logits
        Returns:
            torch.Tensor: вероятности/логиты размера [B, tgt_len, vocab]
        """
        device = source.device

        # 1. Эмбеддинги и масштабирование
        source_embed = self.source_embedding(source) * math.sqrt(self.embed_dim)
        target_embed = self.target_embedding(target) * math.sqrt(self.embed_dim)

        # 2. Добавляем позиционное кодирование
        source_embed = self.pos_encoding_encoder(source_embed)
        target_embed = self.pos_encoding_decoder(target_embed)

        # 3. Проекция в пространство d_model
        source_proj = self.embed_to_model_projection(source_embed)
        target_proj = self.embed_to_model_projection(target_embed)

        # 4. Маски для паддингов
        src_key_padding_mask = (source == self.padding_idx).to(device)   # [B, src_len]
        tgt_key_padding_mask = (target == self.padding_idx).to(device)   # [B, tgt_len]

        # Маска subsequent для decoder‑части
        tgt_seq_len = target.size(1)
        tgt_mask = subsequent_mask(tgt_seq_len, device=target.device)

        # 5. Encoder + Decoder
        memory = self.transformer.encoder(
            source_proj,
            src_key_padding_mask=src_key_padding_mask
        )
        output = self.transformer.decoder(
            target_proj,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )  # [B, tgt_len, d_model] (batch_first=True)

        # 6. Классификация
        logits = self.classifier(output)  # [B, tgt_len, vocab]

        if apply_softmax:
            logits = F.softmax(logits / temperature, dim=2)

        return logits
