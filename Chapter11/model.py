import torch
import math
from torch import nn, Tensor

from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=200):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2).float() * math.log(100) / emb_size)
        pos = torch.arange(0, maxlen).float().reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0),:])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, num_encoder_layers=4, emb_size=512, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.src_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=8, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.generator = nn.Linear(emb_size, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src, src_mask, src_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        logit = self.generator(memory)
        return memory, logit
