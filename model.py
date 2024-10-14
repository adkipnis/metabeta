from typing import Union, Tuple
import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    # Input embedding layer (words to vocab indices to real vectors)
    # d_model: the number of features in the input embedding
    # vocab_size: the number of words in the vocabulary
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.linear(x)
        return self.activation(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # learnable scale
        self.beta = nn.Parameter(torch.zeros(1)) # learnable bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.w_query = nn.Linear(d_model, d_model)
        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor,
                  key: torch.Tensor,
                  value: torch.Tensor,
                  mask: Union[None, torch.Tensor],
                  dropout: Union[None, nn.Dropout],
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # wherever the mask is 0, the attention score is set to -1e9 (negative infinity)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        # project the query, key, and value
        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)
        # split the query, key, and value into num_heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        # this is transposed to (batch_size, num_heads, seq_len, d_k)
        # as each head should see (seq_len, d_k), the same sentence but different embedding features
        query = query.view(query.size(0), query.size(1), self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.num_heads, self.d_k).transpose(1, 2)
        # apply the attention mechanism
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)
        return self.w_out(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 n_heads: int,
                 dropout: float) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # we need a mask to prevent interaction between the padding word with the rest of the words
        attn = lambda x: self.self_attention(x, x, x, mask)
        x = self.residual_connections[0](x, attn)
        return self.residual_connections[1](x, self.feed_forward)


class DecoderBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 n_heads: int,
                 dropout: float) -> None:
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        slf_attn = lambda x: self.self_attention(x, x, x, tgt_mask)
        cross_attn = lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.residual_connections[0](x, slf_attn)
        x = self.residual_connections[1](x, cross_attn)
        return self.residual_connections[2](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, n_blocks: int, dropout: float) -> None:
        super(Encoder, self).__init__()
        n_blocks = n_blocks
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, n_heads, dropout) for _ in range(n_blocks)])
        self.norm = LayerNormalization()

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, n_blocks: int, dropout: float) -> None:
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_ff, n_heads, dropout) for _ in range(n_blocks)])
        self.norm = LayerNormalization()

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    # Projection layer (real vectors to vocab indices)
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 n_heads: int,
                 n_blocks_e: int,
                 n_blocks_d: int,
                 vocab_size: int,
                 dropout: float,
                 ) -> None:
        super(Transformer, self).__init__()
        self.embed = InputEmbedding(d_model, vocab_size)
        self.encoder = Encoder(d_model, d_ff, n_heads, n_blocks_e, dropout)
        self.decoder = Decoder(d_model, d_ff, n_heads, n_blocks_d, dropout)
        self.projection = ProjectionLayer(d_model, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.embed(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt = self.embed(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


