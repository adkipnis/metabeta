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
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # Positional encoding adds information about the position of tokens in the sequence
    # d_model: the number of features in the positional encoding
    # seq_len: the maximum length of the input sequence
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply the sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # add a batch dimension
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # register the buffer so that it is saved in the state_dict
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        with torch.no_grad():
            x = x + self.pe[:, :seq_len]
        return self.dropout(x)


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
                 self_attention: MultiHeadAttention,
                 feed_forward: FeedForwardBlock,
                 dropout: float) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # print("Encoding...")
        # we need a mask to prevent interaction between the padding word with the rest of the words
        attn = lambda x: self.self_attention(x, x, x, mask)
        ff = lambda x: self.feed_forward(x)
        x = self.residual_connections[0](x, attn)
        return self.residual_connections[1](x, ff)


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention: MultiHeadAttention,
                 cross_attention: MultiHeadAttention,
                 feed_forward: FeedForwardBlock,
                 dropout: float) -> None:
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        # print("Decoding...")
        slf_attn = lambda x: self.self_attention(x, x, x, tgt_mask)
        cross_attn = lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)
        ff = lambda x: self.feed_forward(x)
        x = self.residual_connections[0](x, slf_attn)
        x = self.residual_connections[1](x, cross_attn)
        return self.residual_connections[2](x, ff)


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super(Decoder, self).__init__()
        self.layers = layers
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
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbedding,
                 tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection: ProjectionLayer) -> None:
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


def buildTransformer(src_vocab_size: int,
                     tgt_vocab_size: int,
                     src_seq_len: int,
                     tgt_seq_len: int,
                     d_model: int = 512, # embedding dimension
                     n_blocks: int = 6, # number of encoder and decoder blocks
                     n_heads: int = 8, # number of attention heads
                     dropout: float = 0.1,
                     d_ff: int = 2048, # feed forward dimension
                     ) -> Transformer:
    # input embeddings
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # encoder blocks
    encoder_blocks = []
    for _ in range(n_blocks):
        encoder_self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        encoder_ff = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, encoder_ff, dropout)
        encoder_blocks += [encoder_block]

    # decoder blocks
    decoder_blocks = []
    for _ in range(n_blocks):
        decoder_self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        decoder_ff = FeedForwardBlock(d_model, d_ff, dropout)  
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, decoder_ff, dropout)
        decoder_blocks += [decoder_block]

    # build the encoder, decoder, and projection layer
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # build the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters with xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

