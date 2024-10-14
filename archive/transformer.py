import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_size: int = 512,
                 n_head: int = 8,
                 n_layers: int = 6,
                 ):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.n_head = n_head
        self.n_layers = n_layers

        # define the model
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=n_head,
                                          num_encoder_layers=n_layers,
                                          num_decoder_layers=n_layers)
        self.fc_out = nn.Linear(emb_size, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        
        # Generate masks
        src_mask = self.transformer.generate_square_subsequent_mask(len(src)).to(src.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(len(tgt)).to(tgt.device)
        
        # Forward through transformer
        output = self.transformer(src_emb.transpose(0, 1),
                                  tgt_emb.transpose(0, 1),
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask)
        return self.fc_out(output)

# # Instantiate the model
# class Transformer(torch.nn.Module):
#     def __init__(self,
#                  vocab_size: int,
#                  embed_size: int = 128,
#                  ff_size: int = 64,
#                  n_heads: int = 2,
#                  n_layers: int = 2,
#                  ):
#         super(Transformer, self).__init__()
#         self.vocab_size = vocab_size
#         self.embed_size = embed_size
#         self.ff_size = ff_size
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.embed = torch.nn.Embedding(self.vocab_size, self.embed_size)
#         self.encoder = torch.nn.TransformerEncoder(
#                 encoder_layer=torch.nn.TransformerEncoderLayer(
#                     d_model=self.embed_size,
#                     nhead=self.n_heads,
#                     dim_feedforward=self.ff_size,
#                     dropout=0.1,
#                     activation='relu'
#                 ),
#                 num_layers=self.n_layers
#                 )
#         self.decoder = torch.nn.TransformerDecoder(
#                 decoder_layer=torch.nn.TransformerDecoderLayer(
#                     d_model=self.embed_size,
#                     nhead=self.n_heads,
#                     dim_feedforward=self.ff_size,
#                     dropout=0.1,
#                     activation='relu'
#                 ),
#                 num_layers=self.n_layers
#                 )
#         self.fc = torch.nn.Linear(self.embed_size, self.vocab_size)
#         self.softmax = torch.nn.Softmax(dim=-1)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.embed(x)
#         x = self.encoder(x)
#         x = self.decoder(x)
#         x = self.fc(x)
#         return self.softmax(x)

