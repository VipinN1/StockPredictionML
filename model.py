import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class StockTransformer(nn.Module):
    def __init__(self,
                 feature_dim,        # number of input channels (e.g. 3)
                 d_model=64,         # model dimension
                 nhead=4,            # attention heads
                 num_layers=3,       # encoder layers
                 dim_feedforward=256,
                 dropout=0.1,
                 seq_len=30):        # window length
        super().__init__()
        # 1) input embedding
        self.input_embed = nn.Linear(feature_dim, d_model)
        # 2) positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        # 3) encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 4) regression head on the last time step
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        returns: (batch, 1)
        """
        x = self.input_embed(x)          # → (batch, seq_len, d_model)
        x = self.pos_encoder(x)          # add positional info
        x = x.transpose(0,1)             # → (seq_len, batch, d_model)
        out = self.transformer(x)        # → (seq_len, batch, d_model)
        out = out[-1]                    # last time step → (batch, d_model)
        out = self.dropout(out)
        return self.regressor(out)       # → (batch, 1)
