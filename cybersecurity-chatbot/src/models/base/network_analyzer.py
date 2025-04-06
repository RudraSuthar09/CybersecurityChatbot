import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape
        y = self.avg_pool(x.unsqueeze(-1)).view(b, c)
        y = self.fc(y).view(b, c)
        return x * y


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by number of heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

        self.attention = nn.MultiheadAttention(embed_dim=self.embed_size, num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention_output, _ = self.attention(query, key, value, attn_mask=mask)
        return self.dropout(attention_output)  # Apply dropout after attention


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
        self.se = SEBlock(dim)

    def forward(self, x):
        residual = x
        x = self.block(x)
        x = self.se(x)
        x = self.norm(x + residual)
        return x


class NetworkAnalyzer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Enhanced Network Analyzer Model")

        self.input_size = config['model']['network_analyzer']['input_size']
        self.hidden_size = config['model']['network_analyzer']['hidden_size']
        self.num_layers = config['model']['network_analyzer']['num_layers']
        self.dropout = config['model']['network_analyzer']['dropout']
        self.num_classes = config['data']['nslkdd']['classes']

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.hidden_size)
        )

        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(self.hidden_size, heads=8, dropout=self.dropout) 
            for _ in range(3)  # Increased depth
        ])

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.hidden_size, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.se = SEBlock(self.hidden_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_size // 2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_proj(x)

        for attention in self.attention_layers:
            residual = x
            x = attention(x, x, x)  # Ensure query, key, and value are passed correctly
            x = x + residual  # Residual connection

        for block in self.residual_blocks:
            x = block(x)

        x = self.se(x)

        x = self.classifier(x)
        return x
