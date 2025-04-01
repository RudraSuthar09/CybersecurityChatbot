import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from datetime import datetime

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
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).view(b, c)
        y = self.fc(y).view(b, c)
        return x * y

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, N = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, N // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (N ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N)
        x = self.proj(x)
        return self.norm(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
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
        self.logger.info(f"Current Date and Time (UTC): 2025-03-31 07:22:04")
        self.logger.info(f"Current User: RudraSuthar09")
        
        # Model parameters
        self.input_size = config['model']['network_analyzer']['input_size']
        self.hidden_size = config['model']['network_analyzer']['hidden_size']
        self.num_layers = config['model']['network_analyzer']['num_layers']
        self.dropout = config['model']['network_analyzer']['dropout']
        self.num_classes = config['data']['nslkdd']['classes']
        
        # Initial feature extraction
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.hidden_size)
        )
        
        # Multi-head self attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(self.hidden_size) 
            for _ in range(2)
        ])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.hidden_size)
            for _ in range(self.num_layers)
        ])
        
        # SE block for feature refinement
        self.se = SEBlock(self.hidden_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, self.num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        
        # Apply attention layers
        for attention in self.attention_layers:
            residual = x
            x = attention(x)
            x = x + residual
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final feature refinement
        x = self.se(x)
        
        # Classification
        x = self.classifier(x)
        return x