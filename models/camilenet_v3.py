import torch
from torch import nn
from torch.nn import functional as F

def maml_init_(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        torch.nn.init.constant_(module.weight, 1.0)
        torch.nn.init.constant_(module.bias, 0.0)

def conv_block(in_channels, out_channels):
    """
    returns a block conv-bn-relu-pool
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, hidden_size=64, embedding_size=64):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            conv_block(input_channels, hidden_size),
            conv_block(hidden_size, hidden_size),
            conv_block(hidden_size, hidden_size),
            conv_block(hidden_size, embedding_size),
            nn.AdaptiveAvgPool2d((1,1))  # Ensures output is [N, embedding_size, 1, 1]
        )
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.input_channels = input_channels

    def forward(self, x):
        x = self.features(x)
        # Now x is [batch_size, embedding_size, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, embedding_size]
        return x

class MAMLModel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, final_embedding_size=64, output_size=5):
        super(MAMLModel, self).__init__()
        self.features = feature_extractor
        self.classifier = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size // 2),
            nn.ReLU(),
            nn.Linear(final_embedding_size // 2, output_size)
        )

        maml_init_(self.classifier)
        self.output_size = output_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

import torch
import torch.nn as nn
from torchsummary import summary

# Enhanced EdgeQKVAttention with Residual Connection
class EnhancedEdgeQKVAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0, "dim should be divisible by heads"

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # QKV Projection
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_bn = nn.BatchNorm2d(dim * 3)

        # Output Projection
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(dim)

        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

        # Residual connection
        self.residual = nn.Sequential()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # QKV computation
        qkv = self.qkv(x)  # B, 3*C, H, W
        qkv = self.qkv_bn(qkv)
        qkv = qkv.reshape(B, 3, self.heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, heads, N, head_dim

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, heads, N, N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Attention Output
        out = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        out = self.proj(out)
        out = self.proj_bn(out)
        out = self.proj_drop(out)

        # Residual Connection
        out += x
        return out

# Bottleneck Residual Block
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.activation(out)
        out = self.dropout(out)
        return out

# Depthwise Separable Convolution Block
def depthwise_separable_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.GELU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )

# Enhanced Convolutional Block with Optional Residual Connection
def enhanced_conv_block(in_channels, out_channels, pool=True):
    layers = [
        depthwise_separable_conv(in_channels, out_channels),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def enhanced_conv_block_no_pool(in_channels, out_channels):
    return nn.Sequential(
        depthwise_separable_conv(in_channels, out_channels),
    )

# Enhanced ProtoNet with Increased Capacity
class ProtoNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=128, z_dim=128):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            enhanced_conv_block(x_dim, hid_dim),               # [B, 64, H/2, W/2]
            BottleneckBlock(hid_dim, hid_dim // 2, hid_dim),    # [B, 64, H/2, W/2]
            enhanced_conv_block_no_pool(hid_dim, hid_dim),      # [B, 64, H/2, W/2]
            BottleneckBlock(hid_dim, hid_dim // 2, hid_dim),    # [B, 64, H/2, W/2]
            enhanced_conv_block(hid_dim, hid_dim * 2),          # [B, 128, H/4, W/4]
            BottleneckBlock(hid_dim * 2, hid_dim, hid_dim * 2),  # [B, 128, H/4, W/4]
            enhanced_conv_block_no_pool(hid_dim * 2, hid_dim * 2),  # [B, 128, H/4, W/4]
            BottleneckBlock(hid_dim * 2, hid_dim, hid_dim * 2),    # [B, 128, H/4, W/4]
            enhanced_conv_block(hid_dim * 2, hid_dim * 4),          # [B, 256, H/8, W/8]
            BottleneckBlock(hid_dim * 4, hid_dim * 2, hid_dim * 4),  # [B, 256, H/8, W/8]
            enhanced_conv_block_no_pool(hid_dim * 4, z_dim),        # [B, 64, H/8, W/8]
        )
        self.attention = EnhancedEdgeQKVAttention(z_dim, heads=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return x

    def test(self):
        x = torch.randn(128, 3, 112, 112)
        print(self.forward(x).shape)

    def load(
        self,
        path,
        map_location=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        self.load_state_dict(torch.load(path, map_location=map_location))

    @staticmethod
    def init_all_layers(module):
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, 1.0)
                torch.nn.init.constant_(layer.bias, 0.0)

# Function to Count Parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Instantiate and Verify Parameter Count
if __name__ == "__main__":
    model = ProtoNet()
    total_params = count_parameters(model)
    print(f"Enhanced ProtoNet Parameters: {total_params / 1e3:.2f}K")

    # Example forward pass
    input_tensor = torch.randn(8, 3, 112, 112)
    output = model(input_tensor)
    print(f"Output Shape: {output.shape}")  # Expected: [8, 64]
