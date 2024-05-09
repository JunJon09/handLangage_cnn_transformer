import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# CNNのモデル
class Video3DCNNModel(nn.Module):
    def __init__(self):
        super(Video3DCNNModel, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(5, 5, 5), stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(5, 5, 5), stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Transformerのモデル
class TransformerModel(nn.Module):
    def __init__(
        self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.1
    ):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(
            model_dim, input_dim
        )  # 500は仮の最大シーケンス長
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers
        )
        self.decoder = nn.Linear(
            model_dim, input_dim
        )  # 出力サイズをinput_dimに合わせる
        self.classifier = nn.Linear(input_dim, num_classes)  # クラス分類のための出力層

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = self.classifier(output)
        output = F.softmax(output, dim=1)
        # max_indices = torch.argmax(output, dim=1)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[: x.size(0), : x.size(1)]
        return self.dropout(x)
