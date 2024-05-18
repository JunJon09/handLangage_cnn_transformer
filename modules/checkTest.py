import machinLearning
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import datafix

input_dim = 128
model_dim = 128  # モデル次元
num_heads = 8    # アテンションヘッド数
num_layers = 9  # Transformer層数
dropout = 0.1    # ドロップアウト率

model = machinLearning.TransformerModel(input_dim, model_dim, num_heads, num_layers, 64, dropout)
model_path = "./transformer_model.pth"
if os.path.exists(model_path):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)


data_class = datafix.DataFix()
test, label = data_class.get_cnn_test()
dataset = TensorDataset(test, label.long())
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
for input, label in data_loader:

    outputs = model(input)

    print(outputs, label)