import machinLearning
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

criterion = nn.CrossEntropyLoss()
num_samples = 100
feature_dim = 128
batch_size = 2

data = torch.randn(num_samples, feature_dim) #データの自動生成(本来はCNNの特徴量)
value = torch.randint(low=0, high=2, size=(num_samples,))
value = value.float()
dataset = TensorDataset(data, value)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_transformer(model, data_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data_loader)}')

# モデルのインスタンス化
input_dim = feature_dim
model_dim = 128  # モデル次元
num_heads = 8    # アテンションヘッド数
num_layers = 6   # Transformer層数
dropout = 0.1    # ドロップアウト率

model = machinLearning.TransformerModel(input_dim, model_dim, num_heads, num_layers, 2, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# モデル訓練の実行
train_transformer(model, data_loader, criterion, optimizer)
torch.save(model.state_dict(), 'transformer_model.pth')
