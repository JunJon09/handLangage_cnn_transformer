import machinLearning
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datafix


def train_transformer(model, data_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in data_loader:

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data_loader)}')




# モデルのインスタンス化
input_dim = 128
model_dim = 128  # モデル次元
num_heads = 8    # アテンションヘッド数
num_layers = 6   # Transformer層数
dropout = 0.1    # ドロップアウト率


model = machinLearning.TransformerModel(input_dim, model_dim, num_heads, num_layers, 64, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


data_class = datafix.DataFix()
cnn_data, labels = data_class.getMovie_CNN()
dataset = TensorDataset(cnn_data, labels.long())
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# モデル訓練の実行
train_transformer(model, data_loader, criterion, optimizer)
torch.save(model.state_dict(), 'transformer_model.pth')
