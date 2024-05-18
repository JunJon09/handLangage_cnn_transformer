import machinLearning
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datafix


def train_transformer(model, data_loader,val_loader, criterion, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        total_train_loss = 0
        l = len(data_loader)
        for i, (inputs, labels) in enumerate(data_loader):
            print(inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[0].float(), labels[0].long())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        print("評価モードに切り替え")
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_train_loss / len(data_loader)}, Val Loss: {total_val_loss / len(val_loader)}')




# モデルのインスタンス化
input_dim = 128
model_dim = 128  # モデル次元
num_heads = 8    # アテンションヘッド数
num_layers = 9  # Transformer層数
dropout = 0.3    # ドロップアウト率


model = machinLearning.TransformerModel(input_dim, model_dim, num_heads, num_layers, 64, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


data_class = datafix.DataFix()
cnn_pre_data, cnn_train_data, labels_pre_data, labels_train_data = data_class.get_video_cnn()
dataset = TensorDataset(cnn_pre_data, labels_pre_data)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
dataset = TensorDataset(cnn_train_data, labels_train_data.long())
val_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# モデル訓練の実行
train_transformer(model, data_loader, val_loader, criterion, optimizer)
torch.save(model.state_dict(), 'transformer_model.pth')
