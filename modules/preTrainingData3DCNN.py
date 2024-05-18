import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import machinLearning
import datafix
"""
RGB,フレーム数, 幅, 高さの次元をrandomで生成する。(実際の映像になったらこれらを入れ替える)
その答えが1とするの10個。
batch_sizeを使用して、二つまとめて処理
"""


# 訓練ループ
def train_model(model, data_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        print(epoch)
        total_train_loss = 0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs)
            loss = criterion(outputs, labels)
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


# モデルのインスタンス化、損失関数、オプティマイザーの設定
model = machinLearning.Video3DCNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

data_class = datafix.DataFix()
data_class.get_pre_and_train()

train_model(model,data_class.pre, data_class.train, criterion, optimizer)

# モデルの重みを保存
torch.save(model.state_dict(), 'video_3dcnn_model.pth')