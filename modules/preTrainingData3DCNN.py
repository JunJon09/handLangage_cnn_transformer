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
def train_model(model, data_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# モデルのインスタンス化、損失関数、オプティマイザーの設定
model = machinLearning.Video3DCNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#data_loader = DataLoader([(torch.randn(3, 10, 15, 15), torch.tensor(1)) for _ in range(10)], batch_size=1, shuffle=True)

data_class = datafix.DataFix()
data_class.getPreData()
train_model(model,data_class.pre, criterion, optimizer)

# モデルの重みを保存
torch.save(model.state_dict(), 'video_3dcnn_model.pth')