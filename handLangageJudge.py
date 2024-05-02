import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import CNN
import Transformer

class loadVideos():
    def video_to_frames(video_file, resize=(15, 15)):
        cap = cv2.VideoCapture(video_file)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frames.append(frame)

        cap.release()
        return np.array(frames)

class SignLanguageClassifier(nn.Module):
    def __init__(self, model_dim, num_classes):
        super(SignLanguageClassifier, self).__init__()
        self.linear = nn.Linear(model_dim, num_classes)  # 線形変換

    def forward(self, x):
        x = self.linear(x)  # (T, N, num_classes)
        x = F.softmax(x, dim=2)  # SoftMax関数を適用
        return x

class SignLanguageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        self.data = []
        for label, class_dir in enumerate(self.classes):
            class_path = os.path.join(directory, class_dir)
            video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
            for video_file in video_files:
                video_path = os.path.join(class_path, video_file)
                frames = loadVideos.video_to_frames(video_path)
                self.data.append((frames, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, label = self.data[idx]
        return frames, label

# モデルの定義 (SimpleCNNとTransformerModelを組み合わせ)
class CompleteModel(nn.Module):
    def __init__(self, cnn, transformer, classifier):
        super(CompleteModel, self).__init__()
        self.cnn = cnn
        self.transformer = transformer
        self.classifier = classifier


    def forward(self, x):
        frames = x.transpose((3, 0, 1, 2))  # PyTorchが期待する形式に変更: (C, T, H, W)
        frames = np.expand_dims(frames, axis=0)  # バッチの次元を追加: (N, C, T, H, W)
        frames = torch.tensor(frames, dtype=torch.float32)  # NumPy配列をTensorに変換
        x = self.cnn(frames)
        cnn_output = x.squeeze(0)
        cnn_output = cnn_output.view(cnn_output.size(0), -1)  # (T, C*H*W)
        cnn_output = cnn_output.permute(1, 0)  # (C*H*W, T) となっているため (T, C*H*W)
        cnn_output = F.pad(cnn_output, (0, model_dim - cnn_output.size(1)), "constant", 0)  # 必要な場合はパディングを追加
        x = cnn_output.unsqueeze(1)  # バッチ次元を追加 (T, N, model_dim)
        x = self.transformer(x)
        x = x.mean(dim=0, keepdim=True)  # Transformer出力の平均を取る
        x = self.classifier(x)  # Classifierを適用
        return x


# ハイパーパラメータの設定
input_dim = 112 * 112 * 64  # これは3D CNNからの出力の特徴量の次元数
model_dim = 216  # Transformer内の特徴量の次元数
num_heads = 8
num_layers = 3
num_classes = 2  # クラス

# モデルの初期化
cnn = CNN.Simple3DCNN()
transformer_model = Transformer.TransformerModel(input_dim, model_dim, num_heads, num_layers)
classifier = SignLanguageClassifier(model_dim, num_classes)

dataset = SignLanguageDataset('./all/')

model = CompleteModel(cnn, transformer_model, classifier)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練ループ

for epoch in range(1):  # 1エポック訓練（例として1エポックにしています）
    for inputs, labels in dataset:
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze(1)  # ビデオ全体に対する予測
        labels_tensor = torch.tensor([labels], dtype=torch.long)  # ビデオのラベル
        loss = criterion(outputs, labels_tensor)  # ラベルは1つのビデオに対して1つの値
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 推論
model.eval()
test_video = loadVideos.video_to_frames('./b.mp4')
with torch.no_grad():
    prediction = model(test_video)
    predicted_class = torch.argmax(prediction.squeeze(), dim=0)  # 最終的なクラス予測
    print(predicted_class)
    print(f'The predicted class for 01.mp4 is: {dataset.classes[predicted_class.item()]}')
