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

def video_to_frames(video_file, resize=(100, 100)):
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


# モデルの初期化
model = CNN.Simple3DCNN()
# 動画からフレームへ
frames = video_to_frames('./01.mp4')  # 動画ファイルのパスを指定
frames = frames.transpose((3, 0, 1, 2))  # PyTorchが期待する形式に変更: (C, T, H, W)
frames = np.expand_dims(frames, axis=0)  # バッチの次元を追加: (N, C, T, H, W)
frames = torch.tensor(frames, dtype=torch.float32)  # NumPy配列をTensorに変換

# モデルの適用
output = model(frames)
print(output.shape)  # 出力の形状を確認
# ハイパーパラメータの設定
input_dim = 112 * 112 * 64  # これは3D CNNからの出力の特徴量の次元数
model_dim = 216  # Transformer内の特徴量の次元数
num_heads = 8
num_layers = 3

# モデルの初期化
transformer_model = Transformer.TransformerModel(input_dim, model_dim, num_heads, num_layers)

# 3D CNNの出力をTransformerの入力形式に変換
cnn_output = output.squeeze(0)  # バッチ次元を削除
cnn_output = cnn_output.view(cnn_output.size(0), -1)  # (T, C*H*W)
cnn_output = cnn_output.permute(1, 0)  # (C*H*W, T) となっているため (T, C*H*W)
cnn_output = F.pad(cnn_output, (0, model_dim - cnn_output.size(1)), "constant", 0)  # 必要な場合はパディングを追加
cnn_output = cnn_output.unsqueeze(1)  # バッチ次元を追加 (T, N, model_dim)

# Transformerによる処理
transformed_output = transformer_model(cnn_output)
print(transformed_output.shape)  # 出力の形状を確認

class SignLanguageClassifier(nn.Module):
    def __init__(self, model_dim, num_classes):
        super(SignLanguageClassifier, self).__init__()
        self.linear = nn.Linear(model_dim, num_classes)  # 線形変換

    def forward(self, x):
        x = self.linear(x)  # (T, N, num_classes)
        x = F.softmax(x, dim=2)  # SoftMax関数を適用
        return x

# パラメータの設定
num_classes = 2  # 例えば、100個の手話単語クラスがあるとします

# モデルの初期化
classifier = SignLanguageClassifier(model_dim, num_classes)

# Transformerの出力をクラス分類器に適用
classification_output = classifier(transformed_output)

# 出力形状と内容を確認
print(classification_output.shape)  # 出力形状を確認



class SignLanguageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))] 

        self.data = []
        for label, class_dir in enumerate(self.classes):
            class_path = os.path.join(directory, class_dir)
            video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]  # MP4ファイルだけを対象とする
            for video_file in video_files:
                video_path = os.path.join(class_path, video_file)
                frames = video_to_frames(video_path)
                self.data.append((frames, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        frames, label = self.data[idx]
        transformed_frames = []
        if self.transform:
            for frame in frames:
                transformed_frame = self.transform(frame)  # フレームごとにトランスフォームを適用
                transformed_frames.append(transformed_frame)
        return torch.stack(transformed_frames), label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SignLanguageDataset('./all/', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# モデルの定義 (Simple3DCNNとTransformerModelを組み合わせ)
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
        # = x.view(x.size(0), -1).permute(1, 0)
        cnn_output = x.squeeze(0)
        cnn_output = cnn_output.view(cnn_output.size(0), -1)  # (T, C*H*W)
        cnn_output = cnn_output.permute(1, 0)  # (C*H*W, T) となっているため (T, C*H*W)
        cnn_output = F.pad(cnn_output, (0, model_dim - cnn_output.size(1)), "constant", 0)  # 必要な場合はパディングを追加
        x = cnn_output.unsqueeze(1)  # バッチ次元を追加 (T, N, model_dim)
        x = self.transformer(x)
        x = self.classifier(x)
        return x


# モデルコンポーネントのインスタンス化
cnn = CNN.Simple3DCNN()
transformer = Transformer.TransformerModel(input_dim, model_dim, num_heads, num_layers)
classifier = SignLanguageClassifier(model_dim, num_classes)

model = CompleteModel(cnn, transformer, classifier)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練ループ

for epoch in range(10):  # 10エポック訓練
    print(dataloader)
    for inputs, labels in dataloader:
        print("gggg")

        optimizer.zero_grad()
        print("aaaa")
        outputs = model(inputs)
        print("bbbb")
        loss = criterion(outputs, labels)
        print("bbbbbbnbbb")
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

test_video = video_to_frames('01.mp4')
test_video = transform(test_video)  # 前処理
test_video = test_video.unsqueeze(0)  # バッチの次元を追加

# 推論
model.eval()
with torch.no_grad():
    prediction = model(test_video)
    predicted_class = torch.argmax(prediction, dim=1)
    print(f'The predicted class for 01.mp4 is: {dataset.classes[predicted_class.item()]}')
