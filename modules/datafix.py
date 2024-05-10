import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import machinLearning


#データの加工、呼び出しなどを行うclass
class DataFix():
    def __init__(self) -> None:
        self.pre = []
        self.train = []
        self.test = []
        self.data_dict = {}

    def getPreData(self,):
        #作業を効率化するためのコード
        file_path = "../data/RGBdata/movie_cnn_pre.pkl"
        with open(file_path, 'rb') as f:
            self.pre = pickle.load(f)
        file_path = "../data/RGBdata/movie_cnn_train.pkl"
        with open(file_path, 'rb') as f:
            self.train = pickle.load(f)
        return 1
        directory = "/Users/jonmac/jon/研究/手話/CNNTransformer/venv/judge/test/pre" #ファイルパス
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes = sorted(classes)
        for label, class_dir in enumerate(classes):
            class_path = os.path.join(directory, class_dir)
            video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
            for video_file in video_files:
                video_path = os.path.join(class_path, video_file)
                frames = self.video_to_frames(video_path)
                frames = torch.tensor(frames, dtype=torch.float32)# NumPy配列をTensorに変換
                frames = frames.permute(3, 0, 1, 2)
                self.pre.append((frames, torch.tensor(label)))
                print(video_file)
        self.pre = DataLoader(self.pre, batch_size=1, shuffle=True)

        directory = "/Users/jonmac/jon/研究/手話/CNNTransformer/venv/judge/test/train" #ファイルパス
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes = sorted(classes)
        for label, class_dir in enumerate(classes):
            class_path = os.path.join(directory, class_dir)
            video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
            for video_file in video_files:
                video_path = os.path.join(class_path, video_file)
                frames = self.video_to_frames(video_path)
                frames = torch.tensor(frames, dtype=torch.float32)# NumPy配列をTensorに変換
                frames = frames.permute(3, 0, 1, 2)
                self.train.append((frames, torch.tensor(label)))
        self.train = DataLoader(self.train, batch_size=1, shuffle=True)
        file_path = "../data/RGBdata/movie_cnn_pre.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(self.pre, f)
        
        file_path = "../data/RGBdata/movie_cnn_train.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(self.train, f)
        

    #入力動画からCNNを通した値を取得する
    def getMovie_CNN(self):
        model = machinLearning.Video3DCNNModel()
        model_path = "./video_3dcnn_model.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        file_path = "../data/RGBdata/movie_cnn.pkl"
        with open(file_path, 'rb') as f:
            self.pre = pickle.load(f)
        cnn_data = []
        labels_data = []
        for i, (inputs, labels) in enumerate(self.pre):
            outputs = model(inputs)
            if i == 0:
                cnn_data = outputs
                labels_data = labels
                continue
            cnn_data = self.add_tensor(cnn_data, outputs)
            labels_data = self.add_tensor(labels_data, labels)

        return cnn_data, labels_data.float()

    def add_tensor(self, existing_tensors, new_tensor):
        return torch.cat((existing_tensors, new_tensor), dim=0)
    

    def video_to_frames(self, video_file, resize=(50, 50)):
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

a = DataFix()
a.getPreData()