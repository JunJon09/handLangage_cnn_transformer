import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import machinLearning
import gc

"""
命名規則
動詞_何のデータ_ なんのデータ
Video->入力動画
CNN->3DCNN
"""

#データの加工、呼び出しなどを行うclass
class DataFix():
    def __init__(self) -> None:
        self.pre = []
        self.train = []
        self.test = []
        self.data_dict = {}

    #torchListに追加する
    def add_tensor(self, existing_tensors, new_tensor):
        return torch.cat((existing_tensors, new_tensor), dim=0)


    #ビデオデータをフレーム毎に保存
    def save_videos(self):
        def get_video(video_file, resize=(50, 50)):
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
        
        def save_video(directory, file_path):
            classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
            classes = sorted(classes)
            file_list = []
            for label, class_dir in enumerate(classes):
                class_path = os.path.join(directory, class_dir)
                video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
                for video_file in video_files:
                    video_path = os.path.join(class_path, video_file)
                    frames = get_video(video_path)
                    frames = torch.tensor(frames, dtype=torch.float32)# NumPy配列をTensorに変換
                    frames = frames.permute(3, 0, 1, 2)
                    file_list.append((frames, torch.tensor(label)))
                    print(video_file)
            file_list = DataLoader(file_list, batch_size=1, shuffle=False)
            with open(file_path, 'wb') as f:
                pickle.dump(file_list, f)

        pre_directory = "/Users/jonmac/jon/研究/手話/CNNTransformer/venv/judge/test/pre"
        pre_file_path = "../data/RGBdata/movie_pre.pkl"
        a = save_video(pre_directory, pre_file_path)
        del a
        train_directory = "/Users/jonmac/jon/研究/手話/CNNTransformer/venv/judge/test/train"
        train_file_path = "../data/RGBdata/movie_train.pkl"
        a = save_video(train_directory, train_file_path)
        del a
        test_directory = "/Users/jonmac/jon/研究/手話/CNNTransformer/venv/judge/test/test"
        test_file_path = "../data/RGBdata/movie_test.pkl"
        a = save_video(test_directory, test_file_path)

    #ビデオデータの事前データと検証データを取得
    def get_pre_and_train(self):

        file_path = "../data/RGBdata/movie_pre.pkl"
        with open(file_path, 'rb') as f:
            self.pre = pickle.load(f)
        file_path = "../data/RGBdata/movie_train.pkl"
        with open(file_path, 'rb') as f:
            self.train = pickle.load(f)
        print(len(self.pre))


    #ビデオデータをCNNに通した値を保存
    def save_video_cnns(self):
        model = machinLearning.Video3DCNNModel()
        model_path = "./video_3dcnn_model.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)

        def save_video_cnn(data_file_path, save_file_data_path, save_file_label_path):
            with open(data_file_path, 'rb') as f:
               data = pickle.load(f)
            
            cnn_data = []
            labels_data = []
            print(len(data))
            for i, (inputs, labels) in enumerate(data):
                # if i< 1100:
                #     continue
                print(i)
                outputs = model(inputs)
                if i == 0:
                    cnn_data = outputs
                    labels_data = labels
                    continue
                cnn_data = self.add_tensor(cnn_data, outputs)
                labels_data = self.add_tensor(labels_data, labels)
            print(len(cnn_data), "aaaa")
            with open(save_file_data_path, 'wb') as f:
                pickle.dump(cnn_data, f)
            with open(save_file_label_path, 'wb') as f:
                pickle.dump(labels_data, f)
        # #訓練データ
        data_file_path = "../data/RGBdata/movie_pre.pkl"
        save_file_data_path = "../data/RGBdata/movie_cnn_pre_1100.pkl"
        save_file_label_path = "../data/RGBdata/movie_cnn_pre_label_1100.pkl"
        a = save_video_cnn(data_file_path, save_file_data_path, save_file_label_path)
        del a
        print("事前データをCNNに通せた")
        # #検証データ
        # data_file_path = "../data/RGBdata/movie_train.pkl"
        # save_file_data_path = "../data/RGBdata/movie_cnn_train.pkl"
        # save_file_label_path = "../data/RGBdata/movie_cnn_train_label.pkl"
        # a = save_video_cnn(data_file_path, save_file_data_path, save_file_label_path)

    def connect_cnn(self):
        #data
        data_file_path = "../data/RGBdata/movie_cnn_pre_0.pkl"
        with open(data_file_path, 'rb') as f:
            data_1 = pickle.load(f)
        data_file_path = "../data/RGBdata/movie_cnn_pre_1100.pkl"
        with open(data_file_path, 'rb') as f:
            data_2 = pickle.load(f)
        data = self.add_tensor(data_1, data_2)
        print(len(data), len(data_1), len(data_2))
        data_file_path = "../data/RGBdata/movie_cnn_pre.pkl"
        with open(data_file_path, 'wb') as f:
            pickle.dump(data, f)
        #label
        label_file_path = "../data/RGBdata/movie_cnn_pre_label_0.pkl"
        with open(label_file_path, 'rb') as f:
            label_1 = pickle.load(f)
        label_file_path = "../data/RGBdata/movie_cnn_pre_label_1100.pkl"
        with open(label_file_path, 'rb') as f:
            label_2 = pickle.load(f)
        label = self.add_tensor(label_1, label_2)
        print(len(label), len(label_1), len(label_2))

        print(label)
        label_file_path = "../data/RGBdata/movie_cnn_pre_label.pkl"
        with open(label_file_path, 'wb') as f:
            pickle.dump(label, f)



    #入力動画からCNNを通した値を取得する
    def get_video_cnn(self):
        pre_data_file_path = "../data/RGBdata/movie_cnn_pre.pkl"
        with open(pre_data_file_path, 'rb') as f:
            cnn_pre_data = pickle.load(f)
        pre_label_file_path = "../data/RGBdata/movie_cnn_pre_label.pkl"
        with open(pre_label_file_path, 'rb') as f:
            labels_pre_data = pickle.load(f)
        train_data_file_path = "../data/RGBdata/movie_cnn_train.pkl"
        with open(train_data_file_path, 'rb') as f:
            cnn_train_data = pickle.load(f)
        train_label_file_path = "../data/RGBdata/movie_cnn_train_label.pkl"
        with open(train_label_file_path, 'rb') as f:
            labels_train_data = pickle.load(f)
        return cnn_pre_data, cnn_train_data, labels_pre_data, labels_train_data

    #CNNに通した値を保存する
    def save_video_cnn_test(self):

        model = machinLearning.Video3DCNNModel()
        model_path = "./video_3dcnn_model.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)

        def save_video_cnn(data_file_path, save_file_data_path, save_file_label_path):
            with open(data_file_path, 'rb') as f:
               data = pickle.load(f)
            
            cnn_data = []
            labels_data = []
            print(len(data))
            for i, (inputs, labels) in enumerate(data):
                print(i)
                outputs = model(inputs)
                print(outputs)
                print(outputs.size())
                if i == 0:
                    cnn_data = outputs
                    labels_data = labels
                    continue
                cnn_data = self.add_tensor(cnn_data, outputs)
                labels_data = self.add_tensor(labels_data, labels)
            print(len(cnn_data), "aaaa")
            with open(save_file_data_path, 'wb') as f:
                pickle.dump(cnn_data, f)
            with open(save_file_label_path, 'wb') as f:
                pickle.dump(labels_data, f)

        data_file_path = "../data/RGBdata/movie_test.pkl"
        save_file_data_path = "../data/RGBdata/movie_cnn_test.pkl"
        save_file_label_path = "../data/RGBdata/movie_cnn_test_label.pkl"

        save_video_cnn(data_file_path, save_file_data_path, save_file_label_path)


    def get_cnn_test(self):
        file_path = "../data/RGBdata/movie_cnn_test.pkl"
        with open(file_path, 'rb') as f:
            self.test = pickle.load(f)
        file_path = "../data/RGBdata/movie_cnn_test_label.pkl"
        with open(file_path, 'rb') as f:
            label = pickle.load(f)
        return self.test, label

DataFix().save_video_cnn_test()