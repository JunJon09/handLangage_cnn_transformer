import torch.nn as nn
import torch.nn.functional as F

#CNNのモデル
class Video3DCNN(nn.Module):
    def __init__(self):
        super(Video3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(5, 5, 5), stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(5, 5, 5), stride=1, padding=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        return x

#Transformerのモデル

