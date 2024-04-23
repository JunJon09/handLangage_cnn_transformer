import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1
        )
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        print(len(x), "cnn")
        x = self.conv1(x)
        x = self.pool(x)
        return x
