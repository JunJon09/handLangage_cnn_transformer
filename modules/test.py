import os
import machinLearning
import torch

def main():
    model_path = "./video_3dcnn_model.pth"
    model = machinLearning.Video3DCNNModel()
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    test_input = torch.randn(3, 40, 15, 15)
    test_output = model(test_input.unsqueeze(0))
    print("CNN完了")
    model_path = "./transformer_model.pth"
    model = machinLearning.TransformerModel(input_dim=128, model_dim=128, num_heads=8, num_layers=6, num_classes=2, dropout=0.1)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    test = model(test_output)
    print(test)


if __name__ == "__main__":
    main()