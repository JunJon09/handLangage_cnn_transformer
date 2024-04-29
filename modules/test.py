import os
import machinLearning
import torch

def main():
    model_path = "./video_3dcnn_model.pth"
    model = machinLearning.Video3DCNN()
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    test_input = torch.randn(3, 10, 15, 15)
    test_output = model(test_input.unsqueeze(0))
    print(test_output)


if __name__ == "__main__":
    main()