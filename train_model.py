import torch
from torchvision import datasets, transforms

from model import LeNetModel

if __name__ == "__main__":
    train_dataset = datasets.MNIST("data/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST("data/", train=True, transform=transforms.ToTensor(), download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device))

    model = LeNetModel(lr=0.001, device=device)
    for epochs in range(10):
        model.train(train_loader)
        model.test(test_loader)
        model.save_model()
