import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse

from model import LeNetModel

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', type=str, default=False, required=False)
    parser.add_argument('--eps', type=float, default=0.25, required=False)
    return parser

if __name__ == "__main__":
    train_dataset = datasets.MNIST("data/", train=True, transform=transforms.ToTensor(), download=True)
    dataset = torch.utils.data.Subset(train_dataset, [0, 23, 179, 239, 2020])

    model = LeNetModel(lr=0.001, device=torch.device("cpu"))
    model.load_model()

    args = create_parser().parse_args()

    eps = args.eps
    iterations = 200
    alpha = 0.005

    for image, label in dataset:
        yh = model.predict_logprobs(image)
        actual_class = torch.argmax(yh)

        y_ll = torch.argmin(yh)
        noise = torch.zeros_like(image, requires_grad=True)

        for _ in range(iterations):
            y_hh = model.predict_logprobs(image + noise)
            loss = -y_hh[y_ll]

            loss.backward()
            with torch.no_grad():
                noise -= alpha * torch.sign(noise.grad)
                noise.grad.zero_()

                noise = torch.clamp(noise, -eps, +eps) # -eps <= noise <= eps
                noise = torch.max(-image, torch.min(noise, 1 - image)) # 0 <= image + noise <= 255
            noise.requires_grad_()

        if args.show:
            fig=plt.figure(figsize=(10, 10))
            fig.add_subplot(1, 2, 1)
            plt.imshow(image.squeeze(0), cmap='gray', vmin = 0, vmax = 1)
            fig.add_subplot(1, 2, 2)
            plt.imshow((image + noise).squeeze(0).detach().numpy(), cmap='gray', vmin = 0, vmax = 1)
            plt.show()
        
        predicted = model.predict_logprobs(image + noise)
        print("Actual Class {} Target Class {}\nLogprob Actual {} \nLogprob Target {}\n\n".format(
            actual_class, y_ll, predicted[actual_class], predicted[y_ll]))

