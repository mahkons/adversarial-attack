import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm

# LeNet model definition
# with MaxPool layers and ReLU activation 
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(400, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.seq(x)


class LeNetModel():
    def __init__(self, lr=0.01, device=torch.device("cpu")):
        self.device = device
        self.net = LeNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def train(self, traindata):
        loss_hist = list()
        sum_loss = 0
        for i, (X, y) in enumerate(traindata):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.net(X)
            loss = F.nll_loss(y_pred, y)
            sum_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 50 == 0:
                print("Iter {} Loss {}".format(i + 1, sum_loss))
                sum_loss = 0
        return loss_hist

    @torch.no_grad()
    def test(self, testdata):
        loss_hist = list()
        correct_hist = list()
        for i, (X, y) in enumerate(testdata):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.net(X)
            loss = F.nll_loss(y_pred, y)
            classes = torch.argmax(y_pred, dim=1)

            loss_hist.append(loss)
            correct_hist.append((classes == y).sum())

        print("Test. Loss {} Correct {}".format(sum(loss_hist), float(sum(correct_hist)) / len(testdata.dataset)))

        return loss_hist, correct_hist

    def predict(self, X):
        return torch.argmax(self.net(X.to(self.device)), dim=1)

    def save_model(self, path=os.path.join("generated", "LeNet.torch")):
        state_dict = self.net.state_dict()
        torch.save(state_dict, path)

    def load_model(self, path=os.path.join("generated", "LeNet.torch")):
        state_dict = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state_dict)
