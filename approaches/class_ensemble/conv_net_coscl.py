import sys
import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, inputsize, taskcla, use_TG):
        super().__init__()

        ncha, size, _ = inputsize
        self.taskcla = taskcla
        self.nExpert = 5
        self.f_size = 8
        self.last = torch.nn.ModuleList()
        self.s_gate = 1

        self.net1 = nn.Sequential(
            nn.Conv2d(ncha, self.f_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size, self.f_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.f_size, self.f_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*2, self.f_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.f_size*2, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*4, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc1 = nn.Linear(self.f_size*64, 256)  # 2048

        self.net2 = nn.Sequential(
            nn.Conv2d(ncha, self.f_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size, self.f_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.f_size, self.f_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*2, self.f_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.f_size*2, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*4, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc2 = nn.Linear(self.f_size*64, 256)  # 2048

        self.net3 = nn.Sequential(
            nn.Conv2d(ncha, self.f_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size, self.f_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.f_size, self.f_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*2, self.f_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.f_size*2, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*4, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc3 = nn.Linear(self.f_size*64, 256)  # 2048

        self.net4 = nn.Sequential(
            nn.Conv2d(ncha, self.f_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size, self.f_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.f_size, self.f_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*2, self.f_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.f_size*2, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*4, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc4 = nn.Linear(self.f_size*64, 256)  # 2048

        self.net5 = nn.Sequential(
            nn.Conv2d(ncha, self.f_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size, self.f_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.f_size, self.f_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*2, self.f_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.f_size*2, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*4, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc5 = nn.Linear(self.f_size*64, 256)  # 2048

        self.last_aux = []
        for i in range(self.nExpert):
            self.last_aux.append(torch.nn.ModuleList())
            for t, n in self.taskcla:
                self.last_aux[i].append(torch.nn.Linear(256, n))

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        self.relu = torch.nn.ReLU()
        self.sig_gate = torch.nn.Sigmoid()


    def forward(self, x, t, return_expert=False, avg_act=False):

        self.Experts = []
        self.Experts_feature = []

        h1 = self.net1(x)
        h1 = h1.view(x.shape[0], -1)
        self.Experts_feature.append(h1)
        h1 = self.relu(self.fc1(h1))
        h1 = self.drop2(h1)
        self.Experts.append(h1.unsqueeze(0))

        h2 = self.net2(x)
        h2 = h2.view(x.shape[0], -1)
        self.Experts_feature.append(h2)
        h2 = self.relu(self.fc2(h2))
        h2 = self.drop2(h2)
        self.Experts.append(h2.unsqueeze(0))

        h3 = self.net3(x)
        h3 = h3.view(x.shape[0], -1)
        self.Experts_feature.append(h3)
        h3 = self.relu(self.fc3(h3))
        h3 = self.drop2(h3)
        self.Experts.append(h3.unsqueeze(0))

        h4 = self.net4(x)
        h4 = h4.view(x.shape[0], -1)
        self.Experts_feature.append(h4)
        h4 = self.relu(self.fc4(h4))
        h4 = self.drop2(h4)
        self.Experts.append(h4.unsqueeze(0))

        h5 = self.net5(x)
        h5 = h5.view(x.shape[0], -1)
        self.Experts_feature.append(h5)
        h5 = self.relu(self.fc5(h5))
        h5 = self.drop2(h5)
        self.Experts.append(h5.unsqueeze(0))

        self.Experts_y = []
        for i in range(self.nExpert):
            h_exp = self.Experts[i].squeeze(0)
            y_exp = self.last_aux[i][t](h_exp)
            self.Experts_y.append(y_exp)

        y = torch.cat([y_result.unsqueeze(0) for y_result in self.Experts_y], 0)
        y = torch.sum(y, dim=0) / self.nExpert

        if return_expert:
            return self.Experts_y,  y, self.Experts

        else:
            return y
