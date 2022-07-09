import sys
import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F


# Network for AGS
class Net(nn.Module):
    def __init__(self, inputsize, taskcla, use_TG):
        super().__init__()

        ncha, size, _ = inputsize
        self.taskcla = taskcla
        self.nExpert = 5
        self.f_size = 8
        self.s_gate = 1

        self.conv1_1 = nn.Conv2d(ncha, self.f_size, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(self.f_size, self.f_size, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(self.f_size, self.f_size * 2, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(self.f_size * 2, self.f_size * 2, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(self.f_size * 2, self.f_size * 4, kernel_size=3, padding=1)
        self.conv6_1 = nn.Conv2d(self.f_size * 4, self.f_size * 4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.f_size * 64, 256)  # 2048
        # self.efc1 = torch.nn.Embedding(len(self.taskcla), 256)

        self.conv1_2 = nn.Conv2d(ncha, self.f_size, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(self.f_size, self.f_size, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(self.f_size, self.f_size * 2, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(self.f_size * 2, self.f_size * 2, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(self.f_size * 2, self.f_size * 4, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(self.f_size * 4, self.f_size * 4, kernel_size=3, padding=1)
        self.fc2 = nn.Linear(self.f_size * 64, 256)  # 2048
        # self.efc2 = torch.nn.Embedding(len(self.taskcla), 256)

        self.conv1_3 = nn.Conv2d(ncha, self.f_size, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(self.f_size, self.f_size, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(self.f_size, self.f_size * 2, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(self.f_size * 2, self.f_size * 2, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(self.f_size * 2, self.f_size * 4, kernel_size=3, padding=1)
        self.conv6_3 = nn.Conv2d(self.f_size * 4, self.f_size * 4, kernel_size=3, padding=1)
        self.fc3 = nn.Linear(self.f_size * 64, 256)  # 2048
        # self.efc3 = torch.nn.Embedding(len(self.taskcla), 256)

        self.conv1_4 = nn.Conv2d(ncha, self.f_size, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv2d(self.f_size, self.f_size, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(self.f_size, self.f_size * 2, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(self.f_size * 2, self.f_size * 2, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(self.f_size * 2, self.f_size * 4, kernel_size=3, padding=1)
        self.conv6_4 = nn.Conv2d(self.f_size * 4, self.f_size * 4, kernel_size=3, padding=1)
        self.fc4 = nn.Linear(self.f_size * 64, 256)  # 2048
        # self.efc4 = torch.nn.Embedding(len(self.taskcla), 256)

        self.conv1_5 = nn.Conv2d(ncha, self.f_size, kernel_size=3, padding=1)
        self.conv2_5 = nn.Conv2d(self.f_size, self.f_size, kernel_size=3, padding=1)
        self.conv3_5 = nn.Conv2d(self.f_size, self.f_size * 2, kernel_size=3, padding=1)
        self.conv4_5 = nn.Conv2d(self.f_size * 2, self.f_size * 2, kernel_size=3, padding=1)
        self.conv5_5 = nn.Conv2d(self.f_size * 2, self.f_size * 4, kernel_size=3, padding=1)
        self.conv6_5 = nn.Conv2d(self.f_size * 4, self.f_size * 4, kernel_size=3, padding=1)
        self.fc5 = nn.Linear(self.f_size * 64, 256)  # 2048
        # self.efc5 = torch.nn.Embedding(len(self.taskcla), 256)

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        self.last = torch.nn.ModuleList()
        self.num_tasks = 0
        for t, n in self.taskcla:
            self.num_tasks += 1
            self.last.append(torch.nn.Linear(256, n))

        self.relu = torch.nn.ReLU()
        self.sig_gate = torch.nn.Sigmoid()

        self.eg1 = torch.nn.Embedding(len(self.taskcla), 256)
        self.eg2 = torch.nn.Embedding(len(self.taskcla), 256)
        self.eg3 = torch.nn.Embedding(len(self.taskcla), 256)
        self.eg4 = torch.nn.Embedding(len(self.taskcla), 256)
        self.eg5 = torch.nn.Embedding(len(self.taskcla), 256)

    def forward(self, x, t, return_expert=False, avg_act=False):
        masks = self.mask(t, s=self.s_gate)
        eg1, eg2, eg3, eg4, eg5 = masks

        self.Experts = []

        act1_1 = self.relu(self.conv1_1(x))
        act2_1 = self.relu(self.conv2_1(act1_1))
        h = self.drop1(self.MaxPool(act2_1))
        act3_1 = self.relu(self.conv3_1(h))
        act4_1 = self.relu(self.conv4_1(act3_1))
        h = self.drop1(self.MaxPool(act4_1))
        act5_1 = self.relu(self.conv5_1(h))
        act6_1 = self.relu(self.conv6_1(act5_1))
        h = self.drop1(self.MaxPool(act6_1))
        h = h.view(x.shape[0], -1)
        act7_1 = self.relu(self.fc1(h))
        h = self.drop2(act7_1)
        h = h * eg1.expand_as(h)
        self.Experts.append(h.unsqueeze(0))

        act1_2 = self.relu(self.conv1_2(x))
        act2_2 = self.relu(self.conv2_2(act1_2))
        h = self.drop1(self.MaxPool(act2_2))
        act3_2 = self.relu(self.conv3_2(h))
        act4_2 = self.relu(self.conv4_2(act3_2))
        h = self.drop1(self.MaxPool(act4_2))
        act5_2 = self.relu(self.conv5_2(h))
        act6_2 = self.relu(self.conv6_2(act5_2))
        h = self.drop1(self.MaxPool(act6_2))
        h = h.view(x.shape[0], -1)
        act7_2 = self.relu(self.fc2(h))
        h = self.drop2(act7_2)
        h = h * eg2.expand_as(h)
        self.Experts.append(h.unsqueeze(0))

        act1_3 = self.relu(self.conv1_3(x))
        act2_3 = self.relu(self.conv2_3(act1_3))
        h = self.drop1(self.MaxPool(act2_3))
        act3_3 = self.relu(self.conv3_3(h))
        act4_3 = self.relu(self.conv4_3(act3_3))
        h = self.drop1(self.MaxPool(act4_3))
        act5_3 = self.relu(self.conv5_3(h))
        act6_3 = self.relu(self.conv6_3(act5_3))
        h = self.drop1(self.MaxPool(act6_3))
        h = h.view(x.shape[0], -1)
        act7_3 = self.relu(self.fc3(h))
        h = self.drop2(act7_3)
        h = h * eg3.expand_as(h)
        self.Experts.append(h.unsqueeze(0))

        act1_4 = self.relu(self.conv1_4(x))
        act2_4 = self.relu(self.conv2_4(act1_4))
        h = self.drop1(self.MaxPool(act2_4))
        act3_4 = self.relu(self.conv3_4(h))
        act4_4 = self.relu(self.conv4_4(act3_4))
        h = self.drop1(self.MaxPool(act4_4))
        act5_4 = self.relu(self.conv5_4(h))
        act6_4 = self.relu(self.conv6_4(act5_4))
        h = self.drop1(self.MaxPool(act6_4))
        h = h.view(x.shape[0], -1)
        act7_4 = self.relu(self.fc4(h))
        h = self.drop2(act7_4)
        h = h * eg4.expand_as(h)
        self.Experts.append(h.unsqueeze(0))

        act1_5 = self.relu(self.conv1_5(x))
        act2_5 = self.relu(self.conv2_5(act1_5))
        h = self.drop1(self.MaxPool(act2_5))
        act3_5 = self.relu(self.conv3_5(h))
        act4_5 = self.relu(self.conv4_5(act3_5))
        h = self.drop1(self.MaxPool(act4_5))
        act5_5 = self.relu(self.conv5_5(h))
        act6_5 = self.relu(self.conv6_5(act5_5))
        h = self.drop1(self.MaxPool(act6_5))
        h = h.view(x.shape[0], -1)
        act7_5 = self.relu(self.fc5(h))
        h = self.drop2(act7_5)
        h = h * eg5.expand_as(h)
        self.Experts.append(h.unsqueeze(0))

        h = torch.cat([h_result for h_result in self.Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)  # / self.nExpert
        y = self.last[t](h)

        self.grads = {}

        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad

            return hook

        if avg_act == True:

            names = [0, 1, 2, 3, 4, 5, 6,
                     7, 8, 9, 10, 11, 12, 13,
                     14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34]

            act = [act1_1, act2_1, act3_1, act4_1, act5_1, act6_1, act7_1,
                   act1_2, act2_2, act3_2, act4_2, act5_2, act6_2, act7_2,
                   act1_3, act2_3, act3_3, act4_3, act5_3, act6_3, act7_3,
                   act1_4, act2_4, act3_4, act4_4, act5_4, act6_4, act7_4,
                   act1_5, act2_5, act3_5, act4_5, act5_5, act6_5, act7_5]

            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))

        if return_expert:
            y_exp_result = []
            for i in range(self.nExpert):
                h_exp = self.Experts[i].squeeze(0)
                y_exp = self.last[t](h_exp)
                y_exp_result.append(y_exp)
            return y, y_exp_result

        else:
            return y

    def mask(self, t, s=1):
        eg1 = self.sig_gate(s * self.eg1(t))
        eg2 = self.sig_gate(s * self.eg2(t))
        eg3 = self.sig_gate(s * self.eg3(t))
        eg4 = self.sig_gate(s * self.eg4(t))
        eg5 = self.sig_gate(s * self.eg5(t))
        return [eg1, eg2, eg3, eg4, eg5]

