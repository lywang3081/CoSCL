import sys
import torch
import torch.nn as nn
from utils import *


class Net(torch.nn.Module):
    def __init__(self,inputsize,taskcla, use_TG):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.nExpert = 5
        self.f_size = 8

        self.c1_1 = nn.Conv2d(ncha,self.f_size,kernel_size=3,padding=1)
        self.c2_1 = nn.Conv2d(self.f_size,self.f_size,kernel_size=3,padding=1)
        self.c3_1 = nn.Conv2d(self.f_size,self.f_size*2,kernel_size=3,padding=1)
        self.c4_1 = nn.Conv2d(self.f_size*2,self.f_size*2,kernel_size=3,padding=1)
        self.c5_1 = nn.Conv2d(self.f_size*2,self.f_size*4,kernel_size=3,padding=1)
        self.c6_1 = nn.Conv2d(self.f_size*4,self.f_size*4,kernel_size=3,padding=1)
        self.fc1_1 = nn.Linear(self.f_size*64, 256) # 2048

        self.c1_2 = nn.Conv2d(ncha,self.f_size,kernel_size=3,padding=1)
        self.c2_2 = nn.Conv2d(self.f_size,self.f_size,kernel_size=3,padding=1)
        self.c3_2 = nn.Conv2d(self.f_size,self.f_size*2,kernel_size=3,padding=1)
        self.c4_2 = nn.Conv2d(self.f_size*2,self.f_size*2,kernel_size=3,padding=1)
        self.c5_2 = nn.Conv2d(self.f_size*2,self.f_size*4,kernel_size=3,padding=1)
        self.c6_2 = nn.Conv2d(self.f_size*4,self.f_size*4,kernel_size=3,padding=1)
        self.fc1_2 = nn.Linear(self.f_size*64, 256) # 2048

        self.c1_3 = nn.Conv2d(ncha,self.f_size,kernel_size=3,padding=1)
        self.c2_3 = nn.Conv2d(self.f_size,self.f_size,kernel_size=3,padding=1)
        self.c3_3 = nn.Conv2d(self.f_size,self.f_size*2,kernel_size=3,padding=1)
        self.c4_3 = nn.Conv2d(self.f_size*2,self.f_size*2,kernel_size=3,padding=1)
        self.c5_3 = nn.Conv2d(self.f_size*2,self.f_size*4,kernel_size=3,padding=1)
        self.c6_3 = nn.Conv2d(self.f_size*4,self.f_size*4,kernel_size=3,padding=1)
        self.fc1_3 = nn.Linear(self.f_size*64, 256) # 2048

        self.c1_4 = nn.Conv2d(ncha,self.f_size,kernel_size=3,padding=1)
        self.c2_4 = nn.Conv2d(self.f_size,self.f_size,kernel_size=3,padding=1)
        self.c3_4 = nn.Conv2d(self.f_size,self.f_size*2,kernel_size=3,padding=1)
        self.c4_4 = nn.Conv2d(self.f_size*2,self.f_size*2,kernel_size=3,padding=1)
        self.c5_4 = nn.Conv2d(self.f_size*2,self.f_size*4,kernel_size=3,padding=1)
        self.c6_4 = nn.Conv2d(self.f_size*4,self.f_size*4,kernel_size=3,padding=1)
        self.fc1_4 = nn.Linear(self.f_size*64, 256) # 2048

        self.c1_5 = nn.Conv2d(ncha,self.f_size,kernel_size=3,padding=1)
        self.c2_5 = nn.Conv2d(self.f_size,self.f_size,kernel_size=3,padding=1)
        self.c3_5 = nn.Conv2d(self.f_size,self.f_size*2,kernel_size=3,padding=1)
        self.c4_5 = nn.Conv2d(self.f_size*2,self.f_size*2,kernel_size=3,padding=1)
        self.c5_5 = nn.Conv2d(self.f_size*2,self.f_size*4,kernel_size=3,padding=1)
        self.c6_5 = nn.Conv2d(self.f_size*4,self.f_size*4,kernel_size=3,padding=1)
        self.fc1_5 = nn.Linear(self.f_size*64, 256) # 2048

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        
        self.smid= 4 #s
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256,n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1_1 =torch.nn.Embedding(len(self.taskcla),self.f_size)
        self.ec2_1 =torch.nn.Embedding(len(self.taskcla),self.f_size)
        self.ec3_1 =torch.nn.Embedding(len(self.taskcla),self.f_size*2)
        self.ec4_1 =torch.nn.Embedding(len(self.taskcla),self.f_size*2)
        self.ec5_1 =torch.nn.Embedding(len(self.taskcla),self.f_size*4)
        self.ec6_1 =torch.nn.Embedding(len(self.taskcla),self.f_size*4)
        self.efc1_1 =torch.nn.Embedding(len(self.taskcla),256)

        self.ec1_2 =torch.nn.Embedding(len(self.taskcla),self.f_size)
        self.ec2_2 =torch.nn.Embedding(len(self.taskcla),self.f_size)
        self.ec3_2 =torch.nn.Embedding(len(self.taskcla),self.f_size*2)
        self.ec4_2 =torch.nn.Embedding(len(self.taskcla),self.f_size*2)
        self.ec5_2 =torch.nn.Embedding(len(self.taskcla),self.f_size*4)
        self.ec6_2 =torch.nn.Embedding(len(self.taskcla),self.f_size*4)
        self.efc1_2 =torch.nn.Embedding(len(self.taskcla),256)

        self.ec1_3 =torch.nn.Embedding(len(self.taskcla),self.f_size)
        self.ec2_3 =torch.nn.Embedding(len(self.taskcla),self.f_size)
        self.ec3_3 =torch.nn.Embedding(len(self.taskcla),self.f_size*2)
        self.ec4_3 =torch.nn.Embedding(len(self.taskcla),self.f_size*2)
        self.ec5_3 =torch.nn.Embedding(len(self.taskcla),self.f_size*4)
        self.ec6_3 =torch.nn.Embedding(len(self.taskcla),self.f_size*4)
        self.efc1_3 =torch.nn.Embedding(len(self.taskcla),256)

        self.ec1_4 =torch.nn.Embedding(len(self.taskcla),self.f_size)
        self.ec2_4 =torch.nn.Embedding(len(self.taskcla),self.f_size)
        self.ec3_4 =torch.nn.Embedding(len(self.taskcla),self.f_size*2)
        self.ec4_4 =torch.nn.Embedding(len(self.taskcla),self.f_size*2)
        self.ec5_4 =torch.nn.Embedding(len(self.taskcla),self.f_size*4)
        self.ec6_4 =torch.nn.Embedding(len(self.taskcla),self.f_size*4)
        self.efc1_4 =torch.nn.Embedding(len(self.taskcla),256)

        self.ec1_5 =torch.nn.Embedding(len(self.taskcla),self.f_size)
        self.ec2_5 =torch.nn.Embedding(len(self.taskcla),self.f_size)
        self.ec3_5 =torch.nn.Embedding(len(self.taskcla),self.f_size*2)
        self.ec4_5 =torch.nn.Embedding(len(self.taskcla),self.f_size*2)
        self.ec5_5 =torch.nn.Embedding(len(self.taskcla),self.f_size*4)
        self.ec6_5 =torch.nn.Embedding(len(self.taskcla),self.f_size*4)
        self.efc1_5 =torch.nn.Embedding(len(self.taskcla),256)

        return

    def forward(self,t,x,s=1, return_expert=False):
        # Gates
        masks=self.mask(t,s=s)
        gc1_1, gc2_1, gc3_1, gc4_1, gc5_1, gc6_1, gfc1_1, gc1_2, gc2_2, gc3_2, gc4_2, gc5_2, gc6_2, gfc1_2, gc1_3, gc2_3, gc3_3, gc4_3, gc5_3, gc6_3, gfc1_3, \
        gc1_4, gc2_4, gc3_4, gc4_4, gc5_4, gc6_4, gfc1_4 , gc1_5, gc2_5, gc3_5, gc4_5, gc5_5, gc6_5, gfc1_5 =masks

        self.Experts = []

        # Gated
        # Net 1
        h=self.relu(self.c1_1(x))
        h=h*gc1_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c2_1(h))
        h=h*gc2_1.view(1,-1,1,1).expand_as(h)
        h=self.drop1(self.MaxPool(h))
        
        h=self.relu(self.c3_1(h))
        h=h*gc3_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c4_1(h))
        h=h*gc4_1.view(1,-1,1,1).expand_as(h)
        h=self.drop1(self.MaxPool(h))
        
        h=self.relu(self.c5_1(h))
        h=h*gc5_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c6_1(h))
        h=h*gc6_1.view(1,-1,1,1).expand_as(h)
        h=self.drop1(self.MaxPool(h))
        
        h=h.view(x.shape[0],-1)
        h=self.drop2(self.relu(self.fc1_1(h)))
        h=h*gfc1_1.expand_as(h)

        self.Experts.append(h.unsqueeze(0))

        # Net 2
        h = self.relu(self.c1_2(x))
        h = h * gc1_2.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c2_2(h))
        h = h * gc2_2.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = self.relu(self.c3_2(h))
        h = h * gc3_2.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c4_2(h))
        h = h * gc4_2.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = self.relu(self.c5_2(h))
        h = h * gc5_2.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c6_2(h))
        h = h * gc6_2.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = h.view(x.shape[0], -1)
        h = self.drop2(self.relu(self.fc1_2(h)))
        h = h * gfc1_2.expand_as(h)

        self.Experts.append(h.unsqueeze(0))

        # Net 3
        h = self.relu(self.c1_3(x))
        h = h * gc1_3.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c2_3(h))
        h = h * gc2_3.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = self.relu(self.c3_3(h))
        h = h * gc3_3.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c4_3(h))
        h = h * gc4_3.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = self.relu(self.c5_3(h))
        h = h * gc5_3.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c6_3(h))
        h = h * gc6_3.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = h.view(x.shape[0], -1)
        h = self.drop2(self.relu(self.fc1_3(h)))
        h = h * gfc1_3.expand_as(h)

        self.Experts.append(h.unsqueeze(0))


        # Net 4
        h = self.relu(self.c1_4(x))
        h = h * gc1_4.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c2_4(h))
        h = h * gc2_4.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = self.relu(self.c3_4(h))
        h = h * gc3_4.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c4_4(h))
        h = h * gc4_4.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = self.relu(self.c5_4(h))
        h = h * gc5_4.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c6_4(h))
        h = h * gc6_4.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = h.view(x.shape[0], -1)
        h = self.drop2(self.relu(self.fc1_4(h)))
        h = h * gfc1_4.expand_as(h)

        self.Experts.append(h.unsqueeze(0))


        # Net 5
        h = self.relu(self.c1_5(x))
        h = h * gc1_5.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c2_5(h))
        h = h * gc2_5.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = self.relu(self.c3_5(h))
        h = h * gc3_5.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c4_5(h))
        h = h * gc4_5.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = self.relu(self.c5_5(h))
        h = h * gc5_5.view(1, -1, 1, 1).expand_as(h)
        h = self.relu(self.c6_5(h))
        h = h * gc6_5.view(1, -1, 1, 1).expand_as(h)
        h = self.drop1(self.MaxPool(h))

        h = h.view(x.shape[0], -1)
        h = self.drop2(self.relu(self.fc1_5(h)))
        h = h * gfc1_5.expand_as(h)

        self.Experts.append(h.unsqueeze(0))


        h = torch.cat([h_result for h_result in self.Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0) #/ self.nExpert

        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))

        if return_expert:
            self.Experts_y = []
            for i in range(self.nExpert):
                h_exp = self.Experts[i].squeeze(0)

                # using joint classifier
                y_exp = self.last[t](h_exp)
                self.Experts_y.append(y_exp)

            return y, masks, self.Experts_y, self.Experts

        else:
            return y, masks

    def mask(self,t,s=1):
        gc1_1=self.gate(s*self.ec1_1(t))
        gc2_1=self.gate(s*self.ec2_1(t))
        gc3_1=self.gate(s*self.ec3_1(t))
        gc4_1=self.gate(s*self.ec4_1(t))
        gc5_1=self.gate(s*self.ec5_1(t))
        gc6_1=self.gate(s*self.ec6_1(t))
        gfc1_1=self.gate(s*self.efc1_1(t))

        gc1_2=self.gate(s*self.ec1_2(t))
        gc2_2=self.gate(s*self.ec2_2(t))
        gc3_2=self.gate(s*self.ec3_2(t))
        gc4_2=self.gate(s*self.ec4_2(t))
        gc5_2=self.gate(s*self.ec5_2(t))
        gc6_2=self.gate(s*self.ec6_2(t))
        gfc1_2=self.gate(s*self.efc1_2(t))

        gc1_3=self.gate(s*self.ec1_3(t))
        gc2_3=self.gate(s*self.ec2_3(t))
        gc3_3=self.gate(s*self.ec3_3(t))
        gc4_3=self.gate(s*self.ec4_3(t))
        gc5_3=self.gate(s*self.ec5_3(t))
        gc6_3=self.gate(s*self.ec6_3(t))
        gfc1_3=self.gate(s*self.efc1_3(t))

        gc1_4=self.gate(s*self.ec1_4(t))
        gc2_4=self.gate(s*self.ec2_4(t))
        gc3_4=self.gate(s*self.ec3_4(t))
        gc4_4=self.gate(s*self.ec4_4(t))
        gc5_4=self.gate(s*self.ec5_4(t))
        gc6_4=self.gate(s*self.ec6_4(t))
        gfc1_4=self.gate(s*self.efc1_4(t))

        gc1_5=self.gate(s*self.ec1_5(t))
        gc2_5=self.gate(s*self.ec2_5(t))
        gc3_5=self.gate(s*self.ec3_5(t))
        gc4_5=self.gate(s*self.ec4_5(t))
        gc5_5=self.gate(s*self.ec5_5(t))
        gc6_5=self.gate(s*self.ec6_5(t))
        gfc1_5=self.gate(s*self.efc1_5(t))


        masks = [gc1_1,gc2_1,gc3_1,gc4_1,gc5_1,gc6_1,gfc1_1,
                 gc1_2,gc2_2,gc3_2,gc4_2,gc5_2,gc6_2,gfc1_2,
                 gc1_3,gc2_3,gc3_3,gc4_3,gc5_3,gc6_3,gfc1_3,
                 gc1_4,gc2_4,gc3_4,gc4_4,gc5_4,gc6_4,gfc1_4,
                 gc1_5,gc2_5,gc3_5,gc4_5,gc5_5,gc6_5,gfc1_5]

        return masks

    def get_view_for(self,n,masks):
        #gc1_1, gc2_1, gc3_1, gc4_1, gc5_1, gc6_1, gfc1_1, gc1_2, gc2_2, gc3_2, gc4_2, gc5_2, gc6_2, gfc1_2, gc1_3, gc2_3, gc3_3, gc4_3, gc5_3, gc6_3, gfc1_3 =masks
        gc1_1, gc2_1, gc3_1, gc4_1, gc5_1, gc6_1, gfc1_1, gc1_2, gc2_2, gc3_2, gc4_2, gc5_2, gc6_2, gfc1_2, gc1_3, gc2_3, gc3_3, gc4_3, gc5_3, gc6_3, gfc1_3, \
        gc1_4, gc2_4, gc3_4, gc4_4, gc5_4, gc6_4, gfc1_4 , gc1_5, gc2_5, gc3_5, gc4_5, gc5_5, gc6_5, gfc1_5 =masks

        if n=='fc1_1.weight':
            post=gfc1_1.data.view(-1,1).expand_as(self.fc1_1.weight)
            pre=gc6_1.data.view(-1,1,1).expand((self.ec6_1.weight.size(1), self.smid, self.smid)).contiguous().view(1,-1).expand_as(self.fc1_1.weight)
            return torch.min(post,pre)
        elif n=='fc1_1.bias':
            return gfc1_1.data.view(-1)
        elif n=='c1_1.weight':
            return gc1_1.data.view(-1,1,1,1).expand_as(self.c1_1.weight)
        elif n=='c1_1.bias':
            return gc1_1.data.view(-1)
        elif n=='c2_1.weight':
            post=gc2_1.data.view(-1,1,1,1).expand_as(self.c2_1.weight)
            pre=gc1_1.data.view(1,-1,1,1).expand_as(self.c2_1.weight)
            return torch.min(post,pre)
        elif n=='c2_1.bias':
            return gc2_1.data.view(-1)
        elif n=='c3_1.weight':
            post=gc3_1.data.view(-1,1,1,1).expand_as(self.c3_1.weight)
            pre=gc2_1.data.view(1,-1,1,1).expand_as(self.c3_1.weight)
            return torch.min(post,pre)
        elif n=='c3_1.bias':
            return gc3_1.data.view(-1)
        elif n=='c4_1.weight':
            post=gc4_1.data.view(-1,1,1,1).expand_as(self.c4_1.weight)
            pre=gc3_1.data.view(1,-1,1,1).expand_as(self.c4_1.weight)
            return torch.min(post,pre)
        elif n=='c4_1.bias':
            return gc4_1.data.view(-1)
        elif n=='c5_1.weight':
            post=gc5_1.data.view(-1,1,1,1).expand_as(self.c5_1.weight)
            pre=gc4_1.data.view(1,-1,1,1).expand_as(self.c5_1.weight)
            return torch.min(post,pre)
        elif n=='c5_1.bias':
            return gc5_1.data.view(-1)
        elif n=='c6_1.weight':
            post=gc6_1.data.view(-1,1,1,1).expand_as(self.c6_1.weight)
            pre=gc5_1.data.view(1,-1,1,1).expand_as(self.c6_1.weight)
            return torch.min(post,pre)
        elif n=='c6_1.bias':
            return gc6_1.data.view(-1)

        elif n=='fc1_2.weight':
            post=gfc1_2.data.view(-1,1).expand_as(self.fc1_2.weight)
            pre=gc6_2.data.view(-1,1,1).expand((self.ec6_2.weight.size(1), self.smid, self.smid)).contiguous().view(1,-1).expand_as(self.fc1_2.weight)
            return torch.min(post,pre)
        elif n=='fc1_2.bias':
            return gfc1_2.data.view(-1)
        elif n=='c1_2.weight':
            return gc1_2.data.view(-1,1,1,1).expand_as(self.c1_2.weight)
        elif n=='c1_2.bias':
            return gc1_2.data.view(-1)
        elif n=='c2_2.weight':
            post=gc2_2.data.view(-1,1,1,1).expand_as(self.c2_2.weight)
            pre=gc1_2.data.view(1,-1,1,1).expand_as(self.c2_2.weight)
            return torch.min(post,pre)
        elif n=='c2_2.bias':
            return gc2_2.data.view(-1)
        elif n=='c3_2.weight':
            post=gc3_2.data.view(-1,1,1,1).expand_as(self.c3_2.weight)
            pre=gc2_2.data.view(1,-1,1,1).expand_as(self.c3_2.weight)
            return torch.min(post,pre)
        elif n=='c3_2.bias':
            return gc3_2.data.view(-1)
        elif n=='c4_2.weight':
            post=gc4_2.data.view(-1,1,1,1).expand_as(self.c4_2.weight)
            pre=gc3_2.data.view(1,-1,1,1).expand_as(self.c4_2.weight)
            return torch.min(post,pre)
        elif n=='c4_2.bias':
            return gc4_2.data.view(-1)
        elif n=='c5_2.weight':
            post=gc5_2.data.view(-1,1,1,1).expand_as(self.c5_2.weight)
            pre=gc4_2.data.view(1,-1,1,1).expand_as(self.c5_2.weight)
            return torch.min(post,pre)
        elif n=='c5_2.bias':
            return gc5_2.data.view(-1)
        elif n=='c6_2.weight':
            post=gc6_2.data.view(-1,1,1,1).expand_as(self.c6_2.weight)
            pre=gc5_2.data.view(1,-1,1,1).expand_as(self.c6_2.weight)
            return torch.min(post,pre)
        elif n=='c6_2.bias':
            return gc6_2.data.view(-1)

        elif n=='fc1_3.weight':
            post=gfc1_3.data.view(-1,1).expand_as(self.fc1_3.weight)
            pre=gc6_3.data.view(-1,1,1).expand((self.ec6_3.weight.size(1), self.smid, self.smid)).contiguous().view(1,-1).expand_as(self.fc1_3.weight)
            return torch.min(post,pre)
        elif n=='fc1_3.bias':
            return gfc1_3.data.view(-1)
        elif n=='c1_3.weight':
            return gc1_3.data.view(-1,1,1,1).expand_as(self.c1_3.weight)
        elif n=='c1_3.bias':
            return gc1_3.data.view(-1)
        elif n=='c2_3.weight':
            post=gc2_3.data.view(-1,1,1,1).expand_as(self.c2_3.weight)
            pre=gc1_3.data.view(1,-1,1,1).expand_as(self.c2_3.weight)
            return torch.min(post,pre)
        elif n=='c2_3.bias':
            return gc2_3.data.view(-1)
        elif n=='c3_3.weight':
            post=gc3_3.data.view(-1,1,1,1).expand_as(self.c3_3.weight)
            pre=gc2_3.data.view(1,-1,1,1).expand_as(self.c3_3.weight)
            return torch.min(post,pre)
        elif n=='c3_3.bias':
            return gc3_3.data.view(-1)
        elif n=='c4_3.weight':
            post=gc4_3.data.view(-1,1,1,1).expand_as(self.c4_3.weight)
            pre=gc3_3.data.view(1,-1,1,1).expand_as(self.c4_3.weight)
            return torch.min(post,pre)
        elif n=='c4_3.bias':
            return gc4_3.data.view(-1)
        elif n=='c5_3.weight':
            post=gc5_3.data.view(-1,1,1,1).expand_as(self.c5_3.weight)
            pre=gc4_3.data.view(1,-1,1,1).expand_as(self.c5_3.weight)
            return torch.min(post,pre)
        elif n=='c5_3.bias':
            return gc5_3.data.view(-1)
        elif n=='c6_3.weight':
            post=gc6_3.data.view(-1,1,1,1).expand_as(self.c6_3.weight)
            pre=gc5_3.data.view(1,-1,1,1).expand_as(self.c6_3.weight)
            return torch.min(post,pre)
        elif n=='c6_3.bias':
            return gc6_3.data.view(-1)


        elif n=='fc1_4.weight':
            post=gfc1_4.data.view(-1,1).expand_as(self.fc1_4.weight)
            pre=gc6_4.data.view(-1,1,1).expand((self.ec6_4.weight.size(1), self.smid, self.smid)).contiguous().view(1,-1).expand_as(self.fc1_4.weight)
            return torch.min(post,pre)
        elif n=='fc1_4.bias':
            return gfc1_4.data.view(-1)
        elif n=='c1_4.weight':
            return gc1_4.data.view(-1,1,1,1).expand_as(self.c1_4.weight)
        elif n=='c1_4.bias':
            return gc1_4.data.view(-1)
        elif n=='c2_4.weight':
            post=gc2_4.data.view(-1,1,1,1).expand_as(self.c2_4.weight)
            pre=gc1_4.data.view(1,-1,1,1).expand_as(self.c2_4.weight)
            return torch.min(post,pre)
        elif n=='c2_4.bias':
            return gc2_4.data.view(-1)
        elif n=='c3_4.weight':
            post=gc3_4.data.view(-1,1,1,1).expand_as(self.c3_4.weight)
            pre=gc2_4.data.view(1,-1,1,1).expand_as(self.c3_4.weight)
            return torch.min(post,pre)
        elif n=='c3_4.bias':
            return gc3_4.data.view(-1)
        elif n=='c4_4.weight':
            post=gc4_4.data.view(-1,1,1,1).expand_as(self.c4_4.weight)
            pre=gc3_4.data.view(1,-1,1,1).expand_as(self.c4_4.weight)
            return torch.min(post,pre)
        elif n=='c4_4.bias':
            return gc4_4.data.view(-1)
        elif n=='c5_4.weight':
            post=gc5_4.data.view(-1,1,1,1).expand_as(self.c5_4.weight)
            pre=gc4_4.data.view(1,-1,1,1).expand_as(self.c5_4.weight)
            return torch.min(post,pre)
        elif n=='c5_4.bias':
            return gc5_4.data.view(-1)
        elif n=='c6_4.weight':
            post=gc6_4.data.view(-1,1,1,1).expand_as(self.c6_4.weight)
            pre=gc5_4.data.view(1,-1,1,1).expand_as(self.c6_4.weight)
            return torch.min(post,pre)
        elif n=='c6_4.bias':
            return gc6_4.data.view(-1)


        elif n=='fc1_5.weight':
            post=gfc1_5.data.view(-1,1).expand_as(self.fc1_5.weight)
            pre=gc6_5.data.view(-1,1,1).expand((self.ec6_5.weight.size(1), self.smid, self.smid)).contiguous().view(1,-1).expand_as(self.fc1_5.weight)
            return torch.min(post,pre)
        elif n=='fc1_5.bias':
            return gfc1_5.data.view(-1)
        elif n=='c1_5.weight':
            return gc1_5.data.view(-1,1,1,1).expand_as(self.c1_5.weight)
        elif n=='c1_5.bias':
            return gc1_5.data.view(-1)
        elif n=='c2_5.weight':
            post=gc2_5.data.view(-1,1,1,1).expand_as(self.c2_5.weight)
            pre=gc1_5.data.view(1,-1,1,1).expand_as(self.c2_5.weight)
            return torch.min(post,pre)
        elif n=='c2_5.bias':
            return gc2_5.data.view(-1)
        elif n=='c3_5.weight':
            post=gc3_5.data.view(-1,1,1,1).expand_as(self.c3_5.weight)
            pre=gc2_5.data.view(1,-1,1,1).expand_as(self.c3_5.weight)
            return torch.min(post,pre)
        elif n=='c3_5.bias':
            return gc3_5.data.view(-1)
        elif n=='c4_5.weight':
            post=gc4_5.data.view(-1,1,1,1).expand_as(self.c4_5.weight)
            pre=gc3_5.data.view(1,-1,1,1).expand_as(self.c4_5.weight)
            return torch.min(post,pre)
        elif n=='c4_5.bias':
            return gc4_5.data.view(-1)
        elif n=='c5_5.weight':
            post=gc5_5.data.view(-1,1,1,1).expand_as(self.c5_5.weight)
            pre=gc4_5.data.view(1,-1,1,1).expand_as(self.c5_5.weight)
            return torch.min(post,pre)
        elif n=='c5_5.bias':
            return gc5_5.data.view(-1)
        elif n=='c6_5.weight':
            post=gc6_5.data.view(-1,1,1,1).expand_as(self.c6_5.weight)
            pre=gc5_5.data.view(1,-1,1,1).expand_as(self.c6_5.weight)
            return torch.min(post,pre)
        elif n=='c6_5.bias':
            return gc6_5.data.view(-1)

        return None

