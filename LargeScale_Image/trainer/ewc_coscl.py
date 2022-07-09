from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm
import trainer

import networks


class KLD(nn.Module):
    def __init__(self):
        super(KLD, self).__init__()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        KLD_loss = 0
        for k in range(len(x)):
            for l in range(len(x)):
                if l != k:
                    KLD_loss += self.criterion_KLD(F.log_softmax(x[k], dim=1), F.softmax(x[l], dim=1).detach())

        return KLD_loss

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)
        
        self.lamb=args.lamb
        model.s_gate = args.s_gate

        self.lamb1 = args.lamb1
        self.lamb2 = args.lamb2
        self.lamb3 = args.lamb3

        self.kld = KLD()
        
    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

        
    def setup_training(self, lr):
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def train(self, train_loader, test_loader, t):
        
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t>0:
            #self.update_frozen_model()
            self.old_param = {}
            for n, p in self.model.named_parameters():
                self.old_param[n] = p.data.clone().detach()

            self.update_fisher()
        
        # Now, you can update self.t
        self.t = t
        
        kwargs = {'num_workers': 8, 'pin_memory': True}
        #kwargs = {'num_workers': 0, 'pin_memory': False}
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False, **kwargs)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True, **kwargs)
        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.cuda(), target.cuda()
                batch_size = data.shape[0]

                task = torch.autograd.Variable(torch.LongTensor([self.t]).cuda())
                output, outputs_expert, _ = self.model.forward(data, task, return_expert=True)#[t] #(data, t)
                output = output[t]
                loss_CE = self.criterion(output,target) + self.lamb1 * self.kld(outputs_expert)

                self.optimizer.zero_grad()
                (loss_CE).backward()
                self.optimizer.step()

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            valid_loss,valid_acc=self.evaluator.evaluate(self.model, self.test_iterator, t)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print()

        '''
        # Activations mask
        task=torch.autograd.Variable(torch.LongTensor([t]).cuda())
        mask=self.model.mask(task,s=self.model.s_gate)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        '''

    def criterion(self,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if self.t>0:
            #for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
            #    loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
            for name, param in self.model.named_parameters():
                loss_reg+=torch.sum(self.fisher[name]*(self.old_param[name] - param).pow(2))/2
        return self.ce(output,targets) + self.lamb*loss_reg
    
    def fisher_matrix_diag(self):
        # Init
        fisher={}
        for n,p in self.model.named_parameters():
            fisher[n]=0*p.data
        # Compute
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        for samples in tqdm(self.fisher_iterator):
            data, target = samples
            data, target = data.cuda(), target.cuda()

            # Forward and backward
            self.model.zero_grad()
            task = torch.autograd.Variable(torch.LongTensor([self.t]).cuda())
            outputs = self.model.forward(data, task)[self.t]
            loss=self.criterion(outputs, target)
            loss.backward()
            
            # Get gradients
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n]+=self.args.batch_size*p.grad.data.pow(2)
        # Mean
        with torch.no_grad():
            for n,_ in self.model.named_parameters():
                fisher[n]=fisher[n]/len(self.train_iterator)
        return fisher
        
    
    def update_fisher(self):
        if self.t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=self.fisher_matrix_diag()
        if self.t>0:
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*self.t)/(self.t+1)
                #self.fisher[n] = self.fisher[n] + fisher_old[n]
