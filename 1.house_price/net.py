import torch.nn as nn
import torch.nn.functional as F  

class housing_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.Linear(13,128)
        self.hidden2=nn.Linear(128,256)
        self.hidden3=nn.Linear(256,256)
        self.out=nn.Linear(256,1)
        self.drop=nn.Dropout(0.05)
    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=self.drop(x)
        x=F.relu(self.hidden2(x))
        x=self.drop(x)
        x=F.relu(self.hidden3(x))
        x=self.drop(x)
        x=self.out(x)
        x=x.squeeze(-1)
        return x
