import torch
import torch.nn as nn
import torch.nn.functional as F

# PolicyNetwork
class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(72*96*4+4, 4096)
        self.fc2 = nn.Linear(4096, 256)
        self.fc3 = nn.Linear(256, 19)
        self.bn_1 = nn.BatchNorm1d(72*96*4 + 4)
        self.bn_2 = nn.BatchNorm1d(4096)
        self.bn_3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, scalar):
        x = torch.tensor(x).float()  # normalize
        x = x.permute(0, 3, 1, 2).contiguous()  # 1 x channels x height x width
        x = x.reshape(x.shape[0], -1)
        x = self.bn_1(torch.cat([x, scalar],1))
        x = self.relu(self.fc1(x))
        x = self.bn_2(x)
        x = self.relu(self.fc2(x))
        x = self.bn_3(x)
        x = self.fc3(x)
        return F.softmax(x, dim = -1)
   
# ValueNetwork  
class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(72*96*4 + 4, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(72*96*4 + 4)
        self.bn_2 = nn.BatchNorm1d(4096)
        self.bn_3 = nn.BatchNorm1d(1024)

    def forward(self, x, scalar):
        x = torch.tensor(x).float()  # normalize
        x = x.permute(0, 3, 1, 2).contiguous()  # 1 x channels x height x width
        x = x.reshape(x.shape[0], -1)
        x = self.bn_1(torch.cat([x, scalar],1))
        x = self.relu(self.fc1(x))
        x = self.bn_2(x)
        x = self.relu(self.fc2(x))
        x = self.bn_3(x)
        x = self.fc3(x)
        return x