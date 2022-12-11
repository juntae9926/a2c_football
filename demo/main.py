
import torch
import torch.nn as nn
import numpy as np
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import torch.nn.functional as F
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=8, kernel_size=8, stride=4, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(1280+4, 256, bias=True)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.b_1 = nn.BatchNorm2d(4)
        self.b_2 = nn.BatchNorm2d(8)
        self.b_3 = nn.BatchNorm2d(16)
        self.b_4 = nn.BatchNorm1d(1280+4)
        self.b_5 = nn.BatchNorm1d(256)
        
    def forward(self, x, scalar):
        x = torch.tensor(x).float()  # normalize
        x = x.permute(0, 3, 1, 2).contiguous()  # 1 x channels x height x width
        x = self.b_1(x)
        x = self.relu(self.conv1(x))
        x = self.b_2(x)
        x = self.relu(self.conv2(x))
        x = self.b_3(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.b_4(torch.cat([x, scalar], 1))
        x = self.relu(self.fc1(x))
        x = self.b_5(x)
        x = self.fc2(x)
        return x
#Transform
w_step = 2/96.0
h_step = 0.84/72

def get_coordinates(arr):
    x, y = arr
    x_i = 0
    y_i = 0
    for i in range(1, 96):
        if x <-1 or x>1:
            if x<-1:
                x_i = 0
            else:
                x_i = 95
        else:
            if -1+ (i-1)*w_step <= x <= -1 + i*w_step:
                x_i = i
                break

    for i in range(1, 72):
        if y <-0.42 or y>0.42:
            if y<-0.42:
                y_i = 0
            else:
                y_i = 71
        else:
            if -0.42+ (i-1)*h_step <= y <= -0.42 + i*h_step:
                y_i = i
                break
    return [y_i, x_i]


def get_team_coordinates(team_arr):
    answ = []
    for j in range(len(team_arr)):
        answ.append(get_coordinates(team_arr[j]))
    return answ

def angle(src, tgt):
    dx = tgt[0] - src[0]
    dy = tgt[1] - src[1]
    theta = round(math.atan2(dx, -dy) * 180 / math.pi, 2)
    while theta < 0:
        theta += 360
    return theta

def direction(src, tgt):
    actions = [3, 4, 5, 
               6,  7, 
               8, 1, 2]
    theta = angle(src, tgt)
    index = int(((theta+45/2)%360)/45)
    return actions[index]


def create_obs(obs):
    ball_coord = get_coordinates(obs['ball'][:-1])
    left_team_coord = get_team_coordinates(obs['left_team'])
    right_team_coord = get_team_coordinates(obs['right_team'])
    player_coord =  get_coordinates(obs['left_team'][obs['active']])
    
    
    obs_1 = np.zeros(shape = (1, 72, 96, 4))
    
    obs_1[0, ball_coord[0], ball_coord[1], 0] = 1
            
    obs_1[0, player_coord[0], player_coord[1], 0] = 1
    
    for i, l in enumerate(left_team_coord):
        
        obs_1[0, l[0], l[1], 2] = 1

    for i, r in enumerate(right_team_coord):
        obs_1[0, r[0], r[1], 3] = 1

    ball_next_coord = get_coordinates([obs['ball'][0] + obs['ball_direction'][0], obs['ball'][1] + obs['ball_direction'][1]])

    left_team_next_coord = []
    for i in range(len(obs['left_team'])):
        left_team_next_coord.append([obs['left_team'][i][0] + obs['left_team_direction'][i][0], obs['left_team'][i][1] + obs['left_team_direction'][i][1]])
    
    right_team_next_coord = []
    for i in range(len(obs['right_team'])):
        right_team_next_coord.append([obs['right_team'][i][0] + obs['right_team_direction'][i][0], obs['right_team'][i][1] + obs['right_team_direction'][i][1]])
        
        
    scalar = np.zeros(shape = (1, 4))
    scalar[0,0] = obs['ball_owned_team']
    scalar[0,1] = obs['game_mode']
    scalar[0,2] = direction(obs['ball'][:-1], obs['ball_direction'][:-1])  
    scalar[0,3] = direction(obs['left_team'][obs['active']], obs['left_team_direction'][obs['active']])
        
    return obs_1, scalar

actor = ActorNetwork()
actor.load_state_dict(torch.load('actor.pth'))
actor = actor.float().to('cpu').eval()

def agent(obs):
    # Get observations for the first (and only one) player we control.
    obs = obs['players_raw'][0]
    # Agent we trained uses Super Mini Map (SMM) representation.
    # See https://github.com/google-research/seed_rl/blob/master/football/env.py for details.
    converted_obs = create_obs(obs)
    policy = actor(torch.as_tensor(converted_obs[0], dtype = torch.float32), torch.as_tensor(converted_obs[1], dtype = torch.float32))
    return [int(policy.argmax())]