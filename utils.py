import math
import numpy as np

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