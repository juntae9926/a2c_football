import torch
import torch.nn as nn
import numpy as np
import gfootball.env as football_env
from gfootball.env import observation_preprocessing

from .model import *
from .env import agent
from .utils import *

env = football_env.create_environment(
   env_name="11_vs_11_kaggle",
   representation='raw',
   stacked=False,
   logdir='.',
   write_goal_dumps=False,
   write_full_episode_dumps=False,
   render=False,
   number_of_left_players_agent_controls=1,
   dump_frequency=0)

obs = env.reset()

created_obs = create_obs(obs[0])
print(created_obs[0].shape)

actor = Actor()
critic = Critic()

env  = football_env.create_environment(
   env_name="11_vs_11_kaggle",
   representation='raw',
   stacked=False,
   logdir='.',
   write_goal_dumps=False,
   write_full_episode_dumps=False,
   render=False,
   number_of_left_players_agent_controls=1,
   number_of_right_players_agent_controls=1,
   dump_frequency=0)

obs = env.reset()



adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99


step_done = 0
rewards_for_plot = []
for steps_done in range(64):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    games_play = 0
    wins = 0
    loses = 0
    obs = env.reset()
    values = []
    log_probs = []
    done = False
    while not done:

        converted_obs = create_obs(obs[0])
        actor.eval()
        prob = actor(torch.as_tensor(converted_obs[0], dtype = torch.float32), torch.as_tensor(converted_obs[1], dtype = torch.float32))
        actor.train()
        dist = torch.distributions.Categorical(probs = prob)
        act = dist.sample()


        new_obs, reward, done, _ = env.step([act.detach().data.numpy()[0], (agent(obs[1])).value])
        if reward[0]==-1:
            loses+=1
            done = True
        if reward[0] == 1:
            wins+=1
            done = True
        if reward[0]==0 and done:
            reward[0] = 0.25
            
        last_q_val = 0
        if done:
            converted_next_obs = create_obs(new_obs[0])
            critic.eval()
            last_q_val = critic(torch.as_tensor(converted_next_obs[0], dtype = torch.float32), torch.as_tensor(converted_next_obs[1], dtype = torch.float32))
            last_q_val = last_q_val.detach().data.numpy()
            critic.train()

        states.append(obs[0])
        action_arr = np.zeros(19)
        action_arr[act] = 1
        actions.append(action_arr)
        rewards.append(reward[0])
        next_states.append(new_obs[0])
        dones.append(1 - int(done))

        obs = new_obs
        if done:
            obs = env.reset()
            break
            
    rewards = np.array(rewards)
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    dones = np.array(dones)
    
    print('epoch '+ str(steps_done)+ '\t' +'reward_mean ' + str(np.mean(rewards)) + '\t' + 'games_count ' + str(games_play) + '\t' + 'total_wins ' + str(wins) + '\t'+ 'total_loses ' + str(loses))
    rewards_for_plot.append(np.mean(rewards))
    #train
    q_vals = np.zeros((len(rewards), 1))
    for i in range(len(rewards)-1, 0, -1):
        last_q_val = rewards[i] + dones[i]*gamma*last_q_val
        q_vals[i] = last_q_val

    action_tensor = torch.as_tensor(actions, dtype=torch.float32)

    obs_playgraund_tensor = torch.as_tensor(np.array([create_obs(states[i])[0][0] for i in range(len(rewards))]), dtype=torch.float32)

    obs_scalar_tensor = torch.as_tensor(np.array([create_obs(states[i])[1][0] for i in range(len(rewards))]), dtype=torch.float32)

    val = critic(obs_playgraund_tensor, obs_scalar_tensor)
    
    probs = actor(obs_playgraund_tensor, obs_scalar_tensor)
    
    advantage = torch.Tensor(q_vals) - val
    
    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()
    
    
    actor_loss = (-torch.log(probs)*advantage.detach()).mean()
    adam_actor.zero_grad()
    actor_loss.backward(retain_graph=True)
    adam_actor.step()
    
    

#         soft_update(actor, target_actor, 0.8)
#         soft_update(critic, target_critic, 0.8)
        
    if steps_done!=0 and steps_done%50 == 0:
        torch.save(actor.state_dict(), 'actor.pth')

        torch.save(critic.state_dict(), 'critic.pth')