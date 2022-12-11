import torch
import torch.nn as nn
import numpy as np
import gfootball.env as football_env
from gfootball.env import observation_preprocessing

from model import *
from env import agent
from utils import *

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    env = football_env.create_environment(
    env_name="11_vs_11_kaggle",
    representation='raw',
    stacked=False,
    logdir='.',
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False,
    number_of_right_players_agent_controls=1,
    number_of_left_players_agent_controls=1,
    dump_frequency=0)

    actor = ActorNetwork()
    critic = CriticNetwork()
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=5e-4)
    gamma = 0.99
    # batch_size = 32
    epochs = 3000
    rewards_for_plot = []

    # memory = ReplayBuffer(buffer_size=20000, batch_size=32, device=device)

    state = env.reset()

    for episode in range(epochs):
        state = env.reset()
        states, actions, rewards, next_states, dones = [], [], [] ,[], []
        done, win = False, False

        while not done:

            converted_state = create_obs(state[0])
            with torch.no_grad():
                actor.eval()
                prob = actor(torch.as_tensor(converted_state[0], dtype = torch.float32), torch.as_tensor(converted_state[1], dtype = torch.float32))
                dist = torch.distributions.Categorical(probs = prob)
                action = dist.sample()
            next_state, reward, done, _ = env.step([action.detach().data.numpy()[0], agent(state[1]).value])
            
            if reward[0] == 0 and done == True:
                reward[0] = 0.25
            elif (reward[0] == -1):
                done = True
            elif (reward[0] == 1):
                done = True
                win == True

            q_value = 0
            if done:
                with torch.no_grad():
                    critic.eval()
                    converted_next_state = create_obs(next_state[0])
                    q_value = critic(torch.as_tensor(converted_next_state[0], dtype = torch.float32), torch.as_tensor(converted_next_state[1], dtype = torch.float32))
                    q_value = q_value.detach().data.numpy()

            states.append(state[0])
            action_ = np.zeros(19)
            action_[action] = 1
            actions.append(action_)
            rewards.append(reward[0])
            next_states.append(next_state[0])
            dones.append(1 - int(done))

            if done:
                state = env.reset()
                break

            state = next_state

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Train
        print(f'episode {str(episode)}' + '\t' + f'reward_mean {str(np.mean(rewards))}'  + '\t' f"win=={win}")
        rewards_for_plot.append(np.mean(rewards))
        q_vals = np.zeros((len(rewards), 1))
        for i in range(len(rewards)-1, 0, -1):
            q_value = rewards[i] + dones[i]*gamma*q_value
            q_vals[i] = q_value

        obs_playgraund_tensor = torch.as_tensor(np.array([create_obs(states[i])[0][0] for i in range(len(rewards))]), dtype=torch.float32)
        obs_scalar_tensor = torch.as_tensor(np.array([create_obs(states[i])[1][0] for i in range(len(rewards))]), dtype=torch.float32)

        val = critic(obs_playgraund_tensor, obs_scalar_tensor)
        probs = actor(obs_playgraund_tensor, obs_scalar_tensor)
        advantage = torch.Tensor(q_vals) - val

        actor_loss = (-torch.log(probs)*advantage.detach()).mean()
        actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_optim.step()
        
        critic_loss = advantage.pow(2).mean()
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
        
            
        if (episode+1)%50 == 0:
            torch.save(actor.state_dict(), 'actor.pth')
            torch.save(critic.state_dict(), 'critic.pth')

if __name__ == "__main__":
    
    main()