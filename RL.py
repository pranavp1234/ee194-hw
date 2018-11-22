import gym
import math
import numpy as np
import random

def flipCoin( p ):
    r = random.random()
    return r < p

env = gym.make('CartPole-v0')

# Q = np.zeros((2,960,840))
Q = np.zeros((2,6,9,3,4))

c = 10

import pdb; pdb.set_trace()

actions = [0,1]

epsilon=0.01

alpha=0.1

discount = 0.9

i=0

obs = env.reset()
running_reward = 0
for _ in range(300000):
    env.render()
    if round(obs[0],2) <= obs[0]:
        index_1 = round(obs[0]+2.4,2)
        index_1 = int(round(index_1,0))
    else:
        index_1 = round(obs[0]+2.4,2)
        index_1 = int(round(index_1,0))
    if obs[3] <= -np.radians(16.67):
        # index_1 = (round(obs[0]+2.4,2)/0.01)-1
        index_3 = 0
    elif obs[3] >= -np.radians(16.67):
        index_3 = 2
    else:
        index_3 = 1
    if obs[1] <= -1:
        # index_1 = (round(obs[0]+2.4,2)/0.01)-1
        index_4 = 0
    elif obs[1] <= -0:
        index_4 = 1
    elif obs[1] <= 1:
        index_4 = 2
    else:
        index_4 = 3
    # degr = (360/(2*math.pi)) * obs[2]
    if round(obs[2],2) <= obs[2]:
        index_2 = (round(obs[2]+0.42,3)/0.1)
        index_2 = int(round(index_2,0))
    else:
        index_2 = round(obs[2]+0.42,3)/0.1
        index_2 = int(round(index_2,0))
    q_vals = [Q[act,index_1,index_2,index_3,index_4] for act in actions]
    actionerino = q_vals.index(max(q_vals))
    if flipCoin(epsilon):
        actionerino = random.choice(actions)
    obs, reward, done, info = env.step(actionerino) # take a random action
    running_reward += reward
    if obs[3] <= -np.radians(16.67):
        obs_index_3 = 0
    elif obs[3] >= -np.radians(16.67):
        obs_index_3 = 2
    else:
        obs_index_3 = 1
    # degr = (360/(2*math.pi)) * obs[2]
    if round(obs[2],2) <= obs[2]:
        obs_index_2 = (round(obs[2]+0.42,3)/0.1)
        obs_index_2 = int(round(obs_index_2,0))
    else:
        obs_index_2 = round(obs[2]+0.42,3)/0.1
        obs_index_2 = int(round(obs_index_2,0))
    if round(obs[0],2) <= obs[0]:
        obs_index_1 = round(obs[0]+2.4,2)
        obs_index_1 = int(round(obs_index_1,0))
    else:
        obs_index_1 = round(obs[0]+2.4,2)
        obs_index_1 = int(round(obs_index_1,0))
    if obs[1] <= -1:
        # index_1 = (round(obs[0]+2.4,2)/0.01)-1
        obs_index_4 = 0
    elif obs[1] <= -0:
        obs_index_4 = 1
    elif obs[1] <= 1:
        obs_index_4 = 2
    else:
        obs_index_4 = 3
    # print(obs)
    # import pdb; pdb.set_trace()
    obs_q_vals = [Q[act,obs_index_1,obs_index_2,obs_index_3,obs_index_4] for act in actions]
    # obs_actionerino = obs_q_vals.index(max(obs_q_vals))
    Q[actionerino,index_1,index_2,index_3,index_4] = (alpha*running_reward) + ((1-alpha)*Q[actionerino,index_1,index_2,index_3,index_4]) + (alpha*discount*(max(obs_q_vals))) - ((c)*abs(obs[0])) - ((c)*abs(obs[3]))
    # import pdb; pdb.set_trace()
    if done == True:
        # epsilon *= 0.99
        epsilon = max(epsilon, min(1, 1.0 - np.log10((i + 1) / 25)))
        discount = max(discount, min(0.5, 1.0 - np.log10((i + 1) / 25)))
        Q[actionerino,index_1,index_2,index_3] = -100
        print(running_reward)
        running_reward = 0
        print("reset " + str(i))
        i += 1
        env.reset()
print(Q)
# print("TEST")
# print("TIME")
import pdb; pdb.set_trace()
#
obs = env.reset()
# while(done is not True):
running_reward = 0
while(True):
    env.render()
    index_1 = round(obs[0]+2.4, 2)
    index_1 = int(round(index_1, 0))
    index_2 = (round(obs[2]+0.42, 3) / 0.1)
    index_2 = int(round(index_2, 0))
    if obs[3] <= -np.radians(16.67):
        # index_1 = (round(obs[0]+2.4,2)/0.01)-1
        index_3 = 0
    elif obs[3] >= -np.radians(16.67):
        index_3 = 2
    else:
        index_3 = 1
    if obs[1] <= -1:
        # index_1 = (round(obs[0]+2.4,2)/0.01)-1
        index_4 = 0
    elif obs[1] <= -0:
        index_4 = 1
    elif obs[1] <= 1:
        index_4 = 2
    else:
        index_4 = 3
    q_vals = [Q[act,index_1,index_2,index_3,index_4] for act in actions]
    actionerino = q_vals.index(max(q_vals))
    obs, reward, done, info = env.step(actionerino)
    running_reward += reward
    if done == True:
        print(running_reward)
        running_reward = 0
        print("reset " + str(i))
        i += 1
        env.reset()
