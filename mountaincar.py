import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from Agents import ActorCriticContinious

agent = ActorCriticContinious.Agent(2)

env = gym.make('MountainCarContinuous-v0')
observation = env.reset()
print(env.action_space)
print(env.observation_space)
improvements = []
improvements_steps = []
exp_moving_avg = 0
returns = 0
ep = 0
steps = 0
n_step = 1
n_values = []
while ep < 100:
    old_obs = observation
    if ep > 0:
        # pass
        env.render()
    action, action_value = agent.get_action(observation)
    action = [action]
    observation, reward, done, info = env.step(action)
    returns += reward
    steps += 1
    n_values.append([torch.tensor(-1), observation, old_obs, done])
    if done:
        ep += 1
        n_values[-1][0] = torch.tensor(reward)
        improvements.append(returns)
        improvements_steps.append(steps)
        exp_moving_avg += 0.01*(returns - exp_moving_avg)
        agent.update(reward, observation, old_obs, done, graph=True)
        n_values = []
        if len(improvements) % 1 == 0:
            print(steps)
        returns = 0
        steps = 0
        observation = env.reset()
        continue
    if steps > n_step:
        g = 0
        for i in range(len(n_values)):
            g += n_values[i][0]*0.99**i

        agent.update(g, observation, n_values[0][2], done)
        n_values.pop(0)
env.close()

plt.plot(improvements)
plt.plot(improvements_steps)
plt.show()
