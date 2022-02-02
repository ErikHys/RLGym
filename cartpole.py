import gym
import torch
from matplotlib import pyplot as plt

from Agents import ActorCritic

agent = ActorCritic.Agent(2, 4)

env = gym.make("CartPole-v1")
observation = env.reset()
print(env.action_space)
improvements = []
exp_moving_avg = 0
returns = 0
ep = 0
for _ in range(1000000):
    old_obs = observation
    if ep > 500 == 0:
        env.render()
    action, action_value = agent.get_action(observation)
    # action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action.item())
    returns += reward
    if done:
        ep += 1
        improvements.append(returns)
        exp_moving_avg += 0.01*(returns - exp_moving_avg)
        agent.update(-10, observation, old_obs, done)
        if len(improvements) % 50 == 0:
            print(exp_moving_avg)
            print(action_value)
        returns = 0
        observation = env.reset()
        continue
    agent.update(torch.tensor(reward), observation, old_obs, done)
env.close()

plt.plot(improvements)
plt.show()