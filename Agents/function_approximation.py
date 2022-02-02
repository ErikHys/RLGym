import torch
import numpy as np


class SemiGradientSarsa:

    def __init__(self):
        self.approximator_nn = SimpleNN()
        self.opt = torch.optim.Adam(self.approximator_nn.parameters(), lr=0.0001)

    def get_action(self, x):
        q_a1 = torch.tensor(x).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        action_value = self.approximator_nn(q_a1)
        return torch.argmax(action_value) if torch.rand(1)[0] <= 0.5 else torch.randint(2, (1,))[0], action_value

    def update(self, reward, state, old_action, old_action_value):
        action, action_value = self.get_action(state)
        loss = -(reward + 0.99*action_value[action] - old_action_value[old_action])
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

    def final_update(self, reward, old_action, old_action_value):
        loss = -(reward - old_action_value[old_action])
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()


class SimpleNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.in_layer = torch.nn.Linear(4, 64)
        self.mid_layer = torch.nn.Linear(64, 64)
        self.out_layer = torch.nn.Linear(64, 2)
        self.relu = torch.nn.ReLU()
        self.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        a1 = self.in_layer(x)
        z1 = self.relu(a1)
        a2 = self.mid_layer(z1)
        z2 = self.relu(a2)
        a3 = self.out_layer(z2)
        out = a3
        return out



