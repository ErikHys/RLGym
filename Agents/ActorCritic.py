import torch


class Agent:

    def __init__(self, action_space, feature_space, gamma=0.99):
        self.actor = Actor(action_space, feature_space)
        self.critic = Critic(feature_space)
        self.log_probs = None
        self.gamma = gamma
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def to_tensor_and_device(self, x):
        return torch.tensor(x).to(self.device)

    def get_action(self, x):
        x = self.to_tensor_and_device(x)
        out = self.actor(x)
        probs = torch.distributions.Categorical(out)
        action = probs.sample()
        self.log_probs = probs.log_prob(action)
        return action, None

    def update(self, reward, new_state, old_state, done):
        new_state = self.to_tensor_and_device(new_state)
        old_state = self.to_tensor_and_device(old_state)

        delta = reward + self.gamma*self.critic(new_state)*(1-int(done))-self.critic(old_state)
        critic_loss = delta**2
        actor_loss = -self.log_probs * delta
        (actor_loss + critic_loss).backward()
        self.actor.opt.step()
        self.critic.opt.step()
        self.actor.opt.zero_grad()
        self.critic.opt.zero_grad()


class Actor(torch.nn.Module):

    def __init__(self, action_space, feature_space):
        super().__init__()
        self.in_layer = torch.nn.Linear(feature_space, 64)
        self.mid_layer = torch.nn.Linear(64, 64)
        self.out_layer = torch.nn.Linear(64, action_space)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=0)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        a1 = self.in_layer(x)
        z1 = self.relu(a1)
        a2 = self.mid_layer(z1)
        z2 = self.relu(a2)
        a3 = self.out_layer(z2)
        out = self.softmax(a3)
        return out


class Critic(torch.nn.Module):

    def __init__(self, feature_space):
        super().__init__()
        self.in_layer = torch.nn.Linear(feature_space, 64)
        self.mid_layer = torch.nn.Linear(64, 64)
        self.out_layer = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
        self.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        a1 = self.in_layer(x)
        z1 = self.relu(a1)
        a2 = self.mid_layer(z1)
        z2 = self.relu(a2)
        a3 = self.out_layer(z2)
        out = a3
        return out
