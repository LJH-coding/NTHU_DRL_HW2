import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque
import cv2

def GrayScaleObservation(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    return obs

def ResizeObservation(obs):
    shape = (84, 84)
    obs = cv2.resize(
            obs, shape[::-1], interpolation=cv2.INTER_AREA
        )
    if obs.ndim == 2:
        obs = np.expand_dims(obs, -1)
    return obs

class NoisyLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return F.linear(x, self.weight_mu, self.bias_mu)

class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.advantage_head = nn.Sequential(
            NoisyLinear(2304, 512),
            nn.ReLU(),
            NoisyLinear(512, 7)
        )

        self.value_head = nn.Sequential(
            NoisyLinear(2304, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        adv = self.advantage_head(x)
        val = self.value_head(x)
        q_values = val + adv - adv.mean(1, keepdim=True)
        return q_values

class Agent(object):
    def __init__(self):
        self.model = DeepQNetwork()
        self.model.load_state_dict(torch.load('./111000104_hw2_data', map_location=torch.device('cpu')))
        self.timesteps = 0
        self.frame_skipping = 0
        self.last_action = None
        self.frames = deque(maxlen=4)
        np.random.seed(98207403)

    def reset(self):
        self.timesteps = 0
        self.frame_skipping = 0
        self.last_action = None
        self.frames = deque(maxlen=4)
        np.random.seed(98207403)

    def act(self, obs):
        if self.timesteps == 4327:
            self.reset()
        if self.frame_skipping % 4 == 0:
            obs = ResizeObservation(obs)
            obs = GrayScaleObservation(obs)
            while len(self.frames) < 4:
                self.frames.append(obs)
            self.frames.append(obs)
            obs_stack = np.stack(self.frames, axis=0)
            self.last_action = self.predict(obs_stack)
        self.frame_skipping += 1
        self.timesteps += 1
        return self.last_action

    def predict(self, obs):
        if np.random.random() < 0.01:
            action = torch.tensor(np.random.randint(7))
        else:
            with torch.no_grad():
                action = self.model(torch.Tensor(np.array(obs)).unsqueeze(0))
                action = torch.argmax(action[0])
        return action.item()
    



