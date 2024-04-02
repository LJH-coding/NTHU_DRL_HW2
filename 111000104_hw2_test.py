import torch.nn as nn
import torch
import numpy as np
import gym
from collections import deque


def GrayScaleObservation(observation):
    def permute_orientation(observation):
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def transform(obs):
        return np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])

    observation = permute_orientation(observation)
    observation = transform(observation)
    return observation

def ResizeObservation(image, new_height = 84, new_width = 84):
    """
    Resize an image using a simple area relation (similar to cv2.INTER_AREA) without OpenCV.
    
    Args:
        image (np.ndarray): Input image.
        new_height (int): Height of the resized image.
        new_width (int): Width of the resized image.
    
    Returns:
        np.ndarray: The resized image.
    """
    # Calculate the ratio of the old dimensions to new ones
    height_ratio = image.shape[0] / new_height
    width_ratio = image.shape[1] / new_width
    
    # Create an empty array for the new resized image
    resized_image = np.zeros((new_height, new_width, *image.shape[2:]), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the coordinates of the original pixels to be considered
            start_i = int(i * height_ratio)
            end_i = int((i + 1) * height_ratio)
            start_j = int(j * width_ratio)
            end_j = int((j + 1) * width_ratio)
            
            # Compute the average of the pixels within the block
            block = image[start_i:end_i, start_j:end_j]
            block_mean = block.mean(axis=(0, 1), dtype=np.float64)
            
            # Assign the mean value to the corresponding pixel in the new image
            resized_image[i, j] = block_mean
    
    # Handle the case for grayscale images (2D arrays)
    if len(image.shape) == 2:
        resized_image = resized_image.reshape((new_height, new_width))
    
    return resized_image

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
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        self.value_head = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
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
        self.frame_skipping = 0
        self.last_action = None
        self.frames = deque(maxlen=4)

    def act(self, observation):
        if self.frame_skipping % 4 == 0:
            observation = ResizeObservation(observation)
            observation = GrayScaleObservation(observation)
            while len(self.frames) < 4:
                self.frames.append(observation)
            self.frames.append(observation)
            observation_stack = np.stack(self.frames, axis=0)
            self.last_action = self.predict(observation_stack)
        self.frame_skipping += 1
        return self.last_action

    def predict(self, obs):
        with torch.no_grad():
            action = self.model(torch.Tensor(np.array(obs)).unsqueeze(0))
            action = torch.argmax(action[0])
        return action.item() + 1
    



