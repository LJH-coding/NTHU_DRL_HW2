from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import importlib
import sys
from tqdm import tqdm
import numpy as np
import time
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

agent_path = "111000104_hw2_test.py"
module_name = agent_path.replace('/', '.').replace('.py', '')
spec = importlib.util.spec_from_file_location(module_name, agent_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
Agent = getattr(module, 'Agent')
agent = Agent()

total_reward = 0
total_time = 0
time_limit = 120
episodes = 3


for episode in tqdm(range(episodes), desc="Evaluating"):
    obs = env.reset()
    start_time = time.time()
    episode_reward = 0
    
    while True:
        action = agent.act(obs)
        
        start = time.time()
        obs, reward, done, info = env.step(action)
        end = time.time()
        episode_reward += reward
        env.render()

        if time.time() - start_time > time_limit:
            print(f"Time limit reached for episode {episode}")
            break

        if done:
            break

    end_time = time.time()
    total_reward += episode_reward
    total_time += (end_time - start_time)

env.close()

score = total_reward / episodes
print(f"Final Score: {score}")

