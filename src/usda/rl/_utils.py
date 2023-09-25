# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:49:58 2023

@author: richie bao
"""
import os
from matplotlib import animation
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import matplotlib
import torch

import warnings
warnings.filterwarnings('ignore')


def save_frames_as_gif4gym(frames, path='./', filename='gym_animation',figsize_scale=1,dpi=72,fps=60,interval=50):
    # Mess with this to change frame size
    fig=plt.figure(figsize=(frames[0].shape[1] / 72.0*figsize_scale, frames[0].shape[0] / 72.0*figsize_scale), dpi=dpi)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=interval)
    
    f=os.path.join(path,filename+'.gif')
    # writervideo = animation.FFMpegWriter(fps=fps)     
    anim.save(f,fps=fps)# writer=writervideo
    plt.close(fig)
    
    
def env_info_print(gym_env_name,action=None):
    env = gym.make(gym_env_name,render_mode="rgb_array")
    observation, info = env.reset()
    if action is None:
        action=env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Action Space Shape: {env.action_space}; Observation Space: {env.observation_space}")
    print(f'observation={observation}\nreward={reward},\nterminated={terminated}\ntruncated= {truncated}\ninfo={info}')

    return env    

def gym_env_steps_rgb_array(gym_env_name,steps=1000,**args):
    # Make gym env
    env = gym.make(gym_env_name, render_mode="rgb_array",**args)
    
    # Run the env
    observation, info = env.reset()
    frames = []
    for t in range(steps):
        # Render to frames buffer
        frames.append(env.render())
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            
    env.close()
    return frames

def custom_env_steps_rgb_array(env,steps=1000):   
    # Run the env
    observation, info = env.reset()
    frames = []
    for t in range(steps):
        # Render to frames buffer
        frames.append(env.render())
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            
    env.close()
    return frames

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=400,figsize=(4, 4)):
    fig = plt.figure(figsize=figsize)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = matplotlib.animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

def show_one_episode_QL(env,q_table, n_max_steps=200, seed=42,figsize=(4, 4)):
    frames = []
    np.random.seed(seed)
    new_state, info = env.reset(seed=seed)
    states_tabu=[]
    for step in range(n_max_steps):
        frames.append(env.render())
        if new_state in states_tabu:            
            action = env.action_space.sample()
        else:
            states_tabu.append(new_state)
            action = np.argmax(q_table[new_state][:])
        new_state, reward, terminated, truncated, info = env.step(action)
            
        if terminated or truncated:
            break
        #if step==10:break
    env.close()
    return plot_animation(frames,figsize=figsize)

