# %load_ext autoreload 
# %autoreload 2 
import usda.mpe_realworld  as usda_mpe
from usda.mpe_realworld.mpe import navona_v1
import usda.utils as usda_utils
import usda.rl as usda_rl   

import matplotlib.pyplot as plt
import mapclassify
import matplotlib
from IPython.display import HTML
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.ppo import MlpPolicy
import os
import rioxarray as rxr
import numpy as np

from pettingzoo.test import parallel_api_test
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

piazza_navona_shp_root=r'C:\Users\richie\omen_richiebao\omen_github\USDA_special_study\data\piazza_navona_osm'
area_node_fn=os.path.join(piazza_navona_shp_root,'navona_area_node.tif')
navona_area_node=rxr.open_rasterio(area_node_fn)
plat=np.stack(navona_area_node.data,axis=2)
plat[:,:,0][plat[:,:,0]==-9999]=15   
# plat=plat[50:600,50:600,:]
plat=plat[100:500,100:500:,:]

plat_colors=usda_utils.cmap2hex('Pastel1',16)
print(plat_colors)    
cmap, norm = matplotlib.colors.from_levels_and_colors([i+1 for i in list(plat_colors.keys())],list(plat_colors.values()),extend='max') 

plat_rewards={
    1:100,
    2:-100,
    3:-100,
    4:-100,
    5:-100,
    6:-100,
    7:-100,
    8:-100,
    9:-100,
    10:-100,
    11:0,
    12:-100,
    13:-100,
    14:-100,
    15:-10}

env=navona_v1.parallel_env(
    render_mode="rgb_array",
    plat=plat,
    plat_colors=plat_colors,
    plat_rewards=plat_rewards,
    agents_num=100,
    nodes_radius=50,
    group_dis=0.01,
    # continuous_actions=True,
    )

# parallel_api_test(env)  # Works
# check_env(env)  # Throws error

# env = ss.black_death_v3(env)
# n_actions = env.action_space.shape[-1]

#%%


env.reset() # seed=578
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=12, base_class="stable_baselines3")


# Save a checkpoint every 1000 steps
ckpt_root=r'C:\Users\richie\omen_richiebao\model_ckpts\navona_ckpts'
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path=os.path.join(ckpt_root,"logs"),
  name_prefix=os.path.join(ckpt_root,"navona_model"),
  save_replay_buffer=True,
  save_vecnormalize=True,
)

# model = PPO(
#     MlpPolicy,
#     env,
#     verbose=1,
#     learning_rate=1e-3,
#     batch_size=256, #256
#     )

model = A2C("MlpPolicy", env, verbose=1)

# The noise objects for DDPG
# n_actions=5
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

model.learn(total_timesteps=1e5,callback=checkpoint_callback) #3E5

navona_v1_PPO_path=r'C:\Users\richie\omen_richiebao\omen_temp\navona_v1_A2C.zip'
model.save(navona_v1_PPO_path)
