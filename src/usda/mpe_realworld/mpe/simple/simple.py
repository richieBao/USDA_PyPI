# noqa: D212, D415
"""
Created on Sat Oct  7 14:31:25 2023

@author: richie bao

ref: PettingZoo https://pettingzoo.farama.org/environments/mpe/simple/#

# Simple

```{figure} mpe_simple.gif
:width: 140px
:name: simple
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_v3` |
|--------------------|----------------------------------------|
| Actions            | Discrete/Continuous                    |
| Parallel API       | Yes                                    |
| Manual Control     | No                                     |
| Agents             | `agents= [agent_0]`                    |
| Agents             | 1                                      |
| Action Shape       | (5)                                    |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5,))        |
| Observation Shape  | (4)                                    |
| Observation Values | (-inf,inf)                             |
| State Shape        | (4,)                                   |
| State Values       | (-inf,inf)                             |


In this environment a single agent sees a landmark position and is rewarded based on how close it gets to the landmark (Euclidean distance). This is not a multiagent environment, and is primarily intended for debugging purposes.

Observation space: `[self_vel, landmark_rel_position]`

### Arguments

``` python
simple_v3.env(max_cycles=25, continuous_actions=False)
```



`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""
import sys
sys.path += ['..']

import numpy as np
from gymnasium.utils import EzPickle

# from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
# from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

if __package__:
    from .._mpe_utils.core import Agent, Landmark, World
    from .._mpe_utils.pos_coordi_transform import p_pos2plat_coordi,plat_coordi2p_pos,array2d_idxes,plat_region
    from .._mpe_utils.simple_env import SimpleEnv, make_env
else:
    from _mpe_utils.core import Agent, Landmark, World
    from _mpe_utils.pos_coordi_transform import p_pos2plat_coordi,plat_coordi2p_pos,array2d_idxes,plat_region
    from _mpe_utils.simple_env import SimpleEnv, make_env

# if __package__:
#     from .._mpe_utils.core import Agent, Landmark, World
#     from .._mpe_utils.scenario import BaseScenario
#     from .._mpe_utils.simple_env import SimpleEnv, make_env
#     from usda.mpe_realworld.utils.conversions import parallel_wrapper_fn
#     from .._mpe_utils.pos_coordi_transform import p_pos2plat_coordi,plat_coordi2p_pos,array2d_idxes,plat_region
# else:
#     from _mpe_utils.core import Agent, Landmark, World
#     from _mpe_utils.scenario import BaseScenario
#     from _mpe_utils.simple_env import SimpleEnv, make_env
#     from usda.mpe_realworld.utils.conversions import parallel_wrapper_fn
#     from _mpe_utils.pos_coordi_transform import p_pos2plat_coordi,plat_coordi2p_pos,array2d_idxes,plat_region


class raw_env(SimpleEnv, EzPickle):
    def __init__(self, 
                 max_cycles=25, 
                 continuous_actions=False, 
                 render_mode=None,
                 plat=None,
                 plat_colors=None,      
                 plat_rewards=None,
                 ):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,            
            render_mode=render_mode,     
            plat=plat,
            plat_colors=plat_colors,      
            plat_rewards=plat_rewards,     
        )
        scenario = Scenario()
        world = scenario.make_world(plat,plat_rewards,plat_colors)
        # print(world.plat,'\n',world.plat_rewards,'\n',world.plat_colors)        
        # print('###!@@@~~~',plat,render_mode)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            plat=plat,
            plat_colors=plat_colors,
            plat_rewards=plat_rewards,
        )
        self.metadata["name"] = "simple_realworld"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self,plat=None,plat_rewards=None,plat_colors=None):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            
        if plat is not None:
            world.plat=plat
            world.plat_width,world.plat_height=plat.shape
        if plat_rewards is not None:
            world.plat_rewards=plat_rewards
        if plat_colors is not None:
            world.plat_colors=plat_colors
            
        # print(world.plat,'\n',world.plat_rewards,'\n',world.plat_colors)
        
        
        return world

    def reset_world(self, world, np_random):
        # print('###!',world.plat)
        height=world.plat.shape[1]
        width=world.plat.shape[0]         
        
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            np.random.seed()
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)                 
            # print(agent.state.p_pos)
            agent.state.p_vel = np.zeros(world.dim_p)
            # print(agent.state.p_vel)
            agent.state.c = np.zeros(world.dim_c)
            agent.size = 0.0075
            
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            np.random.seed()
            # print('#',np.random.choice([0,1,2,3,4,5,6,7,8,9],1))
            # print(np.random.choice(plat_region(world.plat)[1],1))
            rdn_idx_1d=np.random.choice(plat_region(world.plat)[1],1)[0]
            # print(rdn_idx_1d)
            rdn_idx_2d=np.unravel_index(rdn_idx_1d, world.plat.shape)
            landmark.state.p_pos =plat_coordi2p_pos(rdn_idx_2d,height,width)
            # print(rdn_idx_2d)
            
            # print('#1',landmark.state.p_pos)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.size = 0.0075
        # print(world.plat)
        # print(world.plat_rewards)

    def reward(self, agent, world):
        pos=agent.state.p_pos
        pos=np.clip(pos,-1+0.05,+1-0.05)
        # print(pos)
        idx_2d=p_pos2plat_coordi(pos,world.plat_width,world.plat_height)   
        # print(idx_2d)
        idx_1d=np.ravel_multi_index(idx_2d,(world.plat_height,world.plat_width))
        pos_region=[k  for k,v in plat_region(world.plat).items() if idx_1d in v][0]
        plat_reward=world.plat_rewards[pos_region]/abs(max(world.plat_rewards.values(),key=abs))
        # print(plat_reward)
        
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        # print(-dist2+plat_reward)
        return -dist2+plat_reward

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)   
    
    
if __name__=="__main__":
    from usda import datasets as usda_datasets
    import usda.rl as usda_rl     
    import matplotlib.pyplot as plt
    import mapclassify
    import matplotlib
    
    size=200
    X_,_=usda_datasets.generate_categorical_2darray(size=size,sigma=7,seed=77)
    X=X_[0].reshape(size,size)*size
    X_BoxPlot=mapclassify.BoxPlot(X)
    y=X_BoxPlot.yb.reshape(size,size)
    y=y[:100,:]
    
    levels = list(range(1,10))
    clrs = ['#FFFFFF','#005ce6', '#3f8f76', '#ffffbe', '#3a5b0d', '#aaff00', '#e1e1e1','#F44336','#eeeee4']    
    clrs_dict={1:'#FFFFFF',2:'#005ce6',3:'#3f8f76',4:'#ffffbe'}
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, clrs,extend='max')       
    
    fig=plt.figure(figsize=(10,10))
    plt.imshow(y,cmap=cmap,norm=norm)    
    plt.show()
            
    
    env=raw_env(render_mode="rgb_array",
                plat=y,
                plat_colors=clrs_dict,
                plat_rewards={1:-2,2:-5,3:0,4:1},
                )
    env.reset() # seed=578
    
    frames=[]
    for agent in env.agent_iter():
        frames.append(env.render())
        observation, reward, termination, truncation, info = env.last()
        # print(observation, reward, termination, truncation, info)        
    
        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
    
        env.step(action)
    env.close()    
    
    anim=usda_rl.plot_animation(frames)
    f=r'C:\Users\richie\omen_richiebao\omen_temp/temp.gif'
    anim.save(f,fps=100)
    
    fig=plt.figure(figsize=(10,10))
    plt.imshow(frames[0],cmap=cmap,norm=norm)    
    plt.show()
    
    # import supersuit as ss
    # from stable_baselines3 import PPO
    # from stable_baselines3.ppo import MlpPolicy
    # import pygame    
    
    # env.reset() 
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 8, num_cpus=num_cpus, base_class="stable_baselines3")    