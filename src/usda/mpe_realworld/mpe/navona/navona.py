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
import usda.utils as usda_utils
from scipy.stats import poisson

if __package__:
    from .._mpe_utils.core_navona import Agent, Landmark, World
    from .._mpe_utils.pos_coordi_transform import p_pos2plat_coordi,plat_coordi2p_pos,array2d_idxes,plat_region
    from .._mpe_utils.simple_env_navona import SimpleEnv, make_env
else:
    from _mpe_utils.core_navona import Agent, Landmark, World
    from _mpe_utils.pos_coordi_transform import p_pos2plat_coordi,plat_coordi2p_pos,array2d_idxes,plat_region
    from _mpe_utils.simple_env_navona import SimpleEnv, make_env

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
                 max_cycles=100, #25, 
                 continuous_actions=False, 
                 render_mode=None,
                 plat=None,
                 plat_colors=None,      
                 plat_rewards=None,
                 agents_num=3,
                 nodes_radius=30,
                 group_dis=0.5,
                 ):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,            
            render_mode=render_mode,     
            plat=plat,
            plat_colors=plat_colors,      
            plat_rewards=plat_rewards,   
            agents_num=agents_num,
            nodes_radius=nodes_radius,
            group_dis=group_dis,
        )
        scenario = Scenario()
        world = scenario.make_world(plat,plat_rewards,plat_colors,agents_num)
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
            nodes_radius=nodes_radius,
            group_dis=group_dis,
        )
        self.metadata["name"] = "simple_realworld"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self,plat=None,plat_rewards=None,plat_colors=None,agents_num=3):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(agents_num)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
        # add landmarks
        
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            
        if plat is not None:
            world.plat=plat       
            world.plat_width,world.plat_height=plat.shape[:2]
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
        # world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            # np.random.seed()
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)                 
            # print(agent.state.p_pos)
            agent.state.p_vel = np.zeros(world.dim_p)
            # print(agent.state.p_vel)
            agent.state.c = np.zeros(world.dim_c)
            agent.size = 0.0075
        
        plat_allowed=[1,11,15]
        # lc=world.plat[:,:,0]
        for i, landmark in enumerate(world.landmarks):    
            lc_idx=plat_allowed[i]
            # lc_idx_2d=np.stack(np.where(lc==lc_idx),axis=1)
            # rand_idx=np.random.randint(lc_idx_2d.shape[0], size=1)
            # lc_idx_2d=np.where(world.plat==lc)
            # print('#',i)
            # print(lc_idx_2d)
            # landmark_idx_2d=lc_idx_2d[rand_idx,:][0]
            # print(landmark_idx_2d)
            rdn_idx_1d=np.random.choice(plat_region(world.plat[:,:,0])[lc_idx],1)[0]
            rdn_idx_2d=np.unravel_index(rdn_idx_1d, world.plat.shape[:2])
            # print(rdn_idx_2d)
            landmark.state.p_pos =plat_coordi2p_pos(rdn_idx_2d,height,width)
            # print(landmark.state.p_pos)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.size = 0.02
            
            
            
        # for i, landmark in enumerate(world.landmarks):
        #     # landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
        #     np.random.seed()
        #     # print('#',np.random.choice([0,1,2,3,4,5,6,7,8,9],1))
        #     # print(np.random.choice(plat_region(world.plat)[1],1))
        #     rdn_idx_1d=np.random.choice(plat_region(world.plat[:,:,0])[1],1)[0]
        #     # print(rdn_idx_1d)
        #     rdn_idx_2d=np.unravel_index(rdn_idx_1d, world.plat.shape[:2])
        #     landmark.state.p_pos =plat_coordi2p_pos(rdn_idx_2d,height,width)
        #     # print(rdn_idx_2d)
            
        #     # print('#1',landmark.state.p_pos)
        #     landmark.state.p_vel = np.zeros(world.dim_p)
        #     landmark.size = 0.0075
        # print(world.plat)
        # print(world.plat_rewards)
        
    def agent_pos_idx_1d(self,agent,world):
        pos=agent.state.p_pos
        pos=np.clip(pos,-1+0.05,+1-0.05)
        # print(pos)
        idx_2d=p_pos2plat_coordi(pos,world.plat_width,world.plat_height)   
        # print(idx_2d)
        idx_1d=np.ravel_multi_index(idx_2d,(world.plat_height,world.plat_width))      
        return idx_1d,idx_2d
    
    def plat_area_rewards(self,idx_1d,world):
        pos_region=[k  for k,v in plat_region(world.plat[:,:,0]).items() if idx_1d in v][0]
        area_reward=world.plat_rewards[pos_region]#/abs(max(world.plat_rewards.values(),key=abs))  
        # print(area_reward)
        return area_reward
    
    def plat_node_rewards(self,idx_2d,world):
        # print(idx_2d)
        nodes=world.plat[:,:,1]
        # print('###',nodes.shape)
        row,col=idx_2d
        # print(row,col)
        # col,row=idx_2d
        nbr_xy=usda_utils.grid_neighbors(nodes,row,col,r=world.nodes_radius)
        # print(nbr_xy.shape)
        
        node_vals=np.array([nodes[*i] for i in nbr_xy])
        node_vals=node_vals[node_vals!=-9999]
        # print( len(node_vals))
        return len(node_vals)*10
        
    def agents_group_rewards(self,agent,world):
        # print('#',len(world.agents),agent.state.p_pos)
        dis=[np.sqrt(np.sum(np.square(agent.state.p_pos-a.state.p_pos))) for a in world.agents if a is not agent]
        # print('###',dis)
        dis_buffer=[i for i in dis if i<world.group_dis] #0.016
        pmf=poisson.pmf(len(dis_buffer), 4)
        # print(len(dis),len(dis_buffer))
        # print(pmf)
        return pmf*1000


    def reward(self, agent, world):
        # print('---?')
        idx_1d,idx_2d=self.agent_pos_idx_1d(agent, world)
        area_reward=self.plat_area_rewards(idx_1d,world)
        node_reward=self.plat_node_rewards(idx_2d,world)
        # node_reward=0
        group_reward=self.agents_group_rewards(agent,world)

        # print(plat_reward)
        
        # dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        # print(dist2)
        rewards=area_reward+node_reward+group_reward
        # print(area_reward,node_reward,group_reward,rewards)
        # print(group_reward)
        # print('###',rewards)
        return rewards
        

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # entity_pos = []
        # for entity in world.agents:  # world.entities:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            # comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)        
        
        # print(np.concatenate([agent.state.p_vel] + other_pos))
        # return np.array(agents_pos)
        # return np.concatenate([agent.state.p_vel])# + other_pos)   
        observation_array=np.concatenate(
            [agent.state.p_vel] 
            + [agent.state.p_pos]
            +other_pos
            ) 
        
        # print(f'###{len(world.agents)};{observation_array}') #
        return observation_array
    
if __name__=="__main__":    
    import os
    import rioxarray as rxr
    import matplotlib
    import usda.rl as usda_rl 
    import matplotlib.pyplot as plt
    import matplotlib
    
    piazza_navona_shp_root=r'C:\Users\richie\omen_richiebao\omen_github\USDA_special_study\data\piazza_navona_osm'
    area_node_fn=os.path.join(piazza_navona_shp_root,'navona_area_node.tif')
    navona_area_node=rxr.open_rasterio(area_node_fn)
    plat=np.stack(navona_area_node.data,axis=2)
    plat[:,:,0][plat[:,:,0]==-9999]=15
    plat=plat[100:500,100:500:,:]
    # print(plat)    
    
    def cmap2hex(cmap_name,N): 
        cmap = matplotlib.cm.get_cmap(cmap_name, N)
        hex_colors={i:matplotlib.colors.rgb2hex(cmap(i)) for i in range(N)}
        return hex_colors        
    
    plat_colors=cmap2hex('Pastel1',16)
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
        15:5}
    
    env=raw_env(
        render_mode="rgb_array",
        plat=plat,
        plat_colors=plat_colors,
        plat_rewards=plat_rewards,
        agents_num=20,
        nodes_radius=50,
        group_dis=0.01,
        # continuous_actions=True,
        )
    
    
    env.reset() # seed=578    
    
    
    frames=[]
    for agent in env.agent_iter():
        # print(env.render())
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
    
    anim=usda_rl.plot_animation(frames,interval=100,figsize=(10, 10))
    f=r'C:\Users\richie\omen_richiebao\omen_temp/temp_2.gif'
    anim.save(f,fps=10)
    # 
    fig=plt.figure(figsize=(10,10))
    plt.imshow(frames[0],cmap=cmap,norm=norm)    
    plt.show()
    
    
    #-------------------------------------------------------------------------
    