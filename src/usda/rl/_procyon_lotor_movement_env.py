# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:51:09 2023

@author: richie bao
"""
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import usda.utils as usda_utils

class ProcyonLotorMovementEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, 
                 array2d,
                 model,
                 agent_location,
                 target_location,
                 vals,
                 dist=22,
                 scale=20000,
                 reward_upper_truncation=10,
                 reward_lower_truncation=0,
                 X_dict={1:0,2:0,3:0,4:0,5:0,6:0,7:0},
                 X_keys=[1,2,3,4,5,6,7],
                 render_mode=None,
                 terminated_count_print=False): 
        self.array2d=array2d
        height,width=array2d.shape
        self.height=height # The size of the square grid height
        self.width=width  # The size of the square grid width
        self.window_width=width # The size of the PyGame window width
        self.window_height=height # The size of the PyGame window height
     
        self.observation_space=spaces.Discrete(width*height)         
        self.action_space = spaces.Discrete(4) # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.model=model

        self.agent_location=np.array(agent_location)
        self.target_location=np.array(target_location)
        self.dist=dist
        self.scale=scale
        self.terminated_count=0
        self.X_dict=X_dict
        self.X_keys=X_keys
        self.reward_upper_truncation=reward_upper_truncation
        self.reward_lower_truncation=reward_lower_truncation
        self.terminated_count_print=terminated_count_print
        self.vals=vals
        
        # self.obstacles=np.array([[i,j] for i in range(20,100) for j in range(170,170+80)])
        # self.obstacles=np.array([[i,j] for i in range(10,30) for j in range(85,85+20)])

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken. I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in human-mode. They will remain `None` until human-mode is used for the first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):        
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):        
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
        
    def reset(self, seed=None, options=None):        
        super().reset(seed=seed) # We need the following line to seed self.np_random
        self._agent_location=self.agent_location
        self._target_location =self.target_location
        self.t_step=0

        observation = np.ravel_multi_index(self._get_obs()['agent'],(self.width,self.height))    
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def procyon_lotor_rewards(self,row,col,r=22):
        nbr_xy=usda_utils.grid_neighbors(self.array2d,row,col,r=r)
        nbr_dist=usda_utils.grid_distance(nbr_xy,row,col)
        group_sum_dict=usda_utils.group_sum(self.vals,nbr_xy,1/nbr_dist)  
        self.X_dict.update(group_sum_dict)
        X=np.array([[self.X_dict[k] for k in self.X_keys]])
        y=self.model.predict(X)
        
        return y[0]/self.scale

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, [0,0], [self.width-self.dist - 1,self.height-self.dist-1])
        
        reward = self.procyon_lotor_rewards(*self._agent_location,self.dist)
        
        if reward<=self.reward_lower_truncation:
            reward=-1
        elif reward>self.reward_upper_truncation:
            reward=-1              

        # terminated = np.array_equal(self._agent_location, self._target_location) # An episode is done if the agent has reached the target   
        if self._agent_location.tolist() in self._target_location.tolist():
            terminated =True
            self.terminated_count+=1
            if self.terminated_count_print:
                print(self.terminated_count)
        else:
            terminated =False
            
        if terminated:
            reward=100
            
        observation = np.ravel_multi_index(self._get_obs()['agent'],(self.width,self.height))
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()        

        self.t_step+=1
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas= pygame.surfarray.make_surface(self.array2d)

        pix_square_size = (
            max(self.window_width,self.window_height)/ max(self.width,self.height)
        )  # The size of a single grid square in pixels

        # First we draw the target
        # pygame.draw.rect(
        #     canvas,
        #     8,#(255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location,
        #         (pix_square_size*5, pix_square_size*5),
        #     ),
        # )
        
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            8,#(0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size*2 ,
        )
        
        for target_loc in self._target_location:
            pygame.draw.rect(canvas, 9, pygame.Rect(pix_square_size *target_loc,(pix_square_size, pix_square_size)))        

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return pygame.surfarray.array2d(canvas)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()           