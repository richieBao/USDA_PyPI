import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding
# import matplotlib.colors
from PIL import ImageColor

from pettingzoo import AECEnv
from pettingzoo.mpe._mpe_utils.core import Agent
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

# from usda.mpe_realworld import AECEnv
# from usda.mpe_realworld.mpe._mpe_utils.core import Agent
# from usda.mpe_realworld.utils import wrappers
# from usda.mpe_realworld.utils.agent_selector import agent_selector

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
step_set=set()

def make_env(raw_env):
    # print('-#!')
    def env(**kwargs):
        env = raw_env(**kwargs)        
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        continuous_actions=False,
        local_ratio=None,
        plat=None,
        plat_colors=None,
        plat_rewards=None,
        nodes_radius=30,
        group_dis=0.5,
    ):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()          
        self.viewer = None
        
        
        # print('?????????',plat)
        if plat is not None:
            world.plat=plat
            world.plat_width,world.plat_height=plat.shape[:2]
            world.nodes_radius=nodes_radius
            world.group_dis=group_dis
        if plat_rewards is not None:
            world.plat_rewards=plat_rewards
        if plat_colors is not None:
            world.plat_colors=plat_colors
            
        # print('###!@@@',world.plat)
        plat=world.plat
        plat_colors=world.plat_colors
        if plat is not None:
            self.plat=plat        
            plat_colors_rgb={k:ImageColor.getcolor(v,'RGB') for k,v in plat_colors.items()}
            # print(array2d_colors_rgb)
            # get keys and values, make sure they are ordered the same
            keys, values = zip(*plat_colors_rgb.items())
            # making use of the fact that the keys are non negative ints
            # create a numpy friendly lookup table
            out = np.empty((max(keys) + 1, 3), int)
            out[list(keys), :] = values      
            # now out can be used to look up the tuples using only numpy indexing  
            if len(plat.shape)>2:
                self.plat_rgb=out[plat[:,:,0],:] 
            else:
                self.plat_rgb=out[plat,:]   
            # print(self.array2d_rgb)            
       
            self.height=world.plat_height # plat.shape[1]
            self.width=world.plat_width # bplat.shape[0]    
            
            self.screen= pygame.surfarray.make_surface(self.plat_rgb) 
        else:
            self.width = 700 
            self.height = 700             
            self.screen = pygame.Surface([self.width, self.height])
        
        # if plat is not None:
        #     self.screen= pygame.surfarray.make_surface(self.plat_rgb)        
        # else:
        #     self.screen = pygame.Surface([self.width, self.height])
    
        
        self.max_size = 1
        # self.game_font = pygame.freetype.Font(
        #     os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        # )
        self.game_font = pygame.font.Font(None, 24)

        # Set up the drawing window

        self.renderOn = False
        self._seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio
        
        # print(world.plat,'\n',world.plat_rewards,'\n',world.plat_colors)
        self.scenario.reset_world(self.world, self.np_random)
        # print(world.plat,'\n',world.plat_rewards,'\n',world.plat_colors)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, shape=(space_dim,)
                )
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0

        self.current_actions = [None] * self.num_agents
        
        

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        ).astype(np.float32)

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            # print(action)
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
                
            # print('-',agent.action.u)
            agent.action.u *= sensitivity
            action = action[1:]
            # print(action)
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        # print(action)
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action
        # print(self.current_actions)

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True        
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()
        
        
        # if  self.steps not in step_set:
        #     print('-',self.steps)
        #     step_set.add(self.steps)

        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            # return pygame.surfarray.array2d(self.screen)
            return observation
            # return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            return

    def draw(self):
        # clear screen
        # self.screen.fill((255, 255, 255))        
        if self.plat is not None:
            self.screen= pygame.surfarray.make_surface(self.plat_rgb)        
        else:
            self.screen.fill(self.array2d_rgb)    
 

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))
        # print('#2',cam_range)

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            x=-1 if x<-1 else 1 if x>1 else x
            y=-1 if y<-1 else 1 if y>1 else y
            
            # print('#1',x,y)
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            # x = (
            #     (x / cam_range) * self.width // 2 * 0.9
            # )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            # y = (y / cam_range) * self.height // 2 * 0.9
            
            x_ =  x  * self.width // 2
            y_ = y * self.height // 2      
            # print('###',x_,y_,self.width // 2,self.height // 2   )
            
            x_ += self.width // 2
            y_ += self.height // 2
            x_=0 if x_<0 else x_
            # print()
            # print('#1',x_,y_,entity.state.p_pos)
            
            # print(entity.color,(x_, y_))
            pygame.draw.circle(
                self.screen, entity.color * 200, (x_, y_), entity.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x_, y_), entity.size * 350, 1
            )  # borders
            assert (
                0 <= x_ <= self.width and 0 <= y_ <= self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
