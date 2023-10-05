# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from ._utils import save_frames_as_gif4gym
from ._utils import env_info_print
from ._utils import gym_env_steps_rgb_array
from ._utils import show_one_episode_QL
from ._utils import custom_env_steps_rgb_array
from ._utils import plot_animation

from ._ql import Params_frozenlake
from ._ql import Qlearning
from ._ql import EpsilonGreedy
from ._ql import run_env
from ._ql import postprocess
from ._ql import qtable_directions_map
from ._ql import plot_q_values_map
from ._ql import plot_states_actions_distribution
from ._ql import plot_steps_and_rewards

from ._dqn import ReplayMemory
from ._dqn import DQN
from ._dqn import select_action
from ._dqn import plot_durations
from ._dqn import optimize_model
from ._dqn import show_one_episode_DQN

from ._animal_movements import predicting_kde
from ._procyon_lotor_movement_env import ProcyonLotorMovementEnv

from ._ising_model import Ising_Metropolis_MCMC

from ._pi_monte_carlo import pi_using_MonteCarlo_anim
from ._pi_monte_carlo import pi_using_MonteCarlo

from ._ising_marl import Ising_MARL
from ._ising_marl import ising_1d_plot
from ._ising_marl import ising_2d_plot

 
__all__ = [
    "save_frames_as_gif4gym",
    "env_info_print",
    "gym_env_steps_rgb_array",
    "Params_frozenlake",
    "Qlearning",
    "EpsilonGreedy",
    "run_env",
    "postprocess",
    "qtable_directions_map",
    "plot_q_values_map",
    "plot_states_actions_distribution",
    "plot_steps_and_rewards",
    "show_one_episode_QL",
    "custom_env_steps_rgb_array",
    "ReplayMemory",
    "DQN",
    "select_action",
    "plot_durations",
    "optimize_model",
    "show_one_episode_DQN",
    "predicting_kde",
    "Ising_Metropolis_MCMC",
    "pi_using_MonteCarlo_anim",
    "pi_using_MonteCarlo",
    "Ising_MARL",
    "ising_1d_plot",
    "ising_2d_plot",
    "plot_animation",
    ]

