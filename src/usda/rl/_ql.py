# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:10:35 2023

@author: richie bao
migrated: Frozenlake benchmark https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
"""
from typing import NamedTuple
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numbers

class Params_frozenlake(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved
    
class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon,rng):
        self.epsilon = epsilon
        self.rng=rng

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action
    
def run_env(params,learner,explorer,env):
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        learner.reset_qtable()  # Reset the Q-table between runs

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
                
            state = env.reset(seed=params.seed)[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions    

def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st

def qtable_directions_map(qtable, map_size,transform=False,directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}):
    """Get the best learned action & map it to arrows."""
    if isinstance(map_size,numbers.Number):
        qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
        qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    else:
        qtable_val_max = qtable.max(axis=1).reshape(map_size[0], map_size[1])
        qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size[0], map_size[1])

    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if abs(qtable_val_max.flatten()[idx]) > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    if isinstance(map_size,numbers.Number):
        qtable_directions = qtable_directions.reshape(map_size, map_size)
    else:
        qtable_directions = qtable_directions.reshape(map_size[0], map_size[1])
        
    if transform:
        return qtable_val_max.T, qtable_directions.T
    else:
        return qtable_val_max, qtable_directions
    
def plot_q_values_map(qtable, env, map_size,figsize=(15, 5),savepath=None,transform=False,directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size,transform,directions )

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    
    if isinstance(map_size,numbers.Number):
        img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    else:
        img_title = f"frozenlake_q_values_{map_size[0]}x{map_size[1]}.png"
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()
    
def plot_states_actions_distribution(states, actions, map_size,figsize=(15, 5),savepath=None):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()    
    
def plot_steps_and_rewards(rewards_df, steps_df,figsize=(15, 5),savepath=None):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()    
    
    
    
if __name__=="__main__":       
    params = Params_frozenlake(
        total_episodes=2000, 
        learning_rate=0.8,
        gamma=0.95,
        epsilon=0.1,
        map_size=11,
        seed=123,
        is_slippery=False,
        n_runs=20, # 20,
        action_size=None,
        state_size=None,
        proba_frozen=0.9,
        savefig_folder=None # Path("../../_static/img/tutorials/"),
    )  

    # Set the seed
    rng = np.random.default_rng(params.seed)
    
    # Create the figure folder if it doesn't exists
    # params.savefig_folder.mkdir(parents=True, exist_ok=True)    
    
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="rgb_array",
        desc=generate_random_map(
            size=params.map_size, p=params.proba_frozen, seed=params.seed
        ),
    )    
           
    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)       
    
    
    learner = Qlearning(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )    
    
    explorer = EpsilonGreedy(
        epsilon=params.epsilon,
        rng=rng
    )    
        
    rewards, steps, episodes, qtables, all_states, all_actions = run_env(params,learner,explorer,env)    
    
    map_size=params.map_size
    res, st = postprocess(episodes, params, rewards, steps, map_size)
    plot_states_actions_distribution(
        states=all_states, actions=all_actions, map_size=map_size
    ) 
    
    qtable = qtables.mean(axis=0) 
    plot_q_values_map(qtable, env, map_size)
    
    plot_steps_and_rewards(res, st)
