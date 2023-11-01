from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

from vmas.simulator.scenario import BaseScenario

import typing
from abc import ABC, abstractmethod
from typing import List

import torch
from torch import Tensor

from vmas.simulator.utils import Color, ScenarioUtils

from vmas.simulator.core import World, Agent, Sphere, Landmark
from vmas.simulator.utils import (
    INITIAL_VIEWER_SIZE,
    VIEWER_MIN_ZOOM,
    AGENT_OBS_TYPE,
    AGENT_REWARD_TYPE,
    AGENT_INFO_TYPE,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class Scenario(BaseScenario):
     def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        # pass any kwargs that we need
        # n_agents is num of pipes to route, default is 2
        self.n_agents = kwargs.get("n_agents", 1)
        # set collision bool
        self.collisions = kwargs.get("collisions", True)
        # set start position per agent
        start_pos = kwargs.get("start_pos",[(0,0)])
        self.start_pos = torch.tensor(start_pos)
        # set goals per agent
        goals = kwargs.get("goals",[(1,1)])
        self.goals = torch.tensor(goals)

        # create world
        world = World(batch_dim, device, dt=0.1, drag = 0)
        # add agents
        for i in range(self.n_agents):
            agent = Agent(name=f"agent {i}",
                             collide=True,
                             mass=1.0,
                             shape=Sphere(radius=0.04),
                             max_speed=None,
                             color=Color.BLUE,
                             u_range=1.0,
            )
            world.add_agent(agent)
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=Color.RED,
            )
            world.add_landmark(goal)
            agent.goal = goal
           

        return world
     
     

     def reset_world_at(self, env_index: int = None):
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(self.start_pos[i], batch_index=env_index)
            agent.goal.set_pos(self.goals[i], batch_index=env_index)

        

     def observation(self, agent: Agent) -> AGENT_OBS_TYPE:
        # get position of all entities in this agent's reference frame
        goal_dist = []
        goal_dist.append(agent.state.pos - agent.goal.state.pos)
        return torch.cat([
                agent.state.pos,
                agent.state.vel,
            ]
            + goal_dist)
     
     def reward(self, agent: Agent) -> AGENT_REWARD_TYPE:
        # reward every agent proportionally to distance from their goal
        dist_rew = torch.sum(
            torch.square(agent.state.pos - agent.goal.state.pos, dim=-1)
        )
        return -dist_rew
     
     def done(self):
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)
     
     def info(self, agent: Agent) -> AGENT_INFO_TYPE:
        return {}
     

start_pos = [(0,0),(0,1)]
goals = [(4,4),(1,4)]
scenario = Scenario()

env = VmasEnv(
    scenario=scenario, 
    num_envs=32, 
    n_agents=2, 
    start_pos = start_pos,
    goals = goals)
# print(env.start_pos)
print("action_spec:", env.full_action_spec)
print("reward_spec:", env.full_reward_spec)
print("done_spec:", env.full_done_spec)
print("observation_spec:", env.observation_spec)

print("action_keys:", env.action_keys)
print("reward_keys:", env.reward_keys)
print("done_keys:", env.done_keys)
     