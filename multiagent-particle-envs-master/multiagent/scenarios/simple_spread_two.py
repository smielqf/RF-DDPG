import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario



class Scenario(BaseScenario):
    def make_world(self,kzseed=1):
        np.random.seed(kzseed)
        world = World()
        # set any world properties first, note that here the number of agents and the number of landmarks should be the same
        world.dim_c = 0
        num_agents = 2
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.ID = i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05+0.1*i
            # agent.costcoeff = np.random.uniform(0.1, 2.0, 1)
            agent.costcoeff = 0.1*20**i
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world,kzseed)
        return world

    def reset_world(self,world,kzseed):
        np.random.seed(kzseed)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_pos = [-0.5997712503653102*(-1)**i+0.3*i*(-1)**i,-0.09533485473632046*(-1)**i+0.2*i*(-1)**i]
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Each agent is rewarded based on the distance with respect to the one it is assigned to
        # rew = 0
        l = world.landmarks[agent.ID]
        # agweight=agent.costcoeff
        rew = -1*np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))


        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 0.5
        return rew*agent.costcoeff*1.0

    def observation(self, agent, world):
        # get positions of all entities in the global frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos)

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        all_pos = []
        for other in world.agents:
            # if other is agent: continue
            comm.append(other.state.c)
            all_pos.append(other.state.p_pos)
        return np.concatenate(entity_pos + all_pos)
