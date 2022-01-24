import numpy as np
from numpy.core.fromnumeric import reshape, size
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import matplotlib.pyplot as plt
import os
import imp
import argparse

from tqdm import tqdm
from collections import namedtuple, deque
from plot import plot_all

DDPG_Transition = namedtuple('DDPG_Transition', ['state', 'action', 'reward', 'next_state', 'next_action'])
class DDPGAgentTrainer(object):
    def __init__(self, dim_input, dim_output, dim_q_input, args,  agent_index=0, device='cpu', buffer_size=1e6):
        self.dim_input = dim_input        
        self.dim_output = dim_output
        self.dim_q_input = dim_q_input
        self.num_units = 64
        self.lr = 1e-3
        self.gamma = 0.95
        self.update_count = 0
        self.update_count_p = 0
        self.update_count_q = 0
        self.agent_index = agent_index

        self.buffer_size = buffer_size
        self.batch_size = args.batch_size
        self.max_episode_len = args.max_episode_len
        self.device = device
        

        # Define main networks
        # Local observation of the agent
        p_input_size = dim_input
        p_output_size = dim_output
        q_input_size = dim_q_input
        q_output_size = 1   # For Q-Value only

        class MLP(nn.Module):
            def __init__(self, input=69, output=5, num_units=64):
                super(MLP, self).__init__()
                self.network_stack = nn.Sequential(
                            nn.Linear(input, num_units),
                            nn.ReLU(),
                            nn.Linear(num_units, num_units),
                            nn.ReLU(),
                            nn.Linear(num_units, output)
                        )

            def forward(self, x):
                x = self.network_stack(x)

                return x

        self.p_network = MLP(input=p_input_size, output=p_output_size, num_units=self.num_units).to(self.device)
        self.q_network = MLP(input=q_input_size, output=q_output_size, num_units=self.num_units).to(self.device)
        self.target_p_network = MLP(input=p_input_size, output=p_output_size, num_units=self.num_units).to(self.device)
        self.target_q_network = MLP(input=q_input_size, output=q_output_size, num_units=self.num_units).to(self.device)
        self.target_p_network.load_state_dict(self.p_network.state_dict())
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.p_optimizer = optim.Adam(self.p_network.parameters(), lr=self.lr)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Create experience buffer
        class ReplayBuffer(object):
            def __init__(self, max_len):
                self.buffer = deque([], maxlen=int(max_len))
            
            def push(self, *args):
                self.buffer.append(DDPG_Transition(*args))

            def sample(self, batch_size):
                return random.sample(self.buffer, batch_size)

            def __len__(self):
                return len(self.buffer)

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        # self.max_replay_buffer_len = self.batch_size * self.max_episode_len
        self.max_replay_buffer_len = 1000*25

    def act(self, obs):
        """ Compute the action to be taken.
        
        Parameters
        ----------
        obs : np.array
            The local observation of the current agent, with size [observation, ]
        
        Returns
        -------
        np.array
            The action to be taken.
        """
        self.eval()
        _obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device).view(1,-1)
        action = self._get_action(self.p_network, _obs)
        if self.device == 'cpu':
            action = action.clone().detach().numpy()[0]
        else:
            action = action.cpu().clone().detach().numpy()[0]
        return action

    def tgt_act(self, obs):
        """ Compute the action to be taken.
        
        Parameters
        ----------
        obs : np.array
            The local observation of the current agent, with size [observation, ]
        
        Returns
        -------
        np.array
            The action to be taken.
        """
        self.eval()
        _obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device).view(1,-1)
        action = self._get_action(self.target_p_network, _obs)
        if self.device == 'cpu':
            action = action.clone().detach().numpy()[0]
        else:
            action = action.cpu().clone().detach().numpy()[0]
        return action

    def _get_action(self, p_network, obs):
        """ Return the action computed by p_network.

        Parameters
        ----------
        p_network : torch.nn.Module
            The model which is used to compute action taken by the agent.
        obs_n : list
            Each element is a tensor with size [batch_size, num_local_observation].

        Returns
        -------
        Tensor
            The tensor with  size [num_action, ].
        """
        p = p_network(obs)
        u = torch.empty(p.size())
        u = torch.nn.init.uniform_(u).to(self.device)
        act = F.softmax(p - torch.log(-torch.log(u)), dim=1)
        

        return act

    def single_update(self, obs, action, reward, new_obs):
        _obs = torch.tensor(obs, device=self.device, dtype=torch.float, requires_grad=False).view(1,-1)
        _action = torch.tensor(action, device=self.device, dtype=torch.float, requires_grad=False).view(1, -1)
        _reward = torch.tensor(reward, device=self.device, dtype=torch.float, requires_grad=False).view(1, -1)
        _new_obs = torch.tensor(new_obs, device=self.device, dtype=torch.float, requires_grad=False).view(1,-1)
        _action_next = self.target_p_network(_new_obs).detach()

        cur_q_input = torch.cat([_obs, _action], dim=1)
        next_q_input = torch.cat([_new_obs, _action_next], dim=1)
        q_value = self.q_network(cur_q_input)
        next_q = self.target_q_network(next_q_input).detach()

        q_loss = torch.mean(torch.pow(_reward + self.gamma * next_q - q_value , 2))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.p_network.parameters(), .5)
        self.q_optimizer.step()

        ### Actor
        action = self._get_action(self.p_network, _obs)
        q_input = torch.cat([_obs, _action], dim=1)
        p_loss = -torch.mean(self.q_network(q_input))
        self.p_optimizer.zero_grad()
        p_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), .5)
        self.p_optimizer.step()

        self.make_update_exp(self.p_network, self.target_p_network)
        self.make_update_exp(self.q_network, self.target_q_network)

        return p_loss.clone().detach().cpu().numpy(), q_loss.clone().detach().cpu().numpy(), torch.mean(q_value).clone().detach().cpu().numpy()

    def batch_update_fully_p_train(self, update_gap, batch_index):
        """ Update trainable parameters of the trainer.
        
        Parameters
        ----------
        agents : list
            It contains all the agents and each element is an instance of MADDPGAgentTrainer.
        t : int
            The amount of steps taken for training.
        """
        self.train()

        # Replay buffer is not large enough
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            # return {'p_loss': np.inf, 'q_loss': np.inf, 'q-value': np.nan, 'target_q_value': np.nan}
            return .0
        if not self.update_count_p % update_gap == 0:  # Only update every 100 steps
            # return {'p_loss': np.inf, 'q_loss': np.inf, 'q-value': np.nan, 'target_q_value': np.nan}
            self.update_count_p += 1
            return .0
        self.update_count_p += 1

        # Sample data from replay buffer
        # transitions= self.replay_buffer.sample(self.batch_size)
        transitions = []
        for index in batch_index:
          transitions.append(self.replay_buffer.buffer[index])
        batch_data = DDPG_Transition(*zip(*transitions))
        _obs = torch.tensor(batch_data.state, device=self.device, dtype=torch.float, requires_grad=False)
        _act = torch.tensor(batch_data.action, device=self.device, dtype=torch.float, requires_grad=False)
        obs = _obs.clone().view(self.batch_size, -1)

        

        ### Actor
        action = self._get_action(self.p_network, _obs.view(self.batch_size, -1))
        _act[:, self.agent_index, :] = action
        # _act = action
        q_input = torch.cat([obs, _act.view(self.batch_size, -1)], dim=1)
        p_loss = -torch.mean(self.q_network(q_input))
        self.p_optimizer.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p_network.parameters(), .5)
        self.p_optimizer.step()

        self.make_update_exp(self.p_network, self.target_p_network)

        return p_loss.clone().detach().cpu().numpy()

    def batch_update_fully_q_train(self, update_gap, batch_index):
        """ Update trainable parameters of the trainer.
        
        Parameters
        ----------
        agents : list
            It contains all the agents and each element is an instance of MADDPGAgentTrainer.
        t : int
            The amount of steps taken for training.
        """
        self.train()

        # Replay buffer is not large enough
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            # return {'p_loss': np.inf, 'q_loss': np.inf, 'q-value': np.nan, 'target_q_value': np.nan}
            return .0, .0
        if not self.update_count_q % update_gap == 0:  # Only update every 100 steps
            # return {'p_loss': np.inf, 'q_loss': np.inf, 'q-value': np.nan, 'target_q_value': np.nan}
            self.update_count_q += 1
            return .0, .0
        self.update_count_q += 1

        # Sample data from replay buffer
        # transitions= self.replay_buffer.sample(self.batch_size)
        transitions = []
        for index in batch_index:
          transitions.append(self.replay_buffer.buffer[index])
        batch_data = DDPG_Transition(*zip(*transitions))
        _obs = torch.tensor(batch_data.state, device=self.device, dtype=torch.float, requires_grad=False)
        _act = torch.tensor(batch_data.action, device=self.device, dtype=torch.float, requires_grad=False)
        reward = np.array(batch_data.reward)
        _obs_next = torch.tensor(batch_data.next_state, device=self.device, dtype=torch.float, requires_grad=False)
        # act_next = self._get_action(self.target_p_network, obs).detach().view(self.batch_size, -1)
        _act_next = torch.tensor(batch_data.next_action, device=self.device, dtype=torch.float, requires_grad=False)
        obs = _obs.clone().view(self.batch_size, -1)
        act = _act.clone().view(self.batch_size, -1)
        obs_next = _obs_next.clone().view(self.batch_size, -1)

        # Critic
        cur_q_input = torch.cat([obs, act], dim=1)
        next_q_input = torch.cat([obs_next, _act_next.view(self.batch_size, -1)], dim=1)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float, requires_grad=False).view(self.batch_size, -1)
        q_value = self.q_network(cur_q_input)
        next_q = self.target_q_network(next_q_input).detach()

        q_loss = torch.mean(torch.pow(reward + self.gamma * next_q - q_value , 2))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), .5)

        return q_loss.clone().detach().cpu().numpy(), torch.mean(q_value).clone().detach().cpu().numpy()


    def get_state_dict(self):
        """Return all the trainable parameters.
        
        Returns
        -------
        dict
            The dictionary which contains all traiable parameters.
        """
        state_dict = {
            'p_network': self.p_network.state_dict(), 'target_p_network': self.target_p_network.state_dict(),
            'q_network': self.q_network.state_dict(), 'target_q_network': self.target_q_network.state_dict(),
            'p_optimizer': self.p_optimizer.state_dict(), 'q_optimizer': self.q_optimizer.state_dict()
            }
        return state_dict
    
    def restore_state(self, ckpt):
        """Restore all the trainable parameters.
        
        Parameters
        ----------
        ckpt : dict
            Contain all information for restoring trainable parameters.
        """
        self.p_network.load_state_dict(ckpt['p_network'])
        self.q_network.load_state_dict(ckpt['q_network'])
        self.target_p_network.load_state_dict(ckpt['target_p_network'])
        self.target_q_network.load_state_dict(ckpt['target_q_network'])
        self.p_optimizer.load_state_dict(ckpt['p_optimizer'])
        self.q_optimizer.load_state_dict(ckpt['q_optimizer'])

    def eval(self):
        """ Switch all models inside the trainer into eval mode.
        """
        self.p_network.eval()
        self.q_network.eval()
        self.target_p_network.eval()
        self.target_q_network.eval()
    
    def train(self):
        """ Switch all models inside the trainer into train mode.
        """
        self.p_network.train()
        self.q_network.train()
        self.target_p_network.train()
        self.target_q_network.train()

    def make_update_exp(self, source, target, rate=1e-2):
        """ Use values of parameters from the source model to update values of parameters from the target model. Each update just change values of paramters from the target model slightly, which aims to provide relative stable evaluation. Note that structures of the two models should be the same. 
        
        Parameters
        ----------
        source : torch.nn.Module
            The model which provides updated values of parameters.
        target : torch.nn.Module
            The model which receives updated values of paramters to update itself.
        """
        polyak = rate
        for tgt, src in zip(target.named_parameters(recurse=True), source.named_parameters(recurse=True)):
            assert src[0] == tgt[0] # The identifiers should be the same
            tgt[1].data = polyak * src[1].data + (1.0 - polyak) * tgt[1].data


def make_env_no_diff_dist(scenario_name, arglist, benchmark=False):
    from multiagent.environment_no_diff_dist import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    pathname = os.path.join(os.path.join(os.path.dirname(__file__), 'scenarios'), scenario_name + '.py')
    scenario = imp.load_source('', pathname).Scenario()
    
    # create world
    world = scenario.make_world()

    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation)

    return env

    
def aggregate_grads(trainers, device):
    q_network_state_dict = trainers[0].q_network.state_dict()

    temp_grads = {}
    for para_name in q_network_state_dict:
        para = q_network_state_dict[para_name]
        temp_grads[para_name] = torch.zeros(size=para.shape, device=device, dtype=torch.float)
    
    for i, agent in enumerate(trainers):
        gradient_dict = {k:v.grad for k, v in zip(agent.q_network.state_dict(), agent.q_network.parameters())}
        for para_name in gradient_dict:
            temp_grads[para_name] += gradient_dict[para_name]    
    for para_name in temp_grads:
        temp_grads[para_name] /= len(trainers)
    
    for i, agent in enumerate(trainers):
        gradient_dict = {k:v.grad for k, v in zip(agent.q_network.state_dict(), agent.q_network.parameters())}
        for para_name in gradient_dict:
            gradient_dict[para_name].copy_(temp_grads[para_name])

def geometric_median(wList):
    max_iter = 80
    tol = 1e-5
    guess = torch.mean(wList, dim=0)
    for _ in range(max_iter):
        dist_li = torch.norm(wList-guess, dim=1)
        for i in range(len(dist_li)):
            if dist_li[i] == 0:
                dist_li[i] = 1
        temp1 = torch.sum(torch.stack([w/d for w, d in zip(wList, dist_li)]), dim=0)
        temp2 = torch.sum(1/dist_li)
        guess_next = temp1 / temp2
        guess_movement = torch.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= tol:
            break
    return guess

def geometric_median_aggregation(grad_dict, num_agent):
    temp_flat_grads = [[] for _ in range(num_agent)]
    temp_flat_size = [[] for _ in range(num_agent)]
    for gradient_name in grad_dict:
        for i, gradient in enumerate(grad_dict[gradient_name]):
            para_size = gradient.nelement()
            temp_flat_grads[i].append(gradient.view(-1))
            temp_flat_size[i].append(para_size)
    
    torch_flat_grads = [torch.cat(x) for x in temp_flat_grads]
    torch_flat_grads = torch.stack(torch_flat_grads, dim=0)
    g_median = geometric_median(torch_flat_grads)
    new_flat_grads = {}
    start = 0
    pos = 0 
    for para_name in grad_dict:
        flat_grad = g_median[start:start+temp_flat_size[i][pos]]
        para_grad = torch.reshape(flat_grad, shape=grad_dict[para_name][0].shape)
        new_flat_grads[para_name] = para_grad
        start += temp_flat_size[i][pos]
        pos += 1

    
    # for i, trainer in enumerate(trainers):
    #     for para_name in q_network_state_dict:
    #         q_network_state_dict[para_name].grad._copy(new_flat_grads[para_name])

    return new_flat_grads
        


def byzantine_grads(trainers, malic=None, attack='sign-flip', robust_aggregate='mean', device='cpu'):
    q_network_state_dict = trainers[0].q_network.state_dict()

    temp_grads = {}
    if robust_aggregate == 'mean':
        for para_name in q_network_state_dict:
            para = q_network_state_dict[para_name]
            temp_grads[para_name] = torch.zeros(size=para.shape, device=device, dtype=torch.float)
    else:
        for para_name in q_network_state_dict:
            temp_grads[para_name] = []
    
    for i, agent in enumerate(trainers):
        gradient_dict = {k:v.grad for k, v in zip(agent.q_network.state_dict(), agent.q_network.parameters())}
        for para_name in gradient_dict:
            if robust_aggregate == 'mean':
                if malic is not None and i in malic:
                    if attack == 'sign-flip':
                        temp_grads[para_name] -= 10. * gradient_dict[para_name]
                    elif attack == 'zero':
                        temp_grads[para_name] += 20. * torch.ones_like(gradient_dict[para_name]).to(device)
                    elif attack == 'gaussian':
                        temp_grads[para_name] += torch.normal(0, 100, size=gradient_dict[para_name].shape).to(device)
                else:
                    temp_grads[para_name] += gradient_dict[para_name]
            else:
                if malic is not None and i in malic:
                    if attack == 'sign-flip':
                        temp_grads[para_name].append(-10. * gradient_dict[para_name])
                    elif attack == 'zero':
                        temp_grads[para_name].append(20. * torch.ones_like(gradient_dict[para_name]).to(device))
                    elif attack == 'gaussian':
                        temp_grads[para_name].append(torch.normal(0, 100, size=gradient_dict[para_name].shape).to(device))
                else:
                    temp_grads[para_name].append(gradient_dict[para_name])   
    
    if  robust_aggregate == 'mean':
        for para_name in temp_grads:
            temp_grads[para_name] /= len(trainers)
    elif robust_aggregate  == 'median':
        for para_name in temp_grads:
            all_paras = torch.stack(temp_grads[para_name], dim=0)
            temp_grads[para_name] = torch.median(all_paras, dim=0)[0]
    elif robust_aggregate == 'geometry_median':
        temp_grads = geometric_median_aggregation(temp_grads, num_agent=len(trainers))
    
    for i, agent in enumerate(trainers):
        # gradient_dict = {k:v.grad for k, v in zip(agent.q_network.state_dict(), agent.q_network.parameters())}
        # for para_name in gradient_dict:
        #     gradient_dict[para_name]= temp_grads[para_name].clone()
        gradient_dict = {k:v for k, v in zip(agent.q_network.state_dict(), agent.q_network.parameters())}
        for para_name in gradient_dict:
            gradient_dict[para_name].grad.copy_(temp_grads[para_name])

def cal_layer_norm(trainers):
    para_norm_list = [[] for _ in range(len(trainers))]
    grad_norm_list = [[] for _ in range(len(trainers))]
    layer_list = [0, 2, 4]
    for i, agent in enumerate(trainers):       
        network_statck  = agent.q_network.network_stack
        for j, layer in enumerate(network_statck):
            if  j in layer_list:
                weight = layer.weight
                bias = layer.bias
                para = torch.cat([weight, bias.view(-1, 1)], dim=1)
                para_norm = torch.norm(para)
                para_norm_list[i].append(para_norm.item())
                if weight.grad is not None and bias.grad is not None:
                    grad = torch.cat([weight.grad, bias.grad.view(-1, 1)], dim=1)
                    grad_norm = torch.norm(grad)
                    grad_norm_list[i].append(grad_norm.item())
                else:
                    grad_norm_list[i].append(.0)
    return para_norm_list, grad_norm_list

def cal_half_layer_norm(trainers):
    first_para_norm_list = [[] for _ in range(len(trainers))]
    first_grad_norm_list = [[] for _ in range(len(trainers))]
    second_para_norm_list = [[] for _ in range(len(trainers))]
    second_grad_norm_list = [[] for _ in range(len(trainers))]
    layer_list = [0, 2, 4]
    for i, agent in enumerate(trainers):       
        network_statck  = agent.q_network.network_stack
        for j, layer in enumerate(network_statck):
            if  j in layer_list:
                weight = layer.weight
                bias = layer.bias
                para = torch.cat([weight, bias.view(-1, 1)], dim=1)
                first_para_norm = torch.norm(para[:,:len(para[0])//2])
                second_para_norm = torch.norm(para[:,len(para[0])//2:])
                first_para_norm_list[i].append(first_para_norm.item())
                second_para_norm_list[i].append(second_para_norm.item())
                if weight.grad is not None and bias.grad is not None:
                    grad = torch.cat([weight.grad, bias.grad.view(-1, 1)], dim=1)
                    first_grad_norm = torch.norm(grad[:, :len(grad[0])//2])
                    second_grad_norm = torch.norm(grad[:, len(grad[0])//2:])
                    first_grad_norm_list[i].append(first_grad_norm.item())
                    second_grad_norm_list[i].append(second_grad_norm.item())
                else:
                    first_grad_norm_list[i].append(.0)
                    second_grad_norm_list.append(.0)
    return first_para_norm_list, first_grad_norm_list, second_para_norm_list, second_grad_norm_list

def cal_vec_norm(trainers):
    par_vec = [[] for _ in range(len(trainers))]
    grad_vec = [[] for _ in range(len(trainers))]
    layer_list = [0, 2, 4]
    for i, agent in enumerate(trainers):       
        network_statck  = agent.q_network.network_stack
        for j, layer in enumerate(network_statck):
            if  j in layer_list:
                weight = layer.weight
                bias = layer.bias
                para = torch.cat([weight.view(-1), bias.view(-1)])
                par_vec[i].append(para)
                if weight.grad is not None and bias.grad is not None:
                    grad = torch.cat([weight.grad.view(-1), bias.grad.view(-1)])
                    grad_vec[i].append(grad)
    par_vec = [torch.cat(x) for x in par_vec]
    grad_vec = [torch.cat(x) if len(x) != 0 else None for x in grad_vec]
    
    para_norm_list = [torch.norm(x).item() for x in par_vec]
    grad_norm_list = [torch.norm(x).item() if x is not None else None for x in grad_vec]

    return para_norm_list, grad_norm_list

def cal_half_vec_norm(trainers):
    par_vec = [[] for _ in range(len(trainers))]
    grad_vec = [[] for _ in range(len(trainers))]

    layer_list = [0, 2, 4]
    for i, agent in enumerate(trainers):       
        network_statck  = agent.q_network.network_stack
        for j, layer in enumerate(network_statck):
            if  j in layer_list:
                weight = layer.weight
                bias = layer.bias
                para = torch.cat([weight.view(-1), bias.view(-1)])
                par_vec[i].append(para)
                if weight.grad is not None and bias.grad is not None:
                    grad = torch.cat([weight.grad.view(-1), bias.grad.view(-1)])
                    grad_vec[i].append(grad)
    par_vec = [torch.cat(x) for x in par_vec]
    grad_vec = [torch.cat(x) if len(x) != 0 else None for x in grad_vec]
    
    first_para_norm_list = [torch.norm(x[:len(x)//2]).item() for x in par_vec]
    first_grad_norm_list = [torch.norm(x[:len(x)//2]).item() if x is not None else None for x in grad_vec]
    second_para_norm_list = [torch.norm(x[len(x)//2:]).item() for x in par_vec]
    second_grad_norm_list = [torch.norm(x[len(x)//2:]).item() if x is not None else None for x in grad_vec]

    return first_para_norm_list, first_grad_norm_list, second_para_norm_list, second_grad_norm_list

def exp_fully(args, seed=0, malic_list=None, root_path=None):
    env = make_env_no_diff_dist(args.scenario, None)
    # env = make_env('spread_oneagent', None)
    num_agent = env.n
    num_episodes = args.num_episodes
    num_step_per_episode = args.max_episode_len
    

    batch_update = True
    if batch_update:
        update_gap = args.update_gap
    

    obs_n = env.reset()
    dim_input = len(obs_n) 
    dim_output = 5
    dim_critic_input = dim_input + dim_output*num_agent
    trainers = [DDPGAgentTrainer(dim_input, dim_output, dim_critic_input, args,  agent_index=i, device=args.device, buffer_size=arglist.replay_buffer_size) for i in range(num_agent) ]
    
    
    
    for i, agent in enumerate(trainers):
        if i == 0:
            pass
        else:
            agent.p_network.load_state_dict(trainers[0].p_network.state_dict())
            agent.q_network.load_state_dict(trainers[0].q_network.state_dict())
            agent.target_p_network.load_state_dict(trainers[0].target_p_network.state_dict())
            agent.target_q_network.load_state_dict(trainers[0].target_q_network.state_dict())

    total_rewards = [[] for _ in range(num_agent)]
    actor_losses = [[] for _ in range(num_agent)]
    critic_losses = [[] for _ in range(num_agent)]
    q_values = [[] for _ in range(num_agent)]
    para_norms = []
    grad_norms = []
    para_vec_norms = []
    grad_vec_norms = []
    first_para_norms = []
    second_para_norms = []
    first_grad_norms = []
    second_grad_norms = []
    first_para_vec_norms = []
    second_para_vec_norms = []
    first_grad_vec_norms = []
    second_grad_vec_norms = []

    count = 0
    para_norm, grad_norm = cal_layer_norm(trainers)
    para_vec_norm, grad_vec_norm = cal_vec_norm(trainers)
    para_norms.append(para_norm)
    para_vec_norms.append(para_vec_norm)

    first_para_norm, first_grad_norm, second_para_norm, second_grad_norm = cal_half_layer_norm(trainers)
    first_para_vec_norm, first_grad_vec_norm, second_para_vec_norm, second_grad_vec_norm = cal_half_vec_norm(trainers)
    first_para_norms.append(first_para_norm)
    second_para_norms.append(second_para_norm)
    first_para_vec_norms.append(first_para_vec_norm)
    second_para_vec_norms.append(second_para_vec_norm)

    for episode in tqdm(range(num_episodes)):
        actor_loss_episode = [[] for _ in range(num_agent)]
        critic_loss_episode = [[] for _ in range(num_agent)]
        rewards_episode = [[] for _ in range(num_agent)]
        obs_n = env.reset()
        q_value_episode = [[] for _ in range(num_agent)]
        action_n = [agent.act(obs_n) for i, agent in enumerate(trainers)]
        for step in range(num_step_per_episode):
            new_obs_n, reward_n, _, _ = env.step(action_n)

            _reward_n = reward_n[1:]
            _reward_n.append(reward_n[0])
            reward_n = _reward_n
            next_action_n = [agent.tgt_act(new_obs_n) for i, agent in enumerate(trainers)]

            # Collect experience for replay buffer
            for i, agent in enumerate(trainers):
                agent.replay_buffer.push(obs_n, action_n, reward_n[i], new_obs_n, next_action_n)
            
            if batch_update:
                if len(trainers[0].replay_buffer) < trainers[0].max_replay_buffer_len:
                # if len(trainers[0].replay_buffer) < 500:
                    batch_index = None
                    for i in range(num_agent):
                        actor_loss_episode[i].append(.0)
                        critic_loss_episode[i].append(.0)
                        q_value_episode[i].append(.0)
                        rewards_episode[i].append(reward_n[i])
                    pass
                else:
                    buffer_len = len(trainers[0].replay_buffer)
                    batch_index = random.sample(range(buffer_len), trainers[0].batch_size)
                        
                        
                    for i, agent in enumerate(trainers):
                        critic_loss, q_value = agent.batch_update_fully_q_train(update_gap, batch_index)
                        critic_loss_episode[i].append(critic_loss)
                        q_value_episode[i].append(q_value)
                        rewards_episode[i].append(reward_n[i])
                    if critic_loss !=0:
                        if args.attack_mode == 'normal':
                            aggregate_grads(trainers, args.device)
                        else:
                            byzantine_grads(trainers, malic_list, args.attack_mode, args.robust_mode, args.device)
                        for i, agent in enumerate(trainers):
                            agent.q_optimizer.step()
                            agent.make_update_exp(agent.q_network, agent.target_q_network)

                        

                        for i, agent in enumerate(trainers):
                            actor_loss = agent.batch_update_fully_p_train(update_gap, batch_index)                                                    
                            actor_loss_episode[i].append(actor_loss)

                        para_norm, grad_norm = cal_layer_norm(trainers)
                        para_vec_norm, grad_vec_norm = cal_vec_norm(trainers)
                        para_norms.append(para_norm)
                        grad_norms.append(grad_norm)
                        para_vec_norms.append(para_vec_norm)
                        grad_vec_norms.append(grad_vec_norm)

                        first_para_norm, first_grad_norm, second_para_norm, second_grad_norm = cal_half_layer_norm(trainers)
                        first_para_vec_norm, first_grad_vec_norm, second_para_vec_norm, second_grad_vec_norm = cal_half_vec_norm(trainers)
                        first_para_norms.append(first_para_norm)
                        second_para_norms.append(second_para_norm)
                        first_para_vec_norms.append(first_para_vec_norm)
                        second_para_vec_norms.append(second_para_vec_norm)
                        first_grad_norms.append(first_grad_norm)
                        second_grad_norms.append(second_grad_norm)
                        first_grad_vec_norms.append(first_grad_vec_norm)
                        second_grad_vec_norms.append(second_grad_vec_norm)
                    else:
                        for i, agent in enumerate(trainers):
                            actor_loss_episode[i].append(.0)
            obs_n = new_obs_n
            action_n = next_action_n
        for i in range(num_agent):
            total_rewards[i].append(np.sum(rewards_episode[i]))
            actor_losses[i].append(np.mean(actor_loss_episode[i]))
            critic_losses[i].append(np.mean(critic_loss_episode[i]))
            q_values[i].append(np.mean(q_value_episode[i]))
    
    
    if root_path is None:
        root_path = os.path.join(args.fig_path, 'agent_{}'.format(num_agent))
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    seed_path = os.path.join(root_path, 'seed_{}'.format(seed))
    if not os.path.exists(seed_path):
        os.mkdir(seed_path)
    attack_mode_path = os.path.join(seed_path, args.attack_mode)
    if not os.path.exists(attack_mode_path):
        os.mkdir(attack_mode_path)
    num_malicious_path = os.path.join(attack_mode_path, "num_malic_{}".format(args.num_malicious))
    if not os.path.exists(num_malicious_path):
        os.mkdir(num_malicious_path)
    robust_mode_path = os.path.join(num_malicious_path, args.robust_mode)
    if not os.path.exists(robust_mode_path):
        os.mkdir(robust_mode_path)
    
    np.save(os.path.join(robust_mode_path, "total_rewards.npy"), np.array(total_rewards))
    np.save(os.path.join(robust_mode_path, "actor_losses.npy"), np.array(actor_losses))
    np.save(os.path.join(robust_mode_path, "critic_losses.npy"), np.array(critic_losses))
    np.save(os.path.join(robust_mode_path, "q_values.npy"), np.array(q_values))
    np.save(os.path.join(robust_mode_path, "layer_para_norm.npy"), np.array(para_norms))
    np.save(os.path.join(robust_mode_path, "layer_grad_norm.npy"), np.array(grad_norms))
    np.save(os.path.join(robust_mode_path, "vec_para_norm.npy"), np.array(para_vec_norms))
    np.save(os.path.join(robust_mode_path, "vec_grad_norm.npy"), np.array(grad_vec_norms))
    np.save(os.path.join(robust_mode_path, "first_layer_para_norm.npy"), np.array(first_para_norms))
    np.save(os.path.join(robust_mode_path, "second_layer_para_norm.npy"), np.array(second_para_norms))
    np.save(os.path.join(robust_mode_path, "first_layer_grad_norm.npy"), np.array(first_grad_norms))
    np.save(os.path.join(robust_mode_path, "second_layer_grad_norm.npy"), np.array(second_grad_norms))
    np.save(os.path.join(robust_mode_path, "first_vec_para_norm.npy"), np.array(first_para_vec_norms))
    np.save(os.path.join(robust_mode_path, "second_vec_para_norm.npy"), np.array(second_para_vec_norms))
    np.save(os.path.join(robust_mode_path, "first_vec_grad_norm.npy"), np.array(first_grad_vec_norms))
    np.save(os.path.join(robust_mode_path, "second_vec_grad_norm.npy"), np.array(second_grad_vec_norms))

    
    plot_all(robust_mode_path, num_agent, update_gap, load=True, log=True)
    plot_all(robust_mode_path, num_agent, update_gap, load=True, log=False)

def parse_args():
    parser = argparse.ArgumentParser("Reinforce Learning for Multi-Agent Environments")

    parser.add_argument("--scenario", type=str, default="spread_fiveagent_no_diff_dist", help='environment')
    parser.add_argument("--num-agent", type=int, default=5, help='number of agents')

    parser.add_argument("--update-gap", type=int, default=10, help='frequency for update')
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-episode-len", type=int, default=25)
    parser.add_argument("--num-episodes", type=int, default=3000)
    parser.add_argument("--replay-buffer-size", type=int, default=1000000)

    parser.add_argument("--num-malicious", type=int, default=0)
    parser.add_argument("--attack-mode", type=str, default="normal", help="the method used for attacks")
    parser.add_argument("--robust-mode", type=str, default='mean', help='the method used for roubuts aggregation')

    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--fig-path", type=str, default="./figures", help="directory in which training state and model should be saved")
    parser.add_argument("--root-path", type=str, default=None, help="root directory in which figures should be saved")
    parser.add_argument("--plot", action="store_true", default=False)

    return parser.parse_args()

if __name__ == '__main__':

    for seed in tqdm(range(5)):
        np.random.seed(0)
        arglist = parse_args()
        malic_list = np.random.choice(range(arglist.num_agent), size=arglist.num_malicious)
        print("malic_list:{}".format(malic_list))
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        random.seed(seed)


        if arglist.root_path is not None:
            exp_fully(arglist, seed, malic_list, os.path.join(arglist.fig_path, arglist.root_path))
        else:
            exp_fully(arglist, seed, malic_list)


