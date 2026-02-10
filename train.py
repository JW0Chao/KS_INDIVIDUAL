import torch
import torch.nn.functional as F
import numpy as np
import os
import utils
import model

from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric

BATCH_SIZE = 200
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001


class Trainer:

    def __init__(self, state_dim, action_dim, action_lim, ram, device, Test, model_dir="./Models", buffer_dir="./Buffer"):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.action_lim  = action_lim
        self.count       = 0
        self.update      = 0
        self.ram         = ram
        self.model_dir   = model_dir
        self.buffer_dir  = buffer_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.buffer_dir, exist_ok=True)
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=100/action_dim, desired_action_stddev=1e-6, adaptation_coefficient=1.1)
        self.noise       = utils.OrnsteinUhlenbeckActionNoise(action_dim=self.action_dim,
                                                        init=self.action_lim*0.2,theta=0.005,sigma=0.001)
        self.device = device

        self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(device)
        self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(device)
        self.actor_pert   = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

        self.critic = model.Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic = model.Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)
        
        self.perturb_actor_parameters(Test)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0

    def get_action(self, state, Test=False, noise=True, param=True):
        if Test:
            noise = None
            param = None
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = torch.from_numpy(state).to(self.device)
        self.actor.eval()
        self.actor_pert.eval()
        if param is not None: 
            new_action = self.actor_pert.forward(state).detach().data.cpu().numpy()
        else:
            new_action = self.actor.forward(state).detach().data.cpu().numpy()
        
        self.actor.train()
        if noise is not None:
            new_action += (self.noise.sample()*(np.random.randint(2, size=self.action_dim)*2-1))
       
        return new_action


    def optimize(self,Test):
        if Test:
            return
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        self.count = self.count+1
        s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)
        
        s1 = torch.from_numpy(s1).to(self.device)
        a1 = torch.from_numpy(a1).to(self.device)
        r1 = torch.from_numpy(r1).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        next_val = self.target_critic.forward(s2, a2).detach().view(-1, 1)
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r1.view(-1, 1) + GAMMA * next_val
        # y_pred = Q( s1, a1)
        y_predicted = self.critic.forward(s1, a1).view(-1, 1)
        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.last_critic_loss = loss_critic.item()
        self.critic_optimizer.step()
        
        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        #Silver 2014 DPG
        loss_actor = -1*torch.mean(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.last_actor_loss = loss_actor.item()
        self.actor_optimizer.step()
        
        self.perturb_actor_parameters(Test)
        utils.soft_update(self.target_actor, self.actor, TAU)
        utils.soft_update(self.target_critic, self.critic, TAU)

    def perturb_actor_parameters(self,Test):
        if Test:
            return
        """Apply parameter noise to actor model, for exploration"""
        utils.hard_update(self.actor_pert, self.actor)
        params = self.actor_pert.state_dict()
        for name in params:
            if 'ln' in name: 
                pass
            param = params[name]
            param += (torch.randn(param.shape) * self.param_noise.current_stddev).to(self.device)
            
    def update_pert(self,Test):
        if Test:
            return
        observ = self.ram.t[-min(self.ram.maxDistSize,self.ram.cont):]
        states = np.float32([tran[0] for tran in [self.ram.buffer[j] for j in observ]])
        unperturbed_actions = self.get_action(states, noise=None, param=None)
        #perturbed_actions = np.float32([tran[1] for tran in [self.ram.buffer[j] for j in observ]])
        perturbed_actions = []
        for case in states:
            self.perturb_actor_parameters(Test)
            perturbed_actions.append(self.get_action(case, noise=None, param=True))


        ddpg_dist = ddpg_distance_metric(perturbed_actions, unperturbed_actions)
        self.param_noise.adapt(ddpg_dist)
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim,init=self.param_noise.current_stddev)
        self.update = ddpg_dist/self.param_noise.desired_action_stddev-1
        self.update = np.sign(self.update)*self.param_noise.current_stddev
    
    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), os.path.join(self.model_dir, str(episode_count) + '_target_actor.pt'))
        torch.save(self.target_critic.state_dict(), os.path.join(self.model_dir, str(episode_count) + '_target_critic.pt'))
        torch.save(self.actor.state_dict(), os.path.join(self.model_dir, str(episode_count) + '_actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join(self.model_dir, str(episode_count) + '_critic.pt'))
        np.savetxt(os.path.join(self.buffer_dir, str(episode_count) + 'reward.dat'), np.array([self.count, episode_count,
                                                    self.param_noise.current_stddev, 
                                                    self.param_noise.adaptation_coefficient,
                                                    self.param_noise.desired_action_stddev]))
        self.ram.save_buffer(episode_count)
        print('Models saved successfully')

    def load_models(self, episode, Test):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load(os.path.join(self.model_dir, str(episode) + '_actor.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(self.model_dir, str(episode) + '_critic.pt')))
        self.target_actor.load_state_dict(torch.load(os.path.join(self.model_dir, str(episode) + '_target_actor.pt')))
        self.target_critic.load_state_dict(torch.load(os.path.join(self.model_dir, str(episode) + '_target_critic.pt')))        
        if not Test:
            rew = np.loadtxt(os.path.join(self.buffer_dir, str(episode) + 'reward.dat'))   #load: reward, iterations, r, _ep
            self.ram.load_buffer(episode)                    #load: previous buffer
            self.count = int(rew[0])
            self.noise.iniOU(self.count)                #initialise noise level
            self.param_noise.current_stddev =  rew[2]
            self.param_noise.adaptation_coefficient = rew[3]
            self.param_noise.desired_action_stddev = rew[4]
        print('Models loaded succesfully')
        
