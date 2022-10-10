import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
import copy

class SoftQ(object):
    def __init__(self, q_net, batch_size, device, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.device = device
        self.args = args
        self.actor = None

        self.log_alpha = torch.tensor(np.log(args.init_temp)).to(self.device)
        self.q_net = q_net
        self.train()
        
        self.target_net = None
        if self.args.use_target_network:
            self.target_net = copy.deepcopy(q_net)
            self.target_net.train()

        self.critic_optimizer = Adam(self.q_net.parameters(), lr=args.adam_lr,
                                     betas=args.adam_betas)

        self.layer_weight = torch.linspace(1, 0.2, 5).to(self.device)
        
    def train(self, training=True):
        self.training = training
        self.q_net.train(training)
        
    def eval(self, training=False):
        self.training = training
        self.q_net.eval()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.q_net

    @property
    def critic_target_net(self):
        return self.target_net

    def select_action(self,
                      state,
                      sample_action,
                      action_mask=None,
                      softmask=False,
                      eps=1e-12):
        with torch.no_grad():
            q = self.q_net(*state)
            probs = F.softmax(q/self.alpha, dim=1)
        if sample_action:
            if action_mask is not None:
                # prevent sample previous actions by re-normalizing probs
                if probs.size(-1) % 2 == 1:
                    probs[:, :-1][action_mask] = eps
                else:
                    probs[action_mask] = eps
                probs /= probs.sum(dim=1).view(probs.size(0), 1)

            m = Categorical(probs)
            actions = m.sample()
            return actions.view(-1), None
        else:
            if action_mask is not None:
                if probs.size(-1) % 2 == 1:
                    probs[:, :-1][action_mask] = eps
                else:
                    probs[action_mask] = eps
            actions = torch.argmax(probs, dim=1)

        return actions, None
    

    def getV(self, obs):
        q = self.q_net(*obs)
        v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return v

    def critic(self, obs, action, do_aux_task=False):
        if do_aux_task:
            q, cm = self.q_net(*obs, True)
            return q[torch.arange(action.size(0)), action], cm
        else:
            q = self.q_net(*obs)
            return q[torch.arange(action.size(0)), action]

    def get_targetV(self, obs):
        q = self.target_net(*obs)
        target_v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return target_v

    def update(self, replay_buffer, logger, step):
        obs, next_obs, action, reward, done = replay_buffer.get_samples(
            self.batch_size, self.device)

        losses = self.update_critic(obs, action, reward, next_obs, done,
                                    logger, step)

        if step % self.critic_target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return losses

    def update_critic(self, obs, action, reward, next_obs, done, logger,
                      step):

        with torch.no_grad():
            next_v = self.get_targetV(next_obs)
            y = reward + (1 - done) * self.gamma * next_v

        critic_loss = F.mse_loss(self.critic(obs, action), y)
        logger.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            'loss/critic': critic_loss.item()}

    # Save model parameters
    def save(self, path, suffix=""):
        critic_path = f"{path}{suffix}"
        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.q_net.state_dict(), critic_path)

    # Load model parameters
    def load(self, path, suffix=""):
        critic_path = f'{path}/{self.args.agent.name}{suffix}'
        print('Loading models from {}'.format(critic_path))
        self.q_net.load_state_dict(torch.load(critic_path, map_location=self.device))

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = torch.FloatTensor([action]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()


class SoftQ_wStop(object):
    def __init__(self, q_net, batch_size, device, args, has_stop):
        self.gamma = args.gamma
        try:
            self.num_actions = q_net.module.num_actions
        except AttributeError:
            self.num_actions = q_net.num_actions
        self.has_stop = has_stop
        self.batch_size = batch_size
        self.device = device
        self.args = args
        self.actor = None

        self.critic_target_update_frequency = 0
        self.log_alpha = torch.tensor(np.log(args.init_temp)).to(self.device)
        self.q_net = q_net
        self.train()
        
        self.target_net = None
        if self.args.use_target_network:
            self.target_net = copy.deepcopy(q_net)
            self.target_net.train()

        self.critic_optimizer = Adam(self.q_net.parameters(), lr=args.adam_lr,
                                     betas=args.adam_betas)

        self.layer_weight = torch.linspace(1, 0.2, 5).to(self.device)
        
    def train(self, training=True):
        self.training = training
        self.q_net.train(training)
        
    def eval(self, training=False):
        self.training = training
        self.q_net.eval()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.q_net

    @property
    def critic_target_net(self):
        return self.target_net

    def select_action(self,
                      state,
                      sample_action,
                      action_mask=None,
                      softmask=False,
                      eps=1e-12):
        with torch.no_grad():
            q = self.q_net(*state)
            if self.has_stop:
                q, s = q
                stop_probs = F.sigmoid(s).view(-1)
#                 stop = Bernoulli(stop_probs).sample() if sample_action else stop_probs > 0.5
                stop = stop_probs > 0.5
            else:
                stop = torch.zeros(q.size(0))
            probs = F.softmax(q/self.alpha, dim=1)

            if sample_action:
                if action_mask is not None:
                    # prevent sample previous actions by re-normalizing probs
                    probs[action_mask] = eps
                    probs /= probs.sum(dim=1).view(probs.size(0), 1)

                m = Categorical(probs)
                actions = m.sample()
            else:
                if action_mask is not None:
                    if probs.size(-1) % 2 == 1:
                        probs[:, :-1][action_mask] = eps
                    else:
                        probs[action_mask] = eps
                actions = torch.argmax(probs, dim=1)

            return actions, stop
    
    def getV(self, obs):
        q = self.q_net(*obs)
        if self.has_stop:
            q = q[0]
        v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return v

    def critic(self, obs, action, do_aux_task=False):
        if self.has_stop:
            # Replace the stop action (id=640) with a valid action (0) which 
            # will not be used for training the Q-net
            new_actions = action.clone()
            new_actions[action == self.num_actions] = 0
            action = new_actions
            
        if do_aux_task:
            out = self.q_net(*obs, True)
            if self.has_stop:
                q, s, cm = out
                return q[torch.arange(action.size(0)), action], s.view(-1), cm
            else:
                q, cm = out
                return q[torch.arange(action.size(0)), action], cm
        else:
            q = self.q_net(*obs)
            if self.has_stop:
                q = q[0]
            return q[torch.arange(action.size(0)), action]

    def get_targetV(self, obs):
        q = self.target_net(*obs)
        if self.has_stop:
            q = q[0]
        target_v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return target_v

    def update(self, replay_buffer, logger, step):
        obs, next_obs, action, reward, done = replay_buffer.get_samples(
            self.batch_size, self.device)

        losses = self.update_critic(obs, action, reward, next_obs, done,
                                    logger, step)

        if step % self.critic_target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return losses

    def update_critic(self, obs, action, reward, next_obs, done, logger,
                      step):

        with torch.no_grad():
            next_v = self.get_targetV(next_obs)
            y = reward + (1 - done) * self.gamma * next_v

        critic_loss = F.mse_loss(self.critic(obs, action), y)
        logger.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            'loss/critic': critic_loss.item()}

    # Save model parameters
    def save(self, path, suffix=""):
        critic_path = f"{path}{suffix}"
        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.q_net.state_dict(), critic_path)

    # Load model parameters
    def load(self, path, suffix=""):
        critic_path = f'{path}/{self.args.agent.name}{suffix}'
        print('Loading models from {}'.format(critic_path))
        self.q_net.load_state_dict(torch.load(critic_path, map_location=self.device))

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = torch.FloatTensor([action]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()
