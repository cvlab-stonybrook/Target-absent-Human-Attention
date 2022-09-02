import torch
from tqdm import tqdm
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim


class ReplayBuffer(object):
    def __init__(self, rollouts):
        self.obs_fovs = rollouts.obs_fovs
        self.tids = rollouts.tids
        self.values = rollouts.returns
        self.probs = None  # to be computed at training
        self.n_samples = self.obs_fovs.size(0)

    def append(self, new_rollouts):
        self.obs_fovs = torch.cat([self.obs_fovs, new_rollouts.obs_fovs])
        self.values = torch.cat([self.values, new_rollouts.returns])
        self.tids = torch.cat([self.tids, new_rollouts.tids])
        self.n_samples = self.obs_fovs.size(0)

    @torch.no_grad()
    def evaluate(self, policy, minibatch_size):
        all_probs = []
        data = self.get_generator(4 * minibatch_size, False, False)
        for batch in data:
            probs, _ = policy(*batch, act_only=True)
            all_probs.append(probs)
        self.probs = torch.cat(all_probs)
        assert self.probs.size(0) == self.n_samples, 'probs size wrong'
        
    def get_generator(self,
                      minibatch_size,
                      return_probs=True,
                      drop_last=True):
        minibatch_size = min(self.n_samples, minibatch_size)
        sampler = BatchSampler(SubsetRandomSampler(range(self.n_samples)),
                               minibatch_size,
                               drop_last=drop_last)
        for ind in sampler:
            obs_fov_batch = self.obs_fovs[ind]
            tids_batch = self.tids[ind]
            values_batch = self.values[ind]
            if return_probs:
                probs_batch = self.probs[ind]
                yield (obs_fov_batch, tids_batch), values_batch, probs_batch
            else:
                yield (obs_fov_batch, tids_batch)

            
class PPG():
    def __init__(self,
                 policy_net,
                 value_net,
                 lr,
                 betas,
                 clip_param,
                 num_epoch,
                 batch_size,
                 buffer_size,
                 num_aux_epoch,
                 entropy_coef=0.1):

        self.policy_net = policy_net
        self.value_net = value_net
        self.clip_param = clip_param
        self.num_epoch = num_epoch
        self.minibatch_size = batch_size
        self.entropy_coef = entropy_coef
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(
            self.value_net.parameters()), lr=lr, betas=betas)
        self.value_loss_fun = torch.nn.MSELoss()
        self.kld_fn = torch.nn.KLDivLoss()
        self.num_aux_epoch = num_aux_epoch
        self.buffer_size = buffer_size
        self._reset_buffer()
        
    def _reset_buffer(self):
        self.buffer = None
        self.buffer_count = 0

    def evaluate_actions(self, obs_batch, actions_batch):

        probs, _ = self.policy_net(*obs_batch, act_only=True)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions_batch)
        values = self.value_net(*obs_batch)

        return values, log_probs, dist.entropy().mean()

    def learn(self, rollouts):
        self.buffer_count += 1
        
        # buffer trajactories
        if self.buffer is None:
            self.buffer = ReplayBuffer(rollouts)
        else:
            self.buffer.append(rollouts)

        # policy phase: update policy and value fn using ppo
        avg_value_loss, avg_action_loss = 0, 0
        avg_entropy_loss = 0
        for e in range(self.num_epoch):
            data_generator = rollouts.get_generator(self.minibatch_size)

            pbar = tqdm(enumerate(data_generator))
            n_batch = int(rollouts.sample_num / self.minibatch_size)
            for i, sample in pbar:
                obs_batch, actions_batch, return_batch, \
                   old_action_log_probs_batch, adv_targ = sample

                self.optimizer.zero_grad()

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.evaluate_actions(
                    obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                entropy_loss = -dist_entropy * self.entropy_coef
                (action_loss + entropy_loss).backward()

                value_loss = self.value_loss_fun(
                    return_batch, values.squeeze())
                value_loss.backward()

                self.optimizer.step()

                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] VLoss: {:.3f}, PLoss: {:.3f}, ELoss: {:.3f}'
                    .format(e, i+1, n_batch,
                            100. * (i+1) / n_batch,
                            value_loss.item(), action_loss.item(),
                            entropy_loss.item()))

                avg_value_loss += value_loss.item()
                avg_action_loss += action_loss.item()
                avg_entropy_loss += entropy_loss.item()

        avg_value_loss /= (self.num_epoch * (i+1))
        avg_action_loss /= (self.num_epoch * (i+1))
        avg_entropy_loss /= (self.num_epoch * (i+1))
        avg_loss = avg_value_loss + avg_action_loss + avg_entropy_loss

        # auxiliary phase
        if self.buffer_count == self.buffer_size:
            print("\nperform auxiliary training with {} transitions...".format(
                self.buffer.n_samples))

            # compute probability dist for current policy (old)
            self.buffer.evaluate(self.policy_net, self.minibatch_size)

            for e in range(self.num_aux_epoch):
                data_generator = self.buffer.get_generator(self.minibatch_size)
                pbar = tqdm(enumerate(data_generator))
                n_batch = int(self.buffer.n_samples / self.minibatch_size)
                for i, sample in pbar:
                    obs_batch, values_targets, probs_targets = sample

                    self.optimizer.zero_grad()
                    
                    # update policy network
                    probs_p, values_p = self.policy_net(*obs_batch)
                    aux_loss = self.kld_fn(
                        torch.log(probs_p), probs_targets) + self.value_loss_fun(
                            values_p, values_targets)
                    aux_loss.backward()
                
                    # update value network
                    values = self.value_net(*obs_batch)
                    value_loss = self.value_loss_fun(
                        values_targets, values.squeeze())
                    value_loss.backward()
                
                    pbar.set_description(
                        'Train Epoch: {} [{}/{} ({:.0f}%)] VLoss: {:.3f}, AuxLoss: {:.3f}'
                        .format(e, i+1, n_batch,
                                100. * (i+1) / n_batch,
                                value_loss.item(), aux_loss.item()))

            # empty buffer
            self._reset_buffer()
            
        return (avg_value_loss, avg_action_loss,
                -avg_entropy_loss, avg_loss)
