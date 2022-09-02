import torch
from tqdm import tqdm
from torch.distributions import Categorical
import torch.optim as optim


class PPO():
    def __init__(self,
                 policy,
                 lr,
                 betas,
                 clip_param,
                 num_epoch,
                 batch_size,
                 value_coef=1.,
                 entropy_coef=0.1):

        self.policy = policy
        self.clip_param = clip_param
        self.num_epoch = num_epoch
        self.minibatch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=lr,
                                    betas=betas)
        self.value_loss_fun = torch.nn.SmoothL1Loss()

    def evaluate_actions(self, obs_batch, actions_batch):

        probs, values = self.policy(*obs_batch)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions_batch)

        return values, log_probs, dist.entropy().mean()

    def learn(self, rollouts):
        avg_value_loss, avg_action_loss = 0, 0
        avg_entropy_loss, avg_loss = 0, 0

        for e in range(self.num_epoch):
            data_generator = rollouts.get_generator(self.minibatch_size)

            pbar = tqdm(enumerate(data_generator))
            n_batch = -int(-rollouts.sample_num // self.minibatch_size)
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

                value_loss = self.value_loss_fun(
                    return_batch, values.squeeze()) * self.value_coef

                entropy_loss = -dist_entropy * self.entropy_coef
                loss = value_loss + action_loss + entropy_loss
                loss.backward()

                self.optimizer.step()

                avg_value_loss += value_loss.item()
                avg_action_loss += action_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_loss += loss.item()

                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] VLoss: {:.3f}, PLoss: {:.3f}, ELoss: {:.3f}'
                    .format(e, i+1, n_batch,
                            100. * (i+1) / n_batch,
                            value_loss.item(), action_loss.item(),
                            entropy_loss.item()))

        avg_value_loss /= (self.num_epoch * (i+1))
        avg_action_loss /= (self.num_epoch * (i+1))
        avg_entropy_loss /= (self.num_epoch * (i+1))
        avg_loss /= (self.num_epoch * (i+1))

        return (avg_value_loss, avg_action_loss,
                -avg_entropy_loss, avg_loss)
