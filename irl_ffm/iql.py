import torch


def focal_loss(pred, gt, alpha=2, beta=4, weights=None):
    ''' 
    Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)
    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                               alpha) * neg_weights * neg_inds

    if weights is not None:
        pos_loss *= weights
        neg_loss *= weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


# apply weights on the target
def focal_loss_v2(pred, gt, alpha=2, beta=4, weights=None):
    ''' 
    Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    if weights is not None:
        gt *= weights

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)
    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                               alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


# Minimal IQ-Learn objective
def get_concat_samples(policy_batch, expert_batch, only_expert_states, device):
    (online_batch_state, online_batch_next_state, online_batch_action,
     online_batch_done, online_batch_aux) = policy_batch

    (expert_batch_state, expert_batch_next_state, expert_batch_action,
     expert_batch_done, expert_batch_aux) = expert_batch

    if only_expert_states:
        is_expert = torch.ones_like(expert_batch_action, dtype=torch.bool)
        return expert_batch_state, expert_batch_next_state, online_batch_action, expert_done, is_expert

    batch_state = [
        torch.cat([x, y], dim=0).to(device)
        for x, y in zip(online_batch_state, expert_batch_state)
    ]
    batch_next_state = [
        torch.cat([x, y], dim=0).to(device)
        for x, y in zip(online_batch_next_state, expert_batch_next_state)
    ]
    batch_action = torch.cat([online_batch_action, expert_batch_action],
                             dim=0).to(device)
    batch_done = torch.cat([online_batch_done, expert_batch_done],
                           dim=0).to(device)
    batch_aux = torch.cat([online_batch_aux, expert_batch_aux],
                          dim=0).to(device)
    is_expert = torch.cat([
        torch.zeros_like(online_batch_action, dtype=torch.bool),
        torch.ones_like(expert_batch_action, dtype=torch.bool)
    ],
                          dim=0).to(device)

    return batch_state, batch_next_state, batch_action, batch_done, batch_aux, is_expert


def iq_learn_update(self, policy_batch, expert_batch,
                    only_expert_states=False):
    obs, next_obs, action, done, aux, is_expert = get_concat_samples(
        policy_batch, expert_batch, only_expert_states, self.device)

    losses = {}

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    done = done.to(torch.float32)
    current_Q = self.critic(obs, action)

    y = (1 - done) * self.gamma * self.getV(next_obs)
    if self.args.use_target_network:
        with torch.no_grad():
            y = (1 - done) * self.gamma * self.get_targetV(next_obs)

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()
    losses['softq_loss'] = loss.item()

    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    value_loss = (self.getV(obs) - y).mean()
    loss += value_loss
    losses['value_loss'] = value_loss.item()

    # Use χ2 divergence (adds a extra term to the loss)
    chi2_loss = 1 / (4 * self.args.alpha) * (reward**2).mean()
    loss += chi2_loss
    losses['chi2_loss'] = chi2_loss.item()
    ######

    losses['total_loss'] = loss.item()

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return losses


# Full IQ-Learn objective with other divergences and options
def irl_update_critic(self,
                      policy_batch,
                      expert_batch,
                      only_expert_states=False):
    obs, next_obs, action, done, aux, is_expert = get_concat_samples(
        policy_batch, expert_batch, only_expert_states, self.device)
    args = self.args

    losses = {}
    done = done.to(torch.float32)
    if args.type == "sqil":
        with torch.no_grad():
            target_Q = reward + (
                1 - done) * self.gamma * self.get_targetV(next_obs)

        current_Q = self.critic(obs, action)
        bell_error = F.mse_loss(current_Q, target_Q, reduction='none')
        loss = (bell_error[is_expert]).mean() + \
            args.method.sqil_lmbda * (bell_error[~is_expert]).mean()
        losses['sqil_loss'] = loss.item()

    elif args.type == "iq":
        # our method, calculate 1st term of loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        current_Q, cm = self.critic(obs, action, True)
        next_v = self.getV(next_obs)
        y = (1 - done) * self.gamma * next_v

        if args.use_target_network:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs)
                y = (1 - done) * self.gamma * next_v

        reward = (current_Q - y)[is_expert]

        with torch.no_grad():
            if args.div == "hellinger":
                phi_grad = 1 / (1 + reward)**2
            elif args.div == "kl":
                phi_grad = torch.exp(-reward - 1)
            elif args.div == "kl2":
                phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
            elif args.div == "kl_fix":
                phi_grad = torch.exp(-reward)
            elif args.div == "js":
                phi_grad = torch.exp(-reward) / (2 - torch.exp(-reward))
            else:
                phi_grad = 1
        loss = -(phi_grad * reward).mean()
        losses['softq_loss'] = loss.item()

        if args.value_loss == "v0":
            # calculate 2nd term for our loss
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - self.gamma) * v0
            loss += v0_loss
            losses['v0_loss'] = v0_loss.item()

        elif args.value_loss == "value":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.getV(obs) - y).mean()
            loss += value_loss
            losses['value_loss'] = value_loss.item()

        elif args.value_loss == "value_policy":
            # alternative 2nd term for our loss (use only policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.getV(obs) - y)[~is_expert].mean()
            loss += value_loss
            losses['value_policy_loss'] = value_loss.item()

        elif args.value_loss == "value_expert":
            # alternative 2nd term for our loss (use only expert states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.getV(obs) - y)[is_expert].mean()
            loss += value_loss
            losses['value_loss'] = value_loss.item()

        elif args.value_loss == "value_mix":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            w = args.mix_coeff
            value_loss = (w * (self.getV(obs) - y)[is_expert] + (1 - w) *
                          (self.getV(obs) - y)[~is_expert]).mean()
            loss += value_loss
            losses['value_loss'] = value_loss.item()

        elif args.value_loss == "skip":
            # No loss
            pass
    else:
        raise ValueError(f'This method is not implemented: {args.type}')

    if args.grad_pen:
        # add a gradient penalty to loss (W1 metric)
        policy_obs, policy_next_obs, policy_action, policy_done = policy_batch
        expert_obs, expert_next_obs, expert_action, expert_done = expert_batch
        expert_obs = [x.to(self.device) for x in expert_obs]
        policy_obs = [x.to(self.device) for x in policy_obs]
        gp_loss = self.critic_net.grad_pen(expert_obs,
                                           expert_action.to(self.device),
                                           policy_obs,
                                           policy_action.to(self.device),
                                           args.lambda_gp)
        losses['gp_loss'] = gp_loss.item()
        loss += gp_loss

    if args.use_chi:
        # Use χ2 divergence (adds a extra term to the loss)
        if args.use_target_network:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs)
        else:
            next_v = self.getV(next_obs)

        y = (1 - done) * self.gamma * next_v

        current_Q = self.critic(obs, action)
        reward = current_Q - y
        chi2_loss = 1 / (4 * args.alpha) * (reward**2)[is_expert].mean()
        loss += chi2_loss
        losses['chi2_loss'] = chi2_loss.item()

    if args.regularize:
        # Use χ2 divergence (adds a extra term to the loss)
        if args.use_target_network:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs)
        else:
            next_v = self.getV(next_obs)

        y = (1 - done) * self.gamma * next_v

        current_Q = self.critic(obs, action)
        reward = current_Q - y
        chi2_loss = 1 / (4 * args.alpha) * (reward**2).mean()
        loss += chi2_loss
        losses['regularize_loss'] = chi2_loss.item()

    if args.use_det_loss:
        if args.use_foveal_weight:
            with torch.no_grad():
                weights = get_foveal_weights(obs[1],
                                             128,
                                             80,
                                             p=self.q_net.FFN.fovea_size,
                                             k=self.q_net.FFN.amplitude,
                                             alpha=self.q_net.FFN.acuity)
                weights = torch.sum(
                    weights * self.layer_weight.unsqueeze(-1).unsqueeze(-1),
                    dim=1,
                    keepdim=True)
                weights = F.interpolate(weights,
                                        size=(20, 32),
                                        mode='bilinear')
        else:
            weights = None

        det_loss = focal_loss(cm, aux, weights=weights)

        loss += args.det_loss_weight * det_loss
        losses['detection_loss'] = det_loss.item()

    losses['total_loss'] = loss.item()

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    loss.backward()
    # step critic
    self.critic_optimizer.step()
    return losses


def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def irl_update(self, policy_batch, expert_batch, step, patch_num):

    losses = self.ilr_update_critic(policy_batch, expert_batch)

    if self.args.use_target_network and step % self.args.critic_target_update_frequency == 0:
        soft_update(self.critic_net, self.critic_target_net,
                    self.args.critic_tau)

    return losses
