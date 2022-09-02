import torch
from tqdm import tqdm
from torch import autograd
import torch.optim as optim
import torch.nn.functional as F
from .utils import get_foveal_weights


class GAIL():
    def __init__(self, discriminator, milestones, state_enc, device, lr,
                 betas):
        self.discriminator = discriminator
        self.state_enc = state_enc
        self.device = device
        self.optimizer = optim.Adam(self.discriminator.parameters(),
                                    lr=lr,
                                    betas=betas)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.1)
        self.update_counter = 0

    def compute_ZS_grad_pen_(self, expert_inputs, lambda_=5):
        assert isinstance(expert_inputs, list), "wrong inputs format!"
        for i in range(len(expert_inputs)):
            expert_inputs[i] = expert_inputs[i].detach()

        expert_inputs[0].requires_grad = True
        # expert_inputs[1].requires_grad = True
        # expert_inputs[2].requires_grad = True
        expert_inputs[3].requires_grad = True

        disc = torch.sigmoid(self.discriminator(*expert_inputs, True))
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(outputs=disc,
                             inputs=(expert_inputs[0], expert_inputs[3]),
                             grad_outputs=ones,
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def compute_grad_pen(self,
                         expert_states,
                         expert_act,
                         expert_task,
                         lambda_=5):

        mixup_states = [x.detach() for x in expert_states]
        mixup_actions = expert_act.detach()
        mixup_tids = expert_task.detach()

        # for x in mixup_states:
        #     x.requires_grad = True
        mixup_states[0].requires_grad = True
        # mixup_actions.requires_grad = True
        # mixup_tids.requires_grad = True

        mixup_data = (*mixup_states, mixup_actions, mixup_tids)
        disc = torch.sigmoid(self.discriminator(*mixup_data,
                                                get_weights=False))
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(outputs=disc,
                             inputs=mixup_states,
                             grad_outputs=ones,
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self,
               true_data_loader,
               fake_data,
               iter_num=1,
               noisy_label_ratio=0):
        running_loss = 0.0
        print_every = fake_data.sample_num // (5 * fake_data.batch_size) + 1
        avg_loss = 0.0
        D_real, D_fake, D_grad = 0.0, 0.0, 0.0
        fake_sample_num = 0
        real_sample_num = 0

        for i_iter in range(iter_num):
            fake_data_generator = iter(fake_data.get_generator())
            pbar = tqdm(enumerate(true_data_loader))
            for i_batch, true_batch in pbar:
                if i_batch == len(fake_data):
                    break
                fake_batch = next(fake_data_generator)

                with torch.no_grad():
                    if self.state_enc is None:
                        real_S = true_batch['true_state'].to(self.device)
                    else:
                        real_S = self.state_enc(true_batch['true_state'].to(
                            self.device))
                    real_A = true_batch['true_action'].to(self.device)
                    real_tids = true_batch['task_id'].to(self.device)
                    
                x_fake = list(fake_batch[0])
                fake_num, real_num = x_fake[-2].size(0), real_A.size(0)
                if fake_num == 0 or real_num == 0:
                    break

                # TODO(zbyang): pass in state representation method as flag to
                # control the model inputs.
                if fake_data.is_composite_state:
                    real_fixs = true_batch['normalized_fixations'].to(
                        self.device)
                    x_real = [real_S, real_fixs, real_A, real_tids]
                    # print('zb_gail_real', real_fixs.shape, real_fixs)
                    # print('zb_gail_fake', x_fake[1].shape, x_fake[1])
                else:
                    x_real = [real_S, real_A, real_tids]

                if fake_data.is_zero_shot:
                    real_hrs = true_batch['hr_feats'].to(self.device)
                    fake_hrs = fake_batch[-1]
                    x_real.append(real_hrs)
                    x_fake.append(fake_hrs)

                real_outputs = self.discriminator(*x_real)
                fake_outputs = self.discriminator(*x_fake)

                real_labels = true_batch['true_or_fake'].to(self.device)
                fake_labels = torch.zeros(fake_outputs.size()).to(self.device)

                # randomly flip labels of training data in order to
                # increase training stability
                if noisy_label_ratio > 0:
                    flip_num = int(real_labels.size(0) * noisy_label_ratio)
                    ind = torch.randint(real_labels.size(0), (flip_num, ))
                    real_labels[ind] = 0
                    fake_labels[ind] = 1

                expert_loss = F.binary_cross_entropy_with_logits(
                    real_outputs, real_labels)
                policy_loss = F.binary_cross_entropy_with_logits(
                    fake_outputs, fake_labels)

                gail_loss = expert_loss + policy_loss
                # if fake_data.is_zero_shot:
                #     grad_pen = self.compute_ZS_grad_pen_(x_real)
                # else:
                #     real_foveal_weights = get_foveal_weights(
                #         real_fixs, 256, 160)
                #     grad_pen = self.compute_grad_pen(
                #         (real_S, real_foveal_weights), real_A, real_tids)
                grad_pen = torch.tensor(0)

                self.optimizer.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                if i_iter == iter_num - 1:
                    avg_loss += gail_loss.item()
                    real_sample_num += real_num
                    fake_sample_num += fake_num
                    D_real += torch.sum(torch.sigmoid(real_outputs)).item()
                    D_fake += torch.sum(torch.sigmoid(fake_outputs)).item()

                running_loss += gail_loss.item()
                D_grad += grad_pen.item()
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] gail_loss: {:.3f}, grad_loss: {:.3f}'
                    .format(i_iter, i_batch + 1, len(fake_data),
                            100. * (i_batch + 1) / len(fake_data),
                            gail_loss.item(), grad_pen.item()))
                self.update_counter += 1

        return (avg_loss / fake_data.sample_num, D_real / real_sample_num,
                D_fake / fake_sample_num)
