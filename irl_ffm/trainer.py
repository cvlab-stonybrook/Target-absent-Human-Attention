import os
import wandb
import torch
import datetime
import numpy as np
from tqdm import tqdm
from . import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from .ppo import PPO
from .ppg import PPG
from .gail import GAIL
from . import utils
from .data import RolloutStorage, FakeDataRollout


class Trainer(object):
    def __init__(self, model, loaded_step, env, dataset, device, hparams):
        # setup logger
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")
        self.log_dir = os.path.join(hparams.Train.log_root, "log_" + date)
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Wandb Initialization
        wandb.init(project='scanpath-prediction', notes=self.log_dir)
        wandb.config.update(hparams.to_dict())

        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoint_every = hparams.Train.checkpoint_every
        self.max_checkpoints = hparams.Train.max_checkpoints

        self.loaded_step = loaded_step
        self.env = env['train']
        self.env_valid = env['valid']
        self.generator = model['gen']
        self.discriminator = model['disc']
        self.bbox_annos = dataset['bbox_annos']
        self.clusters = dataset['fix_clusters']
        if hparams.Data.TAP == 'TP':
            self.human_mean_cdf = dataset['human_cdf']
        self.device = device
        self.TAP = hparams.Data.TAP
        self.has_stop = hparams.Data.has_stop
        self.gt_scanpaths = dataset['valid_scanpaths']
        self.prior_maps = dataset['prior_maps']

        self.im_h = hparams.Data.im_h
        self.im_w = hparams.Data.im_w
        self.is_zero_shot = hparams.Train.zero_shot

        # image dataloader
        self.batch_size = hparams.Train.batch_size
        self.disc_batch_size = hparams.Train.disc_batch_size
        n_workers = hparams.Train.n_workers
        self.train_img_loader = DataLoader(dataset['img_train'],
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=n_workers,
                                           pin_memory=True)
        self.valid_img_loader_TP = DataLoader(dataset['img_valid_TP'],
                                              batch_size=self.batch_size,
                                              shuffle=False,
                                              num_workers=n_workers,
                                              pin_memory=True)
        self.valid_img_loader_TA = DataLoader(dataset['img_valid_TA'],
                                              batch_size=self.batch_size,
                                              shuffle=False,
                                              num_workers=n_workers,
                                              pin_memory=True)

        # human gaze dataloader
        # class balanced sampler
        use_balanced_sampler = hparams.Train.balance
        if use_balanced_sampler:
            labels = np.array(
                [x[-1] for x in dataset['gaze_train'].fix_labels])
            a, b = np.unique(labels, return_counts=True)
            label_count = dict(zip(a, b))
            # weights = torch.Tensor([1 / label_count[x] for x in labels])

            # uniform weights for each item
            weights = torch.ones(len(labels))

            # lower the sampler rate for stopping action
            print("num of actions = {}, num of stop actions = {}".format(
                len(labels), label_count[hparams.Data.patch_count]))
            weights[labels == hparams.Data.
                    patch_count] = hparams.Data.stop_sampling_ratio

            sampler = WeightedRandomSampler(weights, len(labels))
        else:
            sampler = None

        self.train_HG_loader = DataLoader(dataset['gaze_train'],
                                          batch_size=self.disc_batch_size,
                                          shuffle=not use_balanced_sampler,
                                          sampler=sampler,
                                          num_workers=n_workers,
                                          pin_memory=True)
        self.valid_HG_loader = DataLoader(dataset['gaze_valid'],
                                          batch_size=self.disc_batch_size,
                                          shuffle=False,
                                          num_workers=n_workers,
                                          pin_memory=True)

        # training parameters
        self.gamma = hparams.Train.gamma
        self.adv_est = hparams.Train.adv_est
        self.tau = hparams.Train.tau
        self.max_traj_len = hparams.Data.max_traj_length
        self.n_epoches = hparams.Train.num_epoch
        self.n_steps = hparams.Train.num_step
        self.gail_iter_num = hparams.Train.num_iter
        self.disc_noisy_label_ratio = hparams.Train.gail_noisy_label_ratio
        self.gen_update_freq = hparams.Train.gen_update_freq
        self.disc_update_freq = hparams.Train.disc_update_freq
        self.patch_num = hparams.Data.patch_num
        self.eval_every = hparams.Train.evaluate_every

        if hparams.RL_algo.name == 'ppo':
            self.rl_learner = PPO(
                self.generator, hparams.RL_algo.lr, hparams.Train.adam_betas,
                hparams.RL_algo.clip_param, hparams.RL_algo.num_epoch,
                hparams.RL_algo.batch_size, hparams.RL_algo.value_coef,
                hparams.RL_algo.entropy_coef)
        elif hparams.RL_algo.name == 'ppg':
            self.rl_learner = PPG(
                self.generator, model['vnet'], hparams.RL_algo.lr,
                hparams.Train.adam_betas, hparams.RL_algo.clip_param,
                hparams.RL_algo.num_epoch, hparams.RL_algo.batch_size,
                hparams.RL_algo.buffer_size, hparams.RL_algo.num_aux_epoch,
                hparams.RL_algo.entropy_coef)
        else:
            NotImplementedError

        self.gail = GAIL(
            self.discriminator,
            hparams.Train.gail_milestones,
            self.env.state_enc if hparams.Train.repr == 'CFI' else None,
            device,
            lr=hparams.Train.gail_lr,
            betas=hparams.Train.adam_betas)

    def train(self):
        self.generator.train()
        self.discriminator.train()
        self.global_step = self.loaded_step
        for i_epoch in range(self.n_epoches):
            for i_batch, batch in enumerate(self.train_img_loader):
                # run policy to collect trajactories
                print("\ngenerating state-action pairs...")
                trajs_all = []
                self.env.set_data(batch)
                return_train = 0.0
                for i_step in range(self.n_steps):
                    with torch.no_grad():
                        self.env.reset()
                        trajs = utils.collect_trajs(
                            self.env,
                            self.generator,
                            self.patch_num,
                            self.max_traj_len,
                            is_zero_shot=self.is_zero_shot)
                        trajs_all.extend(trajs)
                smp_num = np.sum(list(map(lambda x: x['length'], trajs_all)))
                print("[{} {}] Collected {} state-action pairs".format(
                    i_epoch, i_batch, smp_num))
                if self.has_stop:
                    wandb.log({"avg_sp_len": smp_num / len(trajs_all)},
                              step=self.global_step)

                # train discriminator (reward and value function)
                if i_batch % self.disc_update_freq == 0:
                    print("updating discriminator (step={})...".format(
                        self.gail.update_counter))

                    fake_data = FakeDataRollout(trajs_all,
                                                self.disc_batch_size)
                    D_loss, D_real, D_fake = self.gail.update(
                        self.train_HG_loader,
                        fake_data,
                        iter_num=self.gail_iter_num,
                        noisy_label_ratio=self.disc_noisy_label_ratio)

                    wandb.log(
                        {
                            "disc_fake_loss": D_fake,
                            "disc_real_loss": D_real
                        },
                        step=self.global_step)

                    print("Done updating discriminator!")

                # evaluate generator/policy
                if self.global_step > 0 and self.global_step % self.eval_every == 0:
                    self.generator.eval()
                    print("evaluating policy...")

                    # Generate 1 scanpath in a greedy fashion for each image
                    self.evaluate(self.valid_img_loader_TP,
                                  self.gt_scanpaths, 'TP',
                                  sample_action=False,
                                  sample_scheme='greedy')
                    self.evaluate(self.valid_img_loader_TA,
                                  self.gt_scanpaths, 'TA',
                                  sample_action=False,
                                  sample_scheme='greedy')

                    # Sample 30 scanpaths for each image
                    self.evaluate(self.valid_img_loader_TP,
                                  self.gt_scanpaths, 'TP',
                                  sample_action=True)
                    self.evaluate(self.valid_img_loader_TA,
                                  self.gt_scanpaths, 'TA',
                                  sample_action=True)

                    
                # update generator/policy on every n_critic iter
                if i_batch % self.gen_update_freq == self.gen_update_freq - 1:
                    print("updating policy...")
                    # update reward and value
                    with torch.no_grad():
                        for i in range(len(trajs_all)):
                            states = trajs_all[i]["curr_states"]
                            actions = trajs_all[i]["actions"].unsqueeze(1)
                            tids = trajs_all[i]['task_id']
                            if self.is_zero_shot:
                                inputs = (states, actions, tids,
                                          trajs_all[i]['hr_feats'])
                            else:
                                if isinstance(states, list):
                                    inputs = (*states, actions, tids)
                                else:
                                    inputs = (states, actions, tids)
                            rewards = F.logsigmoid(self.discriminator(*inputs))

                            if self.has_stop:
                                # repeating last reward after stopping action
                                num_fillup = self.max_traj_len - trajs_all[i][
                                    'length']
                                if num_fillup > 0:
                                    rewards[-1] = rewards[-1] * (1 +
                                                                 num_fillup)

                            trajs_all[i]["rewards"] = rewards

                    return_train = utils.process_trajs(trajs_all,
                                                       self.gamma,
                                                       mtd=self.adv_est,
                                                       tau=self.tau)
                    print('average return = {:.3f}'.format(return_train))

                    # update policy
                    rollouts = RolloutStorage(trajs_all,
                                              shuffle=True,
                                              norm_adv=True)
                    value_loss, action_loss, dist_entropy, loss = self.rl_learner.learn(
                        rollouts)
                    wandb.log(
                        {
                            "ppo_return": return_train,
                            "ppo_loss": loss,
                            "ppo_value_loss": value_loss,
                            "ppo_action_loss": action_loss,
                            "ppo_entropy": dist_entropy
                        },
                        step=self.global_step)

                # checkpoints
                if self.global_step % self.checkpoint_every == 0 and \
                   self.global_step > 0:
                    utils.save(global_step=self.global_step,
                               model=self.generator,
                               optim=self.rl_learner.optimizer,
                               name='generator',
                               pkg_dir=self.checkpoints_dir,
                               is_best=True,
                               max_checkpoints=self.max_checkpoints)
                    utils.save(global_step=self.global_step,
                               model=self.discriminator,
                               optim=self.gail.optimizer,
                               name='discriminator',
                               pkg_dir=self.checkpoints_dir,
                               is_best=True,
                               max_checkpoints=self.max_checkpoints)

                self.global_step += 1


    def evaluate(self,
                 dataloader,
                 gt_scanpaths, TAP,
                 sample_action=True,
                 sample_scheme=''):

        self.generator.eval()
        
        # generating scanpaths
        all_actions = []
        for i_sample in tqdm(range(30)):
            for batch in dataloader:
                self.env_valid.set_data(batch)
                img_names_batch = batch['img_name']
                cat_names_batch = batch['cat_name']
                cond_batch = batch['condition']
                with torch.no_grad():
                    trajs = utils.collect_trajs(
                        self.env_valid,
                        self.generator,
                        self.patch_num,
                        self.max_traj_len,
                        is_eval=True,
                        sample_action=sample_action,
                        is_zero_shot=self.is_zero_shot)
                    all_actions.extend([
                        (cat_names_batch[i], img_names_batch[i],
                         cond_batch[i], trajs[i]['actions'])
                        for i in range(self.env_valid.batch_size)
                    ])
            if not sample_action:
                break

        scanpaths = utils.actions2scanpaths(
            all_actions, self.patch_num)

        metrics_dict = {}
        if TAP == 'TP':
            utils.cutFixOnTarget(scanpaths, self.bbox_annos)
            # search effiency
            mean_cdf, _ = utils.compute_search_cdf(
                scanpaths, self.bbox_annos, self.max_traj_len)
            metrics_dict.update(
                dict(
                    zip([
                        f"TFP_top{i}"
                        for i in range(1, len(mean_cdf))
                    ], mean_cdf[1:])))

            # probability mismatch
            sad = np.sum(np.abs(self.human_mean_cdf - mean_cdf))
            metrics_dict["prob_mismatch"] = sad

        # sequence score
        ss_4steps = metrics.get_seq_score(scanpaths, self.clusters,
                                          4, True)
        ss = metrics.get_seq_score(scanpaths, self.clusters,
                                   self.max_traj_len, False)

        metrics_dict.update({
            f"{TAP}_seq_score_max": ss,
            f"{TAP}_seq_score_4steps": ss_4steps,
            'epoch': self.global_step / float(len(self.train_img_loader))
        })

        # temporal spatial saliency metrics
        if sample_action:
            ig, cc, nss = metrics.compute_spatial_metrics_by_step(
                scanpaths, gt_scanpaths, self.im_w, self.im_h,
                self.prior_maps)
            metrics_dict.update({
                f'{TAP}_IG_all': ig,
                f'{TAP}_CC_all': cc,
                f'{TAP}_NSS_all': nss
            })
            for i in [2, 4, 6]:
                ig, cc, nss = metrics.compute_spatial_metrics_by_step(
                    scanpaths, gt_scanpaths, self.im_w, self.im_h,
                    self.prior_maps, i)
                metrics_dict.update({
                    f'{TAP}_IG_{i}': ig,
                    f'{TAP}_CC_{i}': cc,
                    f'{TAP}_NSS_{i}': nss
                })

        if not sample_action:
            prefix = sample_scheme + '_'
            keys = list(metrics_dict.keys())
            for k in keys:
                metrics_dict[prefix + k] = metrics_dict.pop(k)

        wandb.log(metrics_dict, step=self.global_step)

        self.generator.train()

