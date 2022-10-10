"""
Foveated Feature Map Training Script.
This script is a simplified version of the training script in 
https://github.com/cvlab-stonybrook/Scanpath_Prediction
"""
import argparse
import os
import random

import numpy as np

import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from irl_ffm.builder import build
from irl_ffm.config import JsonConfig
from irl_ffm.evaluation import pack_model_inputs, evaluate
from irl_ffm.replay_buffer import Memory
from sklearn.linear_model import LogisticRegression
from irl_ffm.models import TerminatorWrapper
from irl_ffm.sql import SoftQ_wStop

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams',
                        type=str,
                        help='hyper parameters config file path')
    parser.add_argument('--dataset-root', type=str, help='dataset root path')
    return parser.parse_args()


def get_batch(policy_buffer, expert_demos_iter, expert_loader, batch_size):
    """Get batch ready for training."""
    try:
        expert_batch = next(expert_demos_iter)
    except StopIteration:
        expert_demos_iter = iter(expert_loader)
        expert_batch = next(expert_demos_iter)
    expert_batch_state = (expert_batch['true_state'],
                          expert_batch['normalized_fixations'],
                          expert_batch['task_id'])
    expert_batch_next_state = (expert_batch['true_state'],
                               expert_batch['next_normalized_fixations'],
                               expert_batch['task_id'])
    expert_batch_action = expert_batch['true_action'].squeeze()
    expert_batch_aux = expert_batch['centermaps']
    expert_batch_is_done = torch.ones_like(expert_batch_action,
                                           dtype=torch.bool)
    expert_batch = (expert_batch_state, expert_batch_next_state,
                    expert_batch_action, expert_batch_is_done,
                    expert_batch_aux)
    policy_batch = online_memory_replay.get_samples(batch_size)
    return policy_batch, expert_batch, expert_demos_iter


def get_Vs(agent, im_tensor, fixations, tids, actions=None):
    if actions is None:
        Vs = []
        for i in range(hparams.Data.patch_count):
            x = i % hparams.Data.patch_num[0] / hparams.Data.patch_num[0]
            y = i // hparams.Data.patch_num[0] / hparams.Data.patch_num[1]
            fixations[:, -1] = torch.Tensor([x, y])
            Vs.append(agent.getV((im_tensor, fixations, tids)))
        Vs = torch.stack(Vs).squeeze().transpose(0,1)
    else:
        x = actions % hparams.Data.patch_num[0] / hparams.Data.patch_num[0]
        y = actions // hparams.Data.patch_num[0] / hparams.Data.patch_num[1]
        fixations[:, -1] = torch.stack([x, y], dim=1)
        Vs = agent.getV((im_tensor, fixations, tids)).squeeze()
    return Vs


def get_max_Q(agent, dataloader):
    q_values_all, rewards, subj_ids, sp_lens, is_last = [], [], [], [], []
    history_fixs, durations = [], []
    i = 0
    with torch.no_grad():
        for expert_batch in tqdm(dataloader):
            expert_batch_state = (expert_batch['true_state'].to(device), 
                                  expert_batch['normalized_fixations'].to(device), 
                                  expert_batch['task_id'].to(device))
            history_fixs.append(expert_batch['normalized_fixations'])
            durations.append(expert_batch['duration'])
            q_values = agent.q_net(*expert_batch_state)
            q_max, actions = torch.max(q_values, dim=1)
            q_values_all.append(q_values.cpu())
            values = get_Vs(agent, *expert_batch_state, actions)
            rewards.append((q_max - hparams.Train.gamma * values).cpu())
            
            is_last.append(expert_batch['true_action'] == hparams.Data.patch_count)
            subj_ids.append(expert_batch['subj_id'])
            sp_lens.append(expert_batch['scanpath_length'])
            i += 1
            if i == 10:
                break

    q_values_all = torch.cat(q_values_all, dim=0)
    subj_ids = torch.cat(subj_ids, dim=0)
    sp_lens = torch.cat(sp_lens, dim=0)
    durations = torch.cat(durations, dim=0)
    is_last = torch.cat(is_last, dim=0)
    history_fixs = torch.cat(history_fixs, dim=0)
    rewards = torch.cat(rewards, dim=0)
    
    return q_values_all, rewards, subj_ids, sp_lens, is_last, history_fixs, durations

if __name__ == '__main__':
    args = parse_args()
    hparams = JsonConfig(args.hparams)
    assert os.path.exists(hparams.Model.checkpoint), "need trained fixation model!"
    
    hparams_tp = JsonConfig('./configs/coco_search18_TP.json')
    hparams_ta = JsonConfig('./configs/coco_search18_TA.json')
    log_dir = hparams.Train.log_dir
    
    dataset_root = args.dataset_root
    device = torch.device('cuda:0')

    agent_nonstop, train_gaze_loader, train_img_loader, valid_img_loader,\
        valid_img_loader_ta, env, env_valid, global_step, bbox_annos,\
        human_cdf, fix_clusters, prior_maps_tp, prior_maps_ta,\
        sss_strings, valid_gaze_loader_tp, valid_gaze_loader_ta = build(
            hparams, dataset_root, device, is_testing=True)

    q_values_all, rewards, subj_ids, sp_lens, is_last, history_fixs, durations = get_max_Q(
        agent_nonstop, train_gaze_loader)
    
    n_samples = q_values_all.size(0)
    X = torch.cat([
        q_values_all.reshape(n_samples, -1), 
        sp_lens.reshape(n_samples, -1),
    ], dim=1).numpy()
    ys = is_last.to(torch.float32).numpy()

    clf = LogisticRegression(random_state=0, class_weight='balanced', solver='liblinear').fit(X, ys)
    model_stop = TerminatorWrapper(agent_nonstop.q_net, clf, hparams.Train.init_temp)
    agent = SoftQ_wStop(model_stop, hparams.Train.batch_size, device, hparams.Train, has_stop=True)
    rst, sps_greedy = evaluate(env_valid,
                               agent,
                               valid_img_loader_ta,
                               valid_gaze_loader_ta,
                               hparams_ta.Data,
                               bbox_annos,
                               human_cdf,
                               fix_clusters,
                               prior_maps_ta,
                               sss_strings,
                               dataset_root,
                               sample_action=False,
                               sample_scheme='Greedy',
                               sample_stop=True,
                               return_scanpath=True)
    print(rst)
    
    # save results
    for sp in sps_greedy:
        sp['X'] = sp['X'].tolist()
        sp['Y'] = sp['Y'].tolist()
    
    with open(f'{log_dir}/greedy_results.json', 'w') as outfile:
        json.dump(rst, outfile, indent=4)

    with open(f'{log_dir}/greedy_predictions.json', 'w') as outfile:
        json.dump(sps_greedy, outfile, indent=4)
