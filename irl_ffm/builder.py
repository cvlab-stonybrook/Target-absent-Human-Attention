from .environment import FFN_Env
from .dataset import process_data
from .models import FFNGeneratorCond_V2
from .sql import SoftQ
from .iql import irl_update, irl_update_critic
from .config import JsonConfig
from .utils import get_prior_maps
import json
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
import types


def build(hparams, dataset_root, device, is_testing=False):
    # dir of pre-computed beliefs
    data_name = '{}x{}'.format(hparams.Data.im_w, hparams.Data.im_h)

    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()

    # load ground-truth human scanpaths
    fixation_path = join(
        dataset_root, 'coco_search_fixations_512x320_on_target_allvalid.json')
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)

    # exclude incorrect scanpaths
    if hparams.Data.exclude_wrong_trials:
        human_scanpaths = list(
            filter(lambda x: x['correct'] == 1, human_scanpaths))
    human_scanpaths = list(
        filter(lambda x: x['fixOnTarget'] or x['condition'] == 'absent',
               human_scanpaths))
    human_scanpaths_all = human_scanpaths

    # choose data to use: TP = target-present trials, TA = target-absent trials
    # else = TP + TA trials
    human_scanpaths_ta = list(
        filter(lambda x: x['condition'] == 'absent', human_scanpaths_all))
    human_scanpaths_tp = list(
        filter(lambda x: x['condition'] == 'present', human_scanpaths_all))

    if hparams.Data.TAP == 'TP':
        human_scanpaths = list(
            filter(lambda x: x['condition'] == 'present', human_scanpaths))
        human_scanpaths = list(
            filter(lambda x: x['fixOnTarget'], human_scanpaths))
        utils.cutFixOnTarget(human_scanpaths, bbox_annos)
    elif hparams.Data.TAP == 'TA':
        human_scanpaths = list(
            filter(lambda x: x['condition'] == 'absent', human_scanpaths))

    if hparams.Data.subject > -1:
        print(f"excluding subject {hparams.Data.subject} data!")
        human_scanpaths = list(
            filter(lambda x: x['subject'] != hparams.Data.subject,
                   human_scanpaths))

    # process fixation data
    dataset = process_data(human_scanpaths,
                           dataset_root,
                           bbox_annos,
                           hparams,
                           human_scanpaths_all,
                           is_testing=is_testing,
                           sample_scanpath=False,
                           use_coco_annotation=not is_testing)

    batch_size = hparams.Train.batch_size
    n_workers = hparams.Train.n_workers

    train_HG_loader = DataLoader(dataset['gaze_train'],
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=n_workers,
                                 drop_last=True,
                                 pin_memory=True)
    print('num of training batches =', len(train_HG_loader))

    train_img_loader = DataLoader(dataset['img_train'],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  drop_last=True,
                                  pin_memory=True)
    
    valid_img_loader = DataLoader(dataset['img_valid_TP'],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=n_workers,
                                  drop_last=False,
                                  pin_memory=True)
    valid_img_loader_TA = DataLoader(dataset['img_valid_TA'],
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=n_workers,
                                     drop_last=False,
                                     pin_memory=True)
    valid_HG_loader_TP = DataLoader(dataset['gaze_valid_TP'],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=n_workers,
                                    drop_last=False,
                                    pin_memory=True)
    valid_HG_loader_TA = DataLoader(dataset['gaze_valid_TA'],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=n_workers,
                                    drop_last=False,
                                    pin_memory=True)

    ffn_dim = hparams.Model.foveal_feature_dim
    hidden_dim = hparams.Model.gen_hidden_size
    q_net = FFNGeneratorCond_V2(ffn_dim,
                                hidden_size=hidden_dim,
                                is_cumulative=True).to(device)
    try:
        ckp = torch.load(hparams.Model.checkpoint)
        q_net.load_state_dict(ckp['model'])
        print(f"loaded weights from {hparams.Model.checkpoint}.")
    except:
        if len(hparams.Model.checkpoint) > 0:
            print(f"failed to read checkpoint from {hparams.Model.checkpoint}.")
        ckp = None
    q_net = torch.nn.DataParallel(q_net)

    agent = SoftQ(q_net, batch_size, device, hparams.Train)
    agent.irl_update = types.MethodType(irl_update, agent)
    agent.ilr_update_critic = types.MethodType(irl_update_critic, agent)

    if ckp:
        global_step = ckp['step']
        agent.critic_optimizer.load_state_dict(ckp['optimizer'])
    else:
        global_step = 0

    env = FFN_Env(hparams.Data,
                  max_step=hparams.Data.max_traj_length,
                  mask_size=hparams.Data.IOR_size,
                  status_update_mtd=hparams.Train.stop_criteria,
                  device=device,
                  ret_inhibition=True)

    env_valid = FFN_Env(hparams.Data,
                        max_step=hparams.Data.max_traj_length,
                        mask_size=hparams.Data.IOR_size,
                        status_update_mtd=hparams.Train.stop_criteria,
                        device=device,
                        ret_inhibition=True)
    bbox_annos = dataset['bbox_annos']
    human_cdf = dataset['human_cdf']
    fix_clusters = dataset['fix_clusters']
    prior_maps_ta = get_prior_maps(human_scanpaths_ta, hparams.Data.im_w,
                                   hparams.Data.im_h)
    keys = list(prior_maps_ta.keys())
    for k in keys:
        prior_maps_ta[f'{k}_absent'] = torch.tensor(
            prior_maps_ta.pop(k)).to(device)

    prior_maps_tp = get_prior_maps(human_scanpaths_tp, hparams.Data.im_w,
                                   hparams.Data.im_h)
    keys = list(prior_maps_tp.keys())
    for k in keys:
        prior_maps_tp[f'{k}_present'] = torch.tensor(
            prior_maps_tp.pop(k)).to(device)

    sss_strings = np.load(join(dataset_root, 'semantic_seq/test.pkl'),
                          allow_pickle=True)
    return (agent, train_HG_loader, train_img_loader, valid_img_loader,
            valid_img_loader_TA, env, env_valid, global_step, bbox_annos,
            human_cdf, fix_clusters, prior_maps_tp, prior_maps_ta, sss_strings,
            valid_HG_loader_TP, valid_HG_loader_TA)
