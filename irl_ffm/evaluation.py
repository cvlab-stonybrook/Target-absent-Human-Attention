from torch.distributions import Categorical
from . import utils, metrics
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F


def pack_model_inputs(obs_fov, env, has_stop=False):
    is_composite_state = isinstance(obs_fov, tuple)
    if is_composite_state:
        inputs = (*obs_fov, env.task_ids)
    else:
        inputs = (obs_fov, env.task_ids)
        
    if has_stop:
        batch_size = env.task_ids.size(0)
        n_fixs = env.step_id + 1
        normed_n_fixs = torch.ones(batch_size, 1) * n_fixs
        inputs = (*inputs, normed_n_fixs)
    
    return inputs

def collect_trajs(env,
                  policy,
                  patch_num,
                  max_traj_length,
                  is_eval=False,
                  sample_action=True,
                  is_zero_shot=False,
                  sample_scheme='IG_MAX',
                  sample_stop=False):

    rewards = []
    obs_fov = env.observe()

    states = pack_model_inputs(obs_fov, env, sample_stop)
    act, stop = policy.select_action(states,
                                     sample_action,
                                     action_mask=env.action_mask)
    status = [env.status]

    i = 0

    actions = []
    while i < max_traj_length:
        new_obs_fov, curr_status = env.step(act)
        if sample_stop:
            curr_status = stop
        status.append(curr_status)
        actions.append(act)
        obs_fov = new_obs_fov
        states = pack_model_inputs(obs_fov, env, sample_stop)
        act, stop = policy.select_action(states,
                                         sample_action,
                                         action_mask=env.action_mask)
        i = i + 1

    status = torch.stack(status[1:])
    actions = torch.stack(actions)
    bs = len(env.img_names)
    trajs = []
    for i in range(bs):
        ind = (status[:, i] == 1).to(torch.int8).argmax().item() + 1
        if status[:, i].sum() == 0:
            ind = status.size(0)
        trajs.append({'actions': actions[:ind, i]})
    return trajs


def sample_scanpaths(env,
                     model,
                     dataloader,
                     pa,
                     sample_action=True,
                     sample_scheme='CLS_MAX',
                     sample_stop=False):
    env.max_step = pa.max_traj_length + 1

    # generating scanpaths
    all_actions = []
    print('Generating scanapths...')
    for subj_id in range(10):
        for batch in tqdm(dataloader):
            env.set_data(batch)
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            cond_batch = batch['condition']
            with torch.no_grad():
                env.reset()
                trajs = collect_trajs(env,
                                      model,
                                      pa.patch_num,
                                      pa.max_traj_length,
                                      is_eval=True,
                                      sample_action=sample_action,
                                      sample_scheme=sample_scheme,
                                      sample_stop=sample_stop)
                all_actions.extend([(cat_names_batch[i], img_names_batch[i],
                                     cond_batch[i], trajs[i]['actions'])
                                    for i in range(env.batch_size)])
        if not sample_action:
            break
    print('Computing metrics...')
    scanpaths = utils.actions2scanpaths(all_actions, pa.patch_num)

    return scanpaths


def compute_info_gain(predicted_probs, gt_fixs, base_probs, eps=2.2204e-16):
    fired_probs = predicted_probs[torch.arange(gt_fixs.size(0)
                                               ), gt_fixs[:, 1], gt_fixs[:, 0]]
    fired_base_probs = base_probs[torch.arange(gt_fixs.size(0)
                                               ), gt_fixs[:, 1], gt_fixs[:, 0]]
    IG = torch.sum(
        torch.log2(fired_probs + eps) - torch.log2(fired_base_probs + eps))
    return IG


def compute_NSS(saliency_map, gt_fixs):
    mean = saliency_map.view(gt_fixs.size(0), -1).mean(dim=1)
    std = saliency_map.view(gt_fixs.size(0), -1).std(dim=1)
    std[std == 0] = 1  # avoid division by 0

    value = saliency_map[torch.arange(gt_fixs.size(0)
                                      ), gt_fixs[:, 1], gt_fixs[:, 0]]
    value -= mean
    value /= std

    return value.sum()


def compute_conditional_saliency_metrics(pa, agent, dataloader,
                                         task_dep_prior_maps):
    with torch.no_grad():
        n_samples, info_gain, nss = 0, 0, 0

        for expert_batch in tqdm(dataloader):
            expert_batch_state = (expert_batch['true_state'].to(agent.device),
                                  expert_batch['normalized_fixations'].to(
                                      agent.device),
                                  expert_batch['task_id'].to(agent.device))

            gt_next_fixs = (
                expert_batch['next_normalized_fixations'][:, -1] *
                torch.tensor([pa.im_w, pa.im_h])).to(
                    torch.long)

            q = agent.q_net(*expert_batch_state)
            bs = q.size(0)
            q = F.interpolate(q.view(bs, 1, 20, 32),
                              size=(pa.im_h, pa.im_w),
                              mode='bilinear').view(bs, -1)
            probs = F.softmax(q / agent.alpha,
                              dim=1).view(bs, pa.im_h, pa.im_w)

            prior_maps = torch.stack([
                task_dep_prior_maps[f'{task}_{cond}'] for task, cond in zip(
                    expert_batch['task_name'], expert_batch['condition'])
            ])
            info_gain += compute_info_gain(probs, gt_next_fixs, prior_maps)
            nss += compute_NSS(probs, gt_next_fixs)
            n_samples += bs

        info_gain /= n_samples
        nss /= n_samples

    return info_gain.item(), nss.item()


def evaluate(env,
             model,
             dataloader,
             gazeloader,
             pa,
             bbox_annos,
             human_cdf,
             fix_clusters,
             task_dep_prior_maps,
             semSS_strings,
             dataset_root,
             sample_action=True,
             sample_scheme='CLS_MAX',
             sample_stop=False,
             return_scanpath=False):
    model.eval()
    TAP = pa.TAP
    scanpaths = sample_scanpaths(env, model, dataloader, pa, sample_action,
                                 sample_scheme, sample_stop)

    metrics_dict = {}
    if TAP == 'TP':
        if not sample_stop:
            utils.cutFixOnTarget(scanpaths, bbox_annos)
        # search effiency
        mean_cdf, _ = utils.compute_search_cdf(scanpaths, bbox_annos,
                                               pa.max_traj_length)
        metrics_dict.update(
            dict(
                zip([f"TFP_top{i}" for i in range(1, len(mean_cdf))],
                    mean_cdf[1:])))

        # probability mismatch
        metrics_dict['prob_mismatch'] = np.sum(
            np.abs(human_cdf[:len(mean_cdf)] - mean_cdf))

    # sequence score
    ss_2steps = metrics.get_seq_score(scanpaths, fix_clusters, 2, True)
    ss_4steps = metrics.get_seq_score(scanpaths, fix_clusters, 4, True)
    ss_6steps = metrics.get_seq_score(scanpaths, fix_clusters, 6, True)
    ss = metrics.get_seq_score(scanpaths, fix_clusters, pa.max_traj_length,
                               False)

    sss_2steps = metrics.get_semantic_seq_score(
        scanpaths, semSS_strings, 2,
        f'{dataset_root}/semantic_seq/segmentation_maps', True)
    sss_4steps = metrics.get_semantic_seq_score(
        scanpaths, semSS_strings, 4,
        f'{dataset_root}/semantic_seq/segmentation_maps', True)
    sss_6steps = metrics.get_semantic_seq_score(
        scanpaths, semSS_strings, 6,
        f'{dataset_root}/semantic_seq/segmentation_maps', True)
    sss = metrics.get_semantic_seq_score(
        scanpaths, semSS_strings, pa.max_traj_length,
        f'{dataset_root}/semantic_seq/segmentation_maps', False)

    metrics_dict.update({
        f"{TAP}_seq_score_max": ss,
        f"{TAP}_seq_score_2steps": ss_2steps,
        f"{TAP}_seq_score_4steps": ss_4steps,
        f"{TAP}_seq_score_6steps": ss_6steps,
        f"{TAP}_semantic_seq_score_max": sss,
        f"{TAP}_semantic_seq_score_2steps": sss_2steps,
        f"{TAP}_semantic_seq_score_4steps": sss_4steps,
        f"{TAP}_semantic_seq_score_6steps": sss_6steps,
    })

    # temporal spatial saliency metrics
    if not sample_action:
        ig, nss = compute_conditional_saliency_metrics(pa, model, gazeloader,
                                                       task_dep_prior_maps)
        metrics_dict.update({
            f"{TAP}_cIG": ig,
            f"{TAP}_cNSS": nss,
        })

    if sample_stop:
        sp_len_diff = []
        for traj in scanpaths:
            gt_trajs = list(
                filter(
                    lambda x: x['task'] == traj['task'] and x['name'] == traj[
                        'name'], human_scanpath_test))
            sp_len_diff.append(
                len(traj['X']) -
                np.array([len(traj['X']) for traj in gt_trajs]))
        sp_len_diff = np.abs(np.concatenate(sp_len_diff))
        metrics_dict[f'{TAP}_sp_len_err_mean'] = sp_len_diff.mean()
        metrics_dict[f'{TAP}_sp_len_err_std'] = sp_len_diff.std()
        metrics_dict[f'{TAP}_avg_sp_len'] = np.mean(
            [len(x['X']) for x in scanpaths])

    if not sample_action:
        prefix = sample_scheme + '_'
        keys = list(metrics_dict.keys())
        for k in keys:
            metrics_dict[prefix + k] = metrics_dict.pop(k)

    model.train()

    if return_scanpath:
        return metrics_dict, scanpaths
    else:
        return metrics_dict
