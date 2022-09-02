from torch.distributions import Categorical
from . import utils, metrics
from tqdm import tqdm
import torch
import numpy as np


def pack_model_inputs(obs_fov, env):
    is_composite_state = isinstance(obs_fov, tuple)
    if is_composite_state:
        inputs = (*obs_fov, env.task_ids)
    else:
        inputs = (obs_fov, env.task_ids)
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

    states = pack_model_inputs(obs_fov, env)
    act, stop = policy.select_action(states,
                                     sample_action,
                                     action_mask=env.action_mask,
                                     sample_stop=sample_stop)
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
        states = pack_model_inputs(obs_fov, env)
        act, stop = policy.select_action(states,
                                         sample_action,
                                         action_mask=env.action_mask,
                                         sample_stop=sample_stop)
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


def evaluate(env,
             model,
             dataloader,
             pa,
             bbox_annos,
             human_cdf,
             fix_clusters,
             sample_action=True,
             sample_scheme='CLS_MAX',
             sample_stop=False):
    model.eval()
    env.max_step = pa.max_traj_length + 1

    # generating scanpaths
    all_actions = []
    TAP = pa.TAP
    print('Generating scanapths...')
    n_runs = 10 if sample_action else 1
    for _ in tqdm(range(n_runs)):
        for batch in dataloader:
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
    print('Computing metrics...')
    scanpaths = utils.actions2scanpaths(all_actions, pa.patch_num)

    metrics_dict = {}
    if TAP == 'TP':
        if not sample_stop:
            utils.cutFixOnTarget(scanpaths, bbox_annos)
        # search effiency
        mean_cdf, _ = utils.compute_search_cdf(scanpaths,
                                               bbox_annos,
                                               pa.max_traj_length)
        metrics_dict.update(
            dict(
                zip([f"TFP_top{i}" for i in range(1, len(mean_cdf))],
                    mean_cdf[1:])))

        # probability mismatch
        metrics_dict['prob_mismatch'] = np.sum(
            np.abs(human_cdf[:len(mean_cdf)] - mean_cdf))

    # sequence score
    ss_4steps = metrics.get_seq_score(scanpaths, fix_clusters, 4,
                                      True)
    ss = metrics.get_seq_score(scanpaths, fix_clusters,
                               pa.max_traj_length, False)

    metrics_dict.update({
        f"{TAP}_seq_score_max": ss,
        f"{TAP}_seq_score_4steps": ss_4steps
    })

    # temporal spatial saliency metrics
    if sample_action:
        ig, cc, nss = metrics.compute_spatial_metrics_by_step(
            scanpaths, human_scanpaths, pa.im_w, pa.im_h, task_dep_prior_maps)
        metrics_dict.update({'IG_all': ig, 'CC_all': cc, 'NSS_all': nss})
        for i in [2, 4, 6]:
            ig, cc, nss = metrics.compute_spatial_metrics_by_step(
                scanpaths, human_scanpaths, pa.im_w, pa.im_h,
                task_dep_prior_maps, i)
            metrics_dict.update({
                f'IG_{i}': ig,
                f'CC_{i}': cc,
                f'NSS_{i}': nss
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

    return metrics_dict
