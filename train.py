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
    parser.add_argument('--eval-only',
                        action='store_true',
                        help='perform evaluation only')
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


def log_dict(writer, scalars, step, prefix):
    for k, v in scalars.items():
        writer.add_scalar(prefix + "/" + k, v, step)


if __name__ == '__main__':
    args = parse_args()
    hparams = JsonConfig(args.hparams)
    hparams_tp = JsonConfig('./configs/coco_search18_TP.json')
    hparams_ta = JsonConfig('./configs/coco_search18_TA.json')

    dataset_root = args.dataset_root
    device = torch.device('cuda:0')

    agent, train_gaze_loader, train_img_loader, valid_img_loader,\
        valid_img_loader_ta, env, env_valid, global_step, bbox_annos,\
        human_cdf, fix_clusters, prior_maps_tp, prior_maps_ta,\
        sss_strings, valid_gaze_loader_tp, valid_gaze_loader_ta = build(
            hparams, dataset_root, device, is_testing=args.eval_only)
    
    if args.eval_only:  # testing
        if hparams.Data.TAP != 'TA':
            rst_tp = evaluate(
                env_valid,
                agent,
                valid_img_loader,
                valid_gaze_loader_tp,
                hparams_tp.Data,
                bbox_annos,
                human_cdf,
                fix_clusters,
                prior_maps_tp,
                sss_strings,
                dataset_root,
                sample_action=False,
                sample_scheme='Greedy',
            )
            print("TP:", rst_tp)

        rst = evaluate(
            env_valid,
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
        )
        print(rst)
            
    else:
        log_dir = hparams.Train.log_dir
        writer = SummaryWriter(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print("Log dir:", log_dir)

        # Write configuration file to the log dir
        hparams.dump(log_dir, 'config.json')
        
        print_every = 20
        max_iters = hparams.Train.max_iters
        save_every = hparams.Train.checkpoint_every
        eval_every = hparams.Train.evaluate_every

        replay_memory = hparams.Train.replay_memory
        initial_memory = hparams.Train.initial_memory
        online_memory_replay = Memory(replay_memory, SEED)

        expert_demos_iter = iter(train_gaze_loader)
        s_epoch = int(global_step / len(train_img_loader))
        for i_epoch in range(s_epoch, int(1e5)):
            for i_batch, batch in enumerate(train_img_loader):
                # run policy to collect trajactories for a batch of images
                env.set_data(batch)
                obs_fov = env.observe()
                states = pack_model_inputs(obs_fov, env)
                act, stop = agent.select_action(
                    states,
                    True,
                    action_mask=env.action_mask,
                    sample_stop=env.pa.has_stop,
                )
                i_step = 0
                while i_step < hparams.Data.max_traj_length and env.status.min() < 1:
                    i_step += 1
                    new_obs_fov, curr_status = env.step(act)
                    
                    new_states = pack_model_inputs(new_obs_fov, env)
                    online_memory_replay.add_batch(
                        (states, new_states, act, curr_status, batch['centermaps']))
                    states = new_states
                    act, stop = agent.select_action(
                        states,
                        True,
                        action_mask=env.action_mask,
                        sample_stop=env.pa.has_stop,
                    )
                    
                    if online_memory_replay.size() <= initial_memory:
                        continue
                    
                    # Start learning
                    policy_batch, expert_batch, expert_demos_iter = get_batch(
                        online_memory_replay,
                        expert_demos_iter,
                        train_gaze_loader,
                        hparams.Train.batch_size,
                    )
                    losses = agent.irl_update(
                        policy_batch,
                        expert_batch,
                        global_step,
                        hparams.Data.patch_num,
                    )
                
                    if global_step % print_every == print_every - 1 and losses is not None:
                        date = str(datetime.datetime.now())
                        print(
                            "[{}], iter: {}, progress: {:.3f}, epoch: {}, total loss: {:.3f}, value loss: {:.3f}, softq loss: {:.3f}, detection loss: {:.3f}"
                            .format(
                                date[date.rfind(' ')+1:date.rfind('.')],
                                global_step,
                                (global_step / max_iters) * 100,
                                i_epoch,
                                losses['total_loss'],
                                losses['value_loss'],
                                losses['softq_loss'],
                                losses['detection_loss'],
                            ))
                        log_dict(writer, losses, global_step, 'train')
                
                    # Evaluate
                    if global_step % eval_every == eval_every - 1:
                        if hparams.Data.TAP != 'TA':
                            rst = evaluate(
                                env_valid,
                                agent,
                                valid_img_loader,
                                valid_gaze_loader_tp,
                                hparams_tp.Data,
                                bbox_annos,
                                human_cdf,
                                fix_clusters,
                                prior_maps_tp,
                                sss_strings,
                                dataset_root,
                                sample_action=False,
                                sample_scheme='Greedy',
                            )
                            log_dict(writer, rst, global_step, "eval_TP")
                            print("TP:", rst)
                        rst_ta = evaluate(
                            env_valid,
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
                        )
        
                        print("TA:", rst_ta)
                        log_dict(writer, rst_ta, global_step, "eval_TA")
                        writer.add_scalar('epoch', i_epoch, global_step)

                    if global_step % save_every == save_every - 1:
                        save_path = os.path.join(log_dir, f"ckp_{global_step}.pt")
                        torch.save(
                            {
                                'model': agent.q_net.state_dict(),
                                'optimizer': agent.critic_optimizer.state_dict(),
                                'step': global_step + 1,
                            },
                            save_path,
                        )
                        print(f"Saved checkpoint to {save_path}.")
                    global_step += 1
                if global_step >= max_iters:
                    print("Exit training!")
            else:
                continue
            break
