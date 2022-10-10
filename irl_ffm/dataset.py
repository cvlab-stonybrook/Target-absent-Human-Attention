from os.path import join
from torchvision import transforms
import numpy as np
from pycocotools.coco import COCO
from .data import DCB_IRL, DCB_Human_Gaze
from .utils import compute_search_cdf, preprocess_fixations
from .utils import cutFixOnTarget, get_prior_maps
from .data import CFI_IRL, CFI_Human_Gaze
from .data import FFN_IRL, FFN_Human_Gaze


def process_data(target_trajs,
                 dataset_root,
                 target_annos,
                 hparams,
                 target_trajs_all,
                 is_testing=False,
                 sample_scanpath=False,
                 use_coco_annotation=False,
                 out_of_subject_eval=False):

    print("using", hparams.Train.repr)
    if use_coco_annotation:
        annFile = join(
            dataset_root, hparams.Data.coco_anno_dir, 'instances_train2017.json')
        coco_annos_train = COCO(annFile)
        annFile = join(
            dataset_root, hparams.Data.coco_anno_dir, 'instances_val2017.json')
        coco_annos_val = COCO(annFile)
        coco_annos = (coco_annos_train, coco_annos_val)
    else:
        coco_annos = None

    if hparams.Train.repr == 'DCB':
        DCB_HR_dir = join(dataset_root, 'DCBs/HR/')
        DCB_LR_dir = join(dataset_root, 'DCBs/LR/')
    elif hparams.Train.repr == 'CFI' or hparams.Train.repr == 'FFN':
        transform_train = transforms.Compose([
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        raise NotImplementedError

    eval_split = 'test' if is_testing else 'valid'
    print(f"evaluation split = {eval_split}")
    
    valid_target_trajs_all = list(
        filter(lambda x: x['split'] == eval_split, target_trajs_all))

    fix_clusters = np.load(f'{dataset_root}/clusters.npy',
                           allow_pickle=True).item()
    for k, v in fix_clusters.items():
        # remove other subjects' data if "subject" is specified
        if hparams.Data.subject > -1:
            try:
                v['strings'] = [v['strings'][hparams.Data.subject]]
            except:
                v['strings'] = []
        else:
            v['strings'] = list(v['strings'].values())
    scene_labels = np.load(f'{dataset_root}/scene_label_dict.npy',
                           allow_pickle=True).item()

    target_init_fixs = {}
    for traj in target_trajs_all:
        key = traj['task'] + '_' + traj['name']
        target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w,
                                 traj['Y'][0] / hparams.Data.im_h)
    if hparams.Train.zero_shot:
        catIds = np.load(join(dataset_root, 'all_target_ids.npy'),
                         allow_pickle=True).item()
    else:
        cat_names = list(np.unique([x['task'] for x in target_trajs]))
        catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    human_mean_cdf = None

    # training fixation data
    train_target_trajs = list(
        filter(lambda x: x['split'] == 'train', target_trajs))
    # print statistics
    traj_lens = list(map(lambda x: x['length'], train_target_trajs))
    avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
    print('average train scanpath length : {:.3f} (+/-{:.3f})'.format(
        avg_traj_len, std_traj_len))
    print('num of train trajs = {}'.format(len(train_target_trajs)))
    
    train_task_img_pair = np.unique([
        traj['task'] + '_' + traj['name'] + '_' + traj['condition']
        for traj in train_target_trajs
    ])
    train_fix_labels = preprocess_fixations(
        train_target_trajs,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        has_stop=hparams.Data.has_stop,
        sample_scanpath=sample_scanpath)

    # validation fixation data
    valid_target_trajs = list(
        filter(lambda x: x['split'] == eval_split, target_trajs))

    # print statistics
    traj_lens = list(map(lambda x: x['length'], valid_target_trajs))
    avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
    print('average valid scanpath length : {:.3f} (+/-{:.3f})'.format(
        avg_traj_len, std_traj_len))
    print('num of valid trajs = {}'.format(len(valid_target_trajs)))

    valid_task_img_pair = np.unique([
        traj['task'] + '_' + traj['name'] + '_' + traj['condition']
        for traj in valid_target_trajs
    ])

    tp_trajs = list(
        filter(
            lambda x: x['condition'] == 'present' and x[
                'split'] == eval_split,
            target_trajs_all))
    human_mean_cdf, _ = compute_search_cdf(tp_trajs, target_annos,
                                           hparams.Data.max_traj_length)
    print('target fixation prob (valid).:', human_mean_cdf)
    
    valid_fix_labels = preprocess_fixations(
        valid_target_trajs,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        has_stop=hparams.Data.has_stop,
        sample_scanpath=sample_scanpath)

    if hparams.Train.repr == 'DCB':
        # load image data
        train_img_dataset = DCB_IRL(DCB_HR_dir, DCB_LR_dir,
                                    target_init_fixs, train_task_img_pair,
                                    target_annos, hparams.Data, catIds)

        valid_task_img_pair_TP = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'present'
        ])

        valid_img_dataset_TP = DCB_IRL(DCB_HR_dir, DCB_LR_dir,
                                       target_init_fixs,
                                       valid_task_img_pair_TP,
                                       target_annos, hparams.Data, catIds)

        valid_task_img_pair_TA = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'absent'
        ])
        valid_img_dataset_TA = DCB_IRL(DCB_HR_dir, DCB_LR_dir,
                                       target_init_fixs,
                                       valid_task_img_pair_TA,
                                       target_annos, hparams.Data, catIds)
        
        # load human gaze data
        train_HG_dataset = DCB_Human_Gaze(DCB_HR_dir,
                                          DCB_LR_dir,
                                          train_fix_labels,
                                          train_task_img_pair,
                                          target_annos,
                                          hparams.Data,
                                          catIds,
                                          mix_match=hparams.Data.mix_match)
        valid_HG_dataset = DCB_Human_Gaze(DCB_HR_dir,
                                          DCB_LR_dir,
                                          valid_fix_labels,
                                          valid_task_img_pair,
                                          target_annos,
                                          hparams.Data,
                                          catIds,
                                          blur_action=hparams.Data.blur_action)
    elif hparams.Train.repr == 'CFI':
        # load image data
        train_img_dataset = CFI_IRL(dataset_root, target_init_fixs,
                                    train_task_img_pair, target_annos,
                                    transform_train, hparams.Data, catIds)
        valid_task_img_pair_TP = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'present'
        ])

        valid_img_dataset_TP = CFI_IRL(dataset_root, target_init_fixs,
                                       valid_task_img_pair_TP,
                                       target_annos, transform_train,
                                       hparams.Data, catIds)
        
        valid_task_img_pair_TA = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'absent'
        ])
        valid_img_dataset_TA = CFI_IRL(dataset_root, target_init_fixs,
                                       valid_task_img_pair_TA,
                                       target_annos, transform_train,
                                       hparams.Data, catIds)

        if hparams.Data.TAP == 'TAP':
            valid_img_info_TP = list(
                filter(lambda x: x.split('_')[-1] == 'present',
                       valid_task_img_pair))
            valid_img_info_TA = list(
                set(valid_task_img_pair) - set(valid_img_info_TP))
            valid_TP_imgset = CFI_IRL(dataset_root, target_init_fixs,
                                      valid_img_info_TP, target_annos,
                                      transform_test, hparams.Data, catIds)
            valid_TA_imgset = CFI_IRL(dataset_root, target_init_fixs,
                                      valid_img_info_TA, target_annos,
                                      transform_test, hparams.Data, catIds)
            valid_img_dataset = {
                "TP": valid_TP_imgset,
                "TA": valid_TA_imgset
            }
        else:
            valid_img_dataset = CFI_IRL(dataset_root, target_init_fixs,
                                        valid_task_img_pair, target_annos,
                                        transform_test, hparams.Data,
                                        catIds)
            
        # load human gaze data
        train_HG_dataset = CFI_Human_Gaze(dataset_root,
                                          train_fix_labels,
                                          target_annos,
                                          scene_labels,
                                          hparams.Data,
                                          transform_train,
                                          catIds,
                                          blur_action=False)
        valid_HG_dataset = CFI_Human_Gaze(dataset_root,
                                          valid_fix_labels,
                                          target_annos,
                                          scene_labels,
                                          hparams.Data,
                                          transform_test,
                                          catIds,
                                          blur_action=True)
    elif hparams.Train.repr == 'FFN':
        # load image data
        train_img_dataset = FFN_IRL(dataset_root,
                                    None,
                                    train_task_img_pair,
                                    target_annos,
                                    transform_train,
                                    hparams.Data,
                                    catIds,
                                    coco_annos=coco_annos)
        
        valid_task_img_pair_all = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
        ])
        valid_img_dataset_all = FFN_IRL(dataset_root,
                                        None,
                                        valid_task_img_pair_all,
                                        target_annos,
                                        transform_test,
                                        hparams.Data,
                                        catIds,
                                        coco_annos=coco_annos)

        valid_task_img_pair_TP = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'present'
        ])
        valid_img_dataset_TP = FFN_IRL(dataset_root,
                                       None,
                                       valid_task_img_pair_TP,
                                       target_annos,
                                       transform_test,
                                       hparams.Data,
                                       catIds,
                                       coco_annos=coco_annos)
        
        valid_task_img_pair_TA = np.unique([
            traj['task'] + '_' + traj['name'] + '_' + traj['condition']
            for traj in valid_target_trajs_all
            if traj['condition'] == 'absent'
        ])
        valid_img_dataset_TA = FFN_IRL(dataset_root, None,
                                       valid_task_img_pair_TA,
                                       target_annos, transform_test,
                                       hparams.Data, catIds, coco_annos)
        
        train_HG_dataset = FFN_Human_Gaze(dataset_root,
                                          train_fix_labels,
                                          target_annos,
                                          scene_labels,
                                          hparams.Data,
                                          transform_train,
                                          catIds,
                                          blur_action=True,
                                          coco_annos=coco_annos)
        valid_HG_dataset = FFN_Human_Gaze(dataset_root,
                                          valid_fix_labels,
                                          target_annos,
                                          scene_labels,
                                          hparams.Data,
                                          transform_test,
                                          catIds,
                                          blur_action=True,
                                          coco_annos=coco_annos)
        valid_target_trajs_TP = list(
            filter(lambda x: x['condition'] == 'present',
                   valid_target_trajs_all))
        valid_fix_labels_TP = preprocess_fixations(
            valid_target_trajs_TP,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath)
        valid_HG_dataset_TP = FFN_Human_Gaze(dataset_root,
                                             valid_fix_labels_TP,
                                             target_annos,
                                             scene_labels,
                                             hparams.Data,
                                             transform_test,
                                             catIds,
                                             blur_action=True,
                                             coco_annos=coco_annos)
        valid_target_trajs_TA = list(
            filter(lambda x: x['condition'] == 'absent',
                   valid_target_trajs_all))
        valid_fix_labels_TA = preprocess_fixations(
            valid_target_trajs_TA,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath)
        valid_HG_dataset_TA = FFN_Human_Gaze(dataset_root,
                                             valid_fix_labels_TA,
                                             target_annos,
                                             scene_labels,
                                             hparams.Data,
                                             transform_test,
                                             catIds,
                                             blur_action=True,
                                             coco_annos=coco_annos)
        valid_fix_labels_all = preprocess_fixations(
            valid_target_trajs_all,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            has_stop=hparams.Data.has_stop,
            sample_scanpath=sample_scanpath)
        valid_HG_dataset_all = FFN_Human_Gaze(dataset_root,
                                              valid_fix_labels_all,
                                              target_annos,
                                              scene_labels,
                                              hparams.Data,
                                              transform_test,
                                              catIds,
                                              blur_action=True,
                                              coco_annos=coco_annos)
        
    cutFixOnTarget(target_trajs, target_annos)

    return {
        'catIds': catIds,
        'img_train': train_img_dataset,
        'img_valid_TP': valid_img_dataset_TP,
        'img_valid_TA': valid_img_dataset_TA,
        'img_valid': valid_img_dataset_all,
        'gaze_train': train_HG_dataset,
        'gaze_valid': valid_HG_dataset_all,
        'gaze_valid_TP': valid_HG_dataset_TP,
        'gaze_valid_TA': valid_HG_dataset_TA,
        'bbox_annos': target_annos,
        'fix_clusters': fix_clusters,
        'valid_scanpaths': valid_target_trajs_all,
        'human_cdf': human_mean_cdf,
    }
