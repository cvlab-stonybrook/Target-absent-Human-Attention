import torch
from . import utils
from .coco_det import COCODetHelper 
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.ndimage as filters
from os.path import join
import warnings
import torch.utils.data
import torchvision
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DCB_IRL(Dataset):
    """
    Image data for training generator
    """

    def __init__(self, DCB_HR_dir, DCB_LR_dir, initial_fix, img_info, annos,
                 pa, catIds):
        self.img_info = img_info
        self.annos = annos
        self.pa = pa
        self.initial_fix = initial_fix
        self.catIds = catIds
        self.LR_dir = DCB_LR_dir
        self.HR_dir = DCB_HR_dir

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        cat_name, img_name, condition = self.img_info[idx].split('_')
        feat_name = img_name[:-3] + 'pth.tar'
        lr_path = join(self.LR_dir, cat_name.replace(' ', '_'), feat_name)
        hr_path = join(self.HR_dir, cat_name.replace(' ', '_'), feat_name)
        lr = torch.load(lr_path)
        hr = torch.load(hr_path)
        imgId = cat_name + '_' + img_name

        # update state with initial fixation
        init_fix = self.initial_fix[imgId]
        px, py = init_fix
        px, py = px * lr.size(-1), py * lr.size(-2)
        mask = utils.foveal2mask(px, py, self.pa.fovea_radius, hr.size(-2),
                                 hr.size(-1))
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).repeat(hr.size(0), 1, 1)
        lr = (1 - mask) * lr + mask * hr

        # history fixation map
        history_map = torch.zeros((hr.size(-2), hr.size(-1)))
        history_map = (1 - mask[0]) * history_map + mask[0] * 1

        # action mask
        action_mask = torch.zeros((self.pa.patch_num[1], self.pa.patch_num[0]),
                                  dtype=torch.uint8)
        px, py = init_fix
        px, py = int(px * self.pa.patch_num[0]), int(py * self.pa.patch_num[1])
        action_mask[py - self.pa.IOR_size:py + self.pa.IOR_size + 1, px -
                    self.pa.IOR_size:px + self.pa.IOR_size + 1] = 1

        # target location label
        if condition == 'present':
            coding = utils.multi_hot_coding(self.annos[imgId],
                                            self.pa.patch_size,
                                            self.pa.patch_num)
            coding = torch.from_numpy(coding).view(1, -1)
        else:
            coding = torch.zeros(1, self.pa.patch_count)

        return {
            'task_id': self.catIds[cat_name],
            'img_name': img_name,
            'cat_name': cat_name,
            'lr_feats': lr,
            'hr_feats': hr,
            'history_map': history_map,
            'init_fix': torch.FloatTensor(init_fix),
            'label_coding': coding,
            'action_mask': action_mask,
            'condition': condition,
            'hr_feats': hr
        }


class DCB_Human_Gaze(Dataset):
    """
    Human gaze data for training discriminator
    """

    def __init__(self,
                 DCB_HR_dir,
                 DCB_LR_dir,
                 fix_labels,
                 fix_info,
                 annos,
                 pa,
                 catIds,
                 blur_action=False,
                 mix_match=False):
        self.pa = pa
        self.fix_labels = fix_labels
        self.annos = annos
        self.catIds = catIds
        self.LR_dir = DCB_LR_dir
        self.HR_dir = DCB_HR_dir
        self.blur_action = blur_action
        self.mix_match = mix_match  # generate fake state-action pairs if true
        if mix_match:
            self.imglist_by_cat = {}
            for info in fix_info:
                cat, img, _ = info.split('_')
                if cat in self.imglist_by_cat.keys():
                    self.imglist_by_cat[cat].append(img)
                else:
                    self.imglist_by_cat[cat] = [img]

    def __len__(self):
        return len(self.fix_labels)

    def __getitem__(self, idx):
        # load low- and high-res beliefs
        true_or_fake = 1.0
        img_name, cat_name, condition, fixs, action = self.fix_labels[idx][:5]
        if self.mix_match and np.random.uniform(0) < 0.33:
            # randomly select a different image from the same category
            tmp = self.imglist_by_cat[cat_name][:]
            tmp.remove(img_name)
            img_name = np.random.choice(tmp)
            true_or_fake = 0.0

        feat_name = img_name[:-3] + 'pth.tar'
        lr_path = join(self.LR_dir, cat_name.replace(' ', '_'), feat_name)
        hr_path = join(self.HR_dir, cat_name.replace(' ', '_'), feat_name)
        state = torch.load(lr_path)
        hr = torch.load(hr_path)

        # construct DCB
        remap_ratio = self.pa.im_w / float(hr.size(-1))
        history_map = torch.zeros((hr.size(-2), hr.size(-1)))
        for i in range(len(fixs)):
            px, py = fixs[i]
            px, py = px / remap_ratio, py / remap_ratio
            mask = utils.foveal2mask(px, py, self.pa.fovea_radius, hr.size(-2),
                                     hr.size(-1))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0).repeat(hr.size(0), 1, 1)
            state = (1 - mask) * state + mask * hr
            history_map = (1 - mask[0]) * history_map + mask[0] * 1

        # create labels
        imgId = cat_name + '_' + img_name

        if condition == 'present':
            coding = utils.multi_hot_coding(self.annos[imgId],
                                            self.pa.patch_size,
                                            self.pa.patch_num)
            coding = torch.from_numpy(coding).view(1, -1)
        else:
            coding = torch.zeros(1, self.pa.patch_count)

        ret = {
            "task_id": self.catIds[cat_name],
            "true_state": state,
            "true_action": torch.tensor([action], dtype=torch.long),
            'label_coding': coding,
            'history_map': history_map,
            'img_name': img_name,
            'task_name': cat_name,
            'true_or_fake': true_or_fake,
            'hr_feats': hr
        }

        # blur action maps for evaluation
        if self.blur_action:
            action_map = np.zeros(self.pa.patch_count, dtype=np.float32)
            if action < self.pa.patch_count:
                action_map[action] = 1
                action_map = action_map.reshape(self.pa.patch_num[1], -1)
                action_map = filters.gaussian_filter(action_map, sigma=1)
            ret['action_map'] = action_map
        return ret


class CFI_IRL(Dataset):
    def __init__(self, root_dir, initial_fix, img_info, annos, transform, pa,
                 catIds):
        self.img_info = img_info
        self.root_dir = root_dir
        self.img_dir = root_dir + 'images/320x512/'
        self.transform = transform
        self.pa = pa
        self.bboxes = annos
        self.initial_fix = initial_fix
        self.catIds = catIds
        if self.pa.use_DCB_target:
            self.coco_thing_classes = np.load(join(root_dir,
                                                   'coco_thing_classes.npy'),
                                              allow_pickle=True).item()

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        imgId = self.img_info[idx]
        cat_name, img_name, condition = imgId.split('_')
        if imgId in self.bboxes.keys():
            bbox = self.bboxes[imgId]
        else:
            bbox = None

        pyr_path = "{}pyramids/{}/{}.npy".format(self.root_dir, cat_name,
                                                 img_name)
        im_pyr = np.load(pyr_path)

        pyr_tensor = list(map(lambda x: self.transform(x), im_pyr))
        if bbox is not None:
            coding = utils.multi_hot_coding(bbox, self.pa.patch_size,
                                            self.pa.patch_num)
            coding = torch.from_numpy(coding).view(1, -1)
        else:
            coding = torch.zeros(1, self.pa.patch_count)

        # create action mask
        action_mask = np.zeros(self.pa.patch_num[0] * self.pa.patch_num[1],
                               dtype=np.uint8)

        ret = {
            'task_id': self.catIds[cat_name],
            'img_name': img_name,
            'cat_name': cat_name,
            'pyramid': torch.stack(pyr_tensor),
            'label_coding': coding,
            'condition': condition,
            'action_mask': torch.from_numpy(action_mask),
            'gt_length': self.img_info[idx][4]
        }

        if self.pa.use_DCB_target:
            DCBs = torch.load(
                join(self.pa.DCB_dir, cat_name.replace(' ', '_'),
                     img_name[:-3] + 'pth.tar'))
            ret['DCB_target_map'] = DCBs[self.coco_thing_classes[cat_name]]

        return ret


class CFI_Human_Gaze(Dataset):
    """
    Human gaze data in state-action pairs
    """

    def __init__(self,
                 root_dir,
                 fix_labels,
                 bbox_annos,
                 scene_annos,
                 pa,
                 transform,
                 catIds,
                 blur_action=False,
                 acc_foveal=True):
        self.root_dir = root_dir
        self.img_dir = root_dir + 'images/320x512/'
        self.pa = pa
        self.transform = transform
        self.fix_labels = fix_labels
        self.catIds = catIds
        self.blur_action = blur_action
        self.acc_foveal = acc_foveal
        self.bboxes = bbox_annos
        self.scene_labels = scene_annos['labels']
        self.scene_to_id = scene_annos['id_list']

        if self.pa.use_DCB_target:
            self.coco_thing_classes = np.load(join(root_dir,
                                                   'coco_thing_classes.npy'),
                                              allow_pickle=True).item()

    def __len__(self):
        return len(self.fix_labels)

    def __getitem__(self, idx):
        img_name, cat_name, condition, fixs, action, is_last, sid = self.fix_labels[
            idx]
        imgId = cat_name + '_' + img_name
        if imgId in self.bboxes.keys():
            bbox = self.bboxes[imgId]
        else:
            bbox = None
        scene_label = self.scene_labels[cat_name.replace(' ', '_') + '_' +
                                        condition + '_' + img_name]
        scene_id = self.scene_to_id[scene_label]

        pyr_path = "{}pyramids/{}/{}.npy".format(self.root_dir, cat_name,
                                                 img_name)
        im_pyr = np.load(pyr_path)
        fov_img = utils.foveat_img(None, fixs, im_pyr)
        fov_im_tensor = self.transform(Image.fromarray(fov_img))

        if bbox is not None:
            coding = utils.multi_hot_coding(bbox, self.pa.patch_size,
                                            self.pa.patch_num)
            coding = torch.from_numpy(coding).view(1, -1)
        else:
            coding = torch.zeros(1, self.pa.patch_count)

        # fixation history
        fixs_tensor = torch.tensor(fixs)
        ind_x = (fixs_tensor[:, 0] / self.pa.patch_size[0]).to(torch.long)
        ind_y = (fixs_tensor[:, 1] / self.pa.patch_size[1]).to(torch.long)
        history = torch.zeros(self.pa.patch_num[1], self.pa.patch_num[0])
        history[ind_y, ind_x] = 1

        ret = {
            "task_id": self.catIds[cat_name],
            "true_state": fov_im_tensor,
            "true_action": torch.tensor([action], dtype=torch.long),
            'label_coding': coding,
            'img_name': img_name,
            'task_name': cat_name,
            'fix_history': history,
            'is_TP': condition == 'present',
            'is_last': is_last,
            'scene_id': scene_id,
            'true_or_fake': 1.0,
            'subj_id': sid - 1  # sid ranges from [1, 10]
        }

        # precomputed panoptic-FPN target map
        if self.pa.use_DCB_target:
            DCBs = torch.load(
                join(self.pa.DCB_dir, cat_name.replace(' ', '_'),
                     img_name[:-3] + 'pth.tar'))
            ret['DCB_target_map'] = DCBs[self.coco_thing_classes[cat_name]]
            ret['DCBs'] = DCBs

        # compute the map of last fixation
        if self.pa.use_action_map:
            action_map = np.zeros((self.pa.patch_num[1], self.pa.patch_num[0]),
                                  dtype=np.float32)
            action_map[ind_y[-1].item(), ind_x[-1].item()] = 1
            action_map = filters.gaussian_filter(action_map, sigma=1)
            ret['last_fixation_map'] = action_map

        if self.blur_action:
            action_map = np.zeros(self.pa.patch_count, dtype=np.float32)
            if action < self.pa.patch_count:
                action_map[action] = 1
                action_map = action_map.reshape(self.pa.patch_num[1], -1)
                action_map = filters.gaussian_filter(action_map, sigma=1)
            else:
                action_map = action_map.reshape(self.pa.patch_num[1], -1)
            ret['action_map'] = action_map
        return ret


def get_coco_annos_by_img_name(coco_annos, img_name):
    img_id = int(img_name[:-4])
    # List: get annotation id from coco
    coco_annos_train, coco_annos_val = coco_annos
    ann_ids = coco_annos_train.getAnnIds(imgIds=img_id)
    coco = coco_annos_train
    if len(ann_ids) == 0:
        ann_ids = coco_annos_val.getAnnIds(imgIds=img_id)
        coco = coco_annos_val
    
    # Dictionary: target coco_annotation file for an image
    coco_annotation = coco.loadAnns(ann_ids)

    # number of objects in the image
    num_objs = len(coco_annotation)

    # Bounding boxes for objects
    # In coco format, bbox = [xmin, ymin, width, height]
    # In pytorch, the input should be [xmin, ymin, xmax, ymax]
    boxes = []
    for i in range(num_objs):
        xmin = coco_annotation[i]['bbox'][0]
        ymin = coco_annotation[i]['bbox'][1]
        xmax = xmin + coco_annotation[i]['bbox'][2]
        ymax = ymin + coco_annotation[i]['bbox'][3]
        boxes.append([xmin, ymin, xmax, ymax])
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # Labels (In my case, I only one class: target class or background)
    labels = torch.ones((num_objs,), dtype=torch.int64)
    # Tensorise img_id
    img_id = torch.tensor([img_id])
    # Size of bbox (Rectangular)
    areas = []
    for i in range(num_objs):
        areas.append(coco_annotation[i]['area'])
    areas = torch.as_tensor(areas, dtype=torch.float32)
    # Iscrowd
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    # Annotation is in dictionary format
    my_annotation = {}
    my_annotation["boxes"] = boxes
    my_annotation["labels"] = labels
    my_annotation["image_id"] = img_id
    my_annotation["area"] = areas
    my_annotation["iscrowd"] = iscrowd
    return my_annotation


class FFN_IRL(Dataset):
    def __init__(self, root_dir, initial_fix, img_info, annos, transform, pa,
                 catIds, coco_annos=None):
        self.img_info = img_info
        self.root_dir = root_dir
        self.img_dir = root_dir + 'images'
        self.transform = transform
        self.pa = pa
        self.bboxes = annos
        self.initial_fix = initial_fix
        self.catIds = catIds
        self.prior_maps = torch.load(root_dir + pa.prior_maps_dir)
        if self.pa.use_DCB_target:
            self.coco_thing_classes = np.load(join(root_dir,
                                                   'coco_thing_classes.npy'),
                                              allow_pickle=True).item()
        if coco_annos:
            self.coco_helper = COCODetHelper(coco_annos)
        else:
            self.coco_helper = None


    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        imgId = self.img_info[idx]
        cat_name, img_name, condition = imgId.split('_')
        if imgId in self.bboxes.keys():
            bbox = self.bboxes[imgId]
        else:
            bbox = None

        im_path = "{}/{}/{}".format(self.img_dir, cat_name.replace(' ', '_'),
                                    img_name)
        im = Image.open(im_path)
        im_tensor = self.transform(im)

        if bbox is not None:
            coding = utils.multi_hot_coding(bbox, self.pa.patch_size,
                                            self.pa.patch_num)
            coding = torch.from_numpy(coding).view(1, -1)
        else:
            coding = torch.zeros(1, self.pa.patch_count)

        # create action mask
        action_mask = np.zeros(self.pa.patch_num[0] * self.pa.patch_num[1],
                               dtype=np.uint8)            
            
        ret = {
            'task_id': self.catIds[cat_name],
            'img_name': img_name,
            'cat_name': cat_name,
            'im_tensor': im_tensor,
            'label_coding': coding,
            'condition': condition,
            'prior_map': self.prior_maps[cat_name],
            'action_mask': torch.from_numpy(action_mask)
        }
        if self.coco_helper:
            centermaps = self.coco_helper.create_centermap_target(
                img_name, self.pa.patch_num[1], self.pa.patch_num[0])
            ret['centermaps'] = centermaps

        if self.pa.use_DCB_target:
            DCBs = torch.load(
                join(self.pa.DCB_dir, cat_name.replace(' ', '_'),
                     img_name[:-3] + 'pth.tar'))
            ret['DCB_target_map'] = DCBs[self.coco_thing_classes[cat_name]]

        return ret


class FFN_Human_Gaze(Dataset):
    """
    Human gaze data in state-action pairs for foveal feature net
    """

    def __init__(self,
                 root_dir,
                 fix_labels,
                 bbox_annos,
                 scene_annos,
                 pa,
                 transform,
                 catIds,
                 blur_action=False,
                 acc_foveal=True,
                 coco_annos=None):
        self.root_dir = root_dir
        self.img_dir = root_dir + 'images'
        self.pa = pa
        self.transform = transform
        # Remove scanpaths longer than max_traj_length
        self.fix_labels = list(
            filter(lambda x: len(x[3]) <= pa.max_traj_length + 1, fix_labels))
        self.catIds = catIds
        self.blur_action = blur_action
        self.acc_foveal = acc_foveal
        self.bboxes = bbox_annos
        self.prior_maps = torch.load(root_dir + pa.prior_maps_dir)
        self.scene_labels = scene_annos['labels']
        self.scene_to_id = scene_annos['id_list']

        if self.pa.use_DCB_target:
            self.coco_thing_classes = np.load(join(root_dir,
                                                   'coco_thing_classes.npy'),
                                              allow_pickle=True).item()
        if coco_annos:
            self.coco_helper = COCODetHelper(coco_annos)
        else:
            self.coco_helper = None


    def __len__(self):
        return len(self.fix_labels)

    def __getitem__(self, idx):
        img_name, cat_name, condition, fixs, action, is_last, sid, dura = self.fix_labels[
            idx]
        imgId = cat_name + '_' + img_name
        if imgId in self.bboxes.keys():
            bbox = self.bboxes[imgId]
        else:
            bbox = None

        im_path = "{}/{}/{}".format(self.img_dir, cat_name.replace(' ', '_'),
                                    img_name)
        im = Image.open(im_path)
        im_tensor = self.transform(im)

        if bbox is not None:
            # coding = utils.multi_hot_coding(bbox, self.pa.patch_size,
            #                                 self.pa.patch_num, thresh=0.9)
            # coding = filters.gaussian_filter(coding, sigma=1)
            # coding /= coding.sum()
            # coding = torch.from_numpy(coding).view(1, -1)
            coding = utils.get_center_keypoint_map(
                bbox, self.pa.patch_num[::-1],
                box_size_dependent=False).view(1, -1)
        else:
            coding = torch.zeros(1, self.pa.patch_count)

        scanpath_length = len(fixs)
        # Pad fixations to max_traj_lenght
        fixs = fixs + [fixs[-1]] * (self.pa.max_traj_length - len(fixs) + 1)
        is_padding = torch.zeros(self.pa.max_traj_length + 1)
        is_padding[scanpath_length:] = 1
        # Discretize fixations
        fixs_tensor = torch.tensor(fixs)
        fixs_tensor = fixs_tensor // torch.tensor([self.pa.patch_size])
        # Normalize to 0-1
        fixs_tensor /= torch.tensor(self.pa.patch_num)
        
        next_fixs_tensor = fixs_tensor.clone()
        if action < self.pa.patch_count: 
            x, y = utils.action_to_pos(action, self.pa.patch_size, self.pa.patch_num)
            next_fix = (torch.tensor([x, y]) // torch.tensor(
                [self.pa.patch_size])) / torch.tensor(self.pa.patch_num)
            next_fixs_tensor[scanpath_length:] = next_fix

        ret = {
            "task_id": self.catIds[cat_name],
            "condition": condition,
            "true_state": im_tensor,
            "true_action": torch.tensor([action], dtype=torch.long),
            'label_coding': coding,
            'img_name': img_name,
            'task_name': cat_name,
            'normalized_fixations': fixs_tensor,
            'next_normalized_fixations': next_fixs_tensor,
            'is_TP': condition == 'present',
            'is_last': is_last,
            'is_padding': is_padding,
            'true_or_fake': 1.0,
            'prior_map': self.prior_maps[cat_name],
            'scanpath_length': scanpath_length,
            'duration': dura,
            'subj_id': sid - 1  # sid ranges from [1, 10]
        }

        if self.coco_helper:
            centermaps = self.coco_helper.create_centermap_target(
                img_name, self.pa.patch_num[1], self.pa.patch_num[0])
            ret['centermaps'] = centermaps

        # precomputed panoptic-FPN target map
        if self.pa.use_DCB_target:
            DCBs = torch.load(
                join(self.pa.DCB_dir, cat_name.replace(' ', '_'),
                     img_name[:-3] + 'pth.tar'))
            ret['DCB_target_map'] = DCBs[self.coco_thing_classes[cat_name]]
            ret['DCBs'] = DCBs

        # compute the map of last fixation
        if self.pa.use_action_map:
            action_map = np.zeros((self.pa.patch_num[1], self.pa.patch_num[0]),
                                  dtype=np.float32)
            action_map[ind_y[-1].item(), ind_x[-1].item()] = 1
            action_map = filters.gaussian_filter(action_map, sigma=1)
            ret['last_fixation_map'] = action_map

        if self.blur_action:
            action_map = np.zeros(self.pa.patch_count, dtype=np.float32)
            if action < self.pa.patch_count:
                action_map[action] = 1
                action_map = action_map.reshape(self.pa.patch_num[1], -1)
                # action_map = filters.gaussian_filter(action_map, sigma=1)
            else:
                action_map = action_map.reshape(self.pa.patch_num[1], -1)
            ret['action_map'] = action_map
        return ret
