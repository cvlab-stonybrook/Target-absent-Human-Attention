import torch.nn as nn
import torch
from torchvision import models
from torch.distributions import Categorical


class TargetDetector(nn.Module):
    """Task-conditioned scanpath generator using Feature Foveal Network."""

    def __init__(self,
                 foveal_feat_size,
                 learnable_task_embedding=True,
                 num_actions=640,
                 num_targets=18,
                 target_embedding_size=128,
                 hidden_size=32,
                 is_cumulative=True, 
                 dropout_target_prob=1.0):
        super(TargetDetector, self).__init__()
        self.is_cumulative = is_cumulative
        self.num_targets = num_targets
        self.num_actions = num_actions
        self.num_coco_classes = 80
        
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-3]))
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.centermap_predictor = nn.Sequential(
            nn.Conv2d(1024,
                      foveal_feat_size,
                      3,
                      stride=1,
                      padding=1,
                      bias=False), 
            nn.BatchNorm2d(foveal_feat_size), nn.ELU(),

            nn.Conv2d(foveal_feat_size,
                      self.num_targets,
                      1,
                      bias=True),
            nn.Sigmoid()
        )
        
    
    def forward(self, im_tensor):
        with torch.no_grad():
            x = self.resnet(im_tensor)
        bs, ch, h, w = x.size()

        centermaps = self.centermap_predictor(x)
        return centermaps

class ScanpathGenerationWrapper(object):
    def __init__(self, model, has_stop):
        self.model = model
        self.has_stop = has_stop

    def select_action(self,
                      state,
                      sample_action,
                      action_mask=None,
                      softmask=False,
                      eps=1e-12):
        bs = state[0].size(0)
        with torch.no_grad():
            q = self.model(*state, output_stop=self.has_stop)
            if self.has_stop:
                q, s = q
                stop_probs = F.sigmoid(s).view(-1)
                stop = stop_probs > 0.5
            else:
                stop = torch.zeros(q.size(0))
            probs = q

            if sample_action:
                if action_mask is not None:
                    # prevent sample previous actions by re-normalizing probs
                    probs[action_mask] = eps
                    probs /= probs.sum(dim=1).view(probs.size(0), 1)

                m = Categorical(probs)
                actions = m.sample()
            else:
                if action_mask is not None:
                    if probs.size(-1) % 2 == 1:
                        probs[:, :-1][action_mask] = eps
                    else:
                        probs[action_mask] = eps
                actions = torch.argmax(probs, dim=1)

            return actions, stop

class TerminatorWrapper(nn.Module):
    """Termination predictor wrapper for the scanpath predictor"""
    def __init__(self, scanpath_predictor, clf, is_DCB=False):
        super(TerminatorWrapper, self).__init__()
        self.clf = clf
        self.model = scanpath_predictor
        self.is_DCB = is_DCB
        
    def forward(self, im_tensor, fixations, tids=None, n_fixs=None, output_stop=False):
        if self.is_DCB:
            tids = fixations  # the 2nd input is actually tids for DCB
            probs, _ = self.model(im_tensor, tids, act_only=True)
        else:
            q = self.model(im_tensor)
            q = q[torch.arange(q.size(0)), tids].view(q.size(0), -1)
            probs = q / q.sum(dim=1, keepdim=True)
            
        if output_stop:
            assert n_fixs is not None, "missing n_fixs"
            history_emb = get_sinuposition_encoding(
                fixations[:, 0], emb_size, hparams.Data.patch_num)
            for i in range(1, int(n_fixs[0].item())):
                history_emb += get_sinuposition_encoding(
                    fixations[:, i], emb_size, hparams.Data.patch_num)
            x = torch.cat([q.cpu(), history_emb.cpu()], dim=1).numpy()
            
            stop_logits = self.clf.decision_function(x)
        
            return probs, torch.from_numpy(stop_logits)
        else:
            return probs
