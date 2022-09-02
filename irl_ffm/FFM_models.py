import torch.nn as nn
import torch
import numpy as np
from torchvision import models
import torch.nn.functional as F
from .utils import get_foveal_weights


def get_sinuposition_encoding_all(d_model, patch_num):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(patch_num[1]), 
        torch.arange(patch_num[0]))

    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0, d_model, 2) * (
        -np.log(10000.0) / d_model)).unsqueeze(1).unsqueeze(1)
    
    pe_x = torch.zeros(d_model, *grid_y.shape)
    pe_x[0::2] = torch.sin(grid_x * div_term)
    pe_x[1::2] = torch.cos(grid_x * div_term)
    
    pe_y = torch.zeros(d_model, *grid_y.shape)
    pe_y[0::2] = torch.sin(grid_y * div_term)
    pe_y[1::2] = torch.cos(grid_y * div_term)
    
    pe = torch.cat([pe_x, pe_y], dim=0)
    
    return pe

def get_sinuposition_encoding(current_fix, d_model, patch_num):
    batch_size = current_fix.size(0)
    device = current_fix.device
    if current_fix.max() <= 1:
        # Mapping to action space
        current_fix = current_fix * torch.Tensor(patch_num).to(device)
    
    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)).to(device)
    
    fix_x = current_fix[:, 0].unsqueeze(1)
    pe_x = torch.zeros(batch_size, d_model).to(device)
    pe_x[:, 0::2] = torch.sin(fix_x * div_term)
    pe_x[:, 1::2] = torch.cos(fix_x * div_term)
    
    fix_y = current_fix[:, 1].unsqueeze(1)
    pe_y = torch.zeros(batch_size, d_model).to(device)
    pe_y[:, 0::2] = torch.sin(fix_y * div_term)
    pe_y[:, 1::2] = torch.cos(fix_y * div_term)
    
    pe = torch.cat([pe_x, pe_y], dim=1)
    
    return pe

class FeatureFovealNet(nn.Module):
    def __init__(self,
                 feat_dim,
                 kernel_size=1):
        super(FeatureFovealNet, self).__init__()

        self.feat_dim = feat_dim
        self.out_dims = [64, 256, 512, 1024, 2048]
        self.conv1x1_ops = []
        for dim in self.out_dims:
            self.conv1x1_ops.append(nn.Conv2d(dim, self.feat_dim, kernel_size, padding=kernel_size//2))
        self.conv1x1_ops = nn.ModuleList(self.conv1x1_ops)

        self.fovea_size = nn.Parameter(torch.tensor(7.5))
        self.amplitude = nn.Parameter(torch.tensor(0.68))
        self.acuity = nn.Parameter(torch.tensor(1.25))
        

    def forward(self, conv_outs, fixations, get_weight=True, is_cumulative=True):

        uni_conv_outs = [conv_outs[0]]
        for i, out in enumerate(conv_outs[1:]):
            uni_conv_outs.append(F.upsample(out, scale_factor=2**(i + 1)))

        for i in range(len(uni_conv_outs)):
            uni_conv_outs[i] = self.conv1x1_ops[i](
                uni_conv_outs[i]).unsqueeze(1)

        uni_conv_outs = torch.cat(uni_conv_outs, dim=1)

        h, w = uni_conv_outs.shape[-2:]

        weights = get_foveal_weights(
            fixations,
            w,
            h,
            p=self.fovea_size,
            k=self.amplitude,
            alpha=self.acuity) if get_weight else fixations
        weights = weights.unsqueeze(2).expand(-1, -1, self.feat_dim, -1, -1)
        foveal_features = torch.sum(weights * uni_conv_outs, dim=1)

        return foveal_features

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
#         out = out + x
        return out,attention
    

class FeatureFovealNet_4Levels(nn.Module):
    def __init__(self,
                 feat_dim):
        super(FeatureFovealNet_4Levels, self).__init__()

        self.feat_dim = feat_dim
        self.out_dims = [256, 512, 1024, 2048]
        self.conv1x1_ops = []
        for dim in self.out_dims:
            self.conv1x1_ops.append(nn.Conv2d(dim, self.feat_dim, 1))
        self.conv1x1_ops = nn.ModuleList(self.conv1x1_ops)

#         self.fovea_size = nn.Parameter(torch.tensor(4))
        self.amplitude = nn.Parameter(torch.tensor(0.68))
        self.acuity = nn.Parameter(torch.tensor(1.25))
        

    def forward(self, conv_outs, fixations, get_weight=True, is_cumulative=True):

        uni_conv_outs = [conv_outs[0]]
        for i, out in enumerate(conv_outs[1:]):
            uni_conv_outs.append(F.upsample(out, scale_factor=2**(i + 1)))

        for i in range(len(uni_conv_outs)):
            uni_conv_outs[i] = self.conv1x1_ops[i](
                uni_conv_outs[i]).unsqueeze(1)

        uni_conv_outs = torch.cat(uni_conv_outs, dim=1)

        h, w = uni_conv_outs.shape[-2:]

        weights = get_foveal_weights(
            fixations,
            w,
            h,
            p=4, #self.fovea_size,
            k=self.amplitude,
            alpha=self.acuity) if get_weight else fixations
        weights = weights[:, :-1]
        weights = weights.unsqueeze(2).expand(-1, -1, self.feat_dim, -1, -1)
        foveal_features = torch.sum(weights * uni_conv_outs, dim=1)

        return foveal_features
    
class FFNGeneratorCond_V2(nn.Module):
    """Task-conditioned scanpath generator using Feature Foveal Network."""

    def __init__(self,
                 foveal_feat_size,
                 patch_num,
                 kernel_size=1,
                 learnable_task_embedding=True,
                 num_actions=640,
                 num_targets=18,
                 target_embedding_size=128,
                 hidden_size=32,
                 is_cumulative=True, 
                 dropout_target_prob=1.0):
        super(FFNGeneratorCond_V2, self).__init__()
        self.is_cumulative = is_cumulative
        self.FFN = FeatureFovealNet(foveal_feat_size, kernel_size)
        self.num_targets = num_targets
        self.num_actions = num_actions
        self.num_coco_classes = 80
        self.patch_num = patch_num
        
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Actor
        self.layer1 = nn.Sequential(
            nn.Conv2d(foveal_feat_size,
                      hidden_size,
                      3,
                      stride=1,
                      padding=1,
                      bias=False), 
            nn.LayerNorm([hidden_size, 160, 256]), nn.ELU(),
            nn.Conv2d(hidden_size, hidden_size, 1, stride=2, bias=False),
            nn.LayerNorm([hidden_size, 80, 128]))
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_size,
                      hidden_size,
                      3,
                      stride=1,
                      padding=1,
                      bias=False), 
            nn.LayerNorm([hidden_size, 80, 128]), nn.ELU(),
            nn.Conv2d(hidden_size, hidden_size, 1, stride=2, bias=False),
            nn.LayerNorm([hidden_size, 40, 64]))
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_size,
                          hidden_size,
                          3,
                          stride=2,
                          padding=1,
                          bias=False), 
            nn.LayerNorm([hidden_size, 20, 32]), nn.ELU())

        
        self.q_value_estimator = nn.Sequential(
            nn.Conv2d(hidden_size,
                      hidden_size,
                      3,
                      stride=1,
                      padding=1,
                      bias=False), 
            nn.LayerNorm([hidden_size, 20, 32]), nn.ELU(),

            nn.Conv2d(hidden_size,
                      self.num_targets,
                      1,
                      bias=True)
        )
        
        self.centermap_predictor = nn.Sequential(
            nn.Conv2d(hidden_size,
                      2 * hidden_size,
                      3,
                      stride=1,
                      padding=1,
                      bias=False), 
            nn.LayerNorm([2 * hidden_size, 20, 32]), nn.ELU(),

            nn.Conv2d(2 * hidden_size,
                      self.num_coco_classes,
                      1,
                      bias=True),
            nn.Sigmoid()
        )
        
        self.pos_emb_mat = get_sinuposition_encoding_all(hidden_size // 2, self.patch_num).unsqueeze(0)
        self.self_attn_layer = Self_Attn(hidden_size)
        
    def forward(self, im_tensor, fixations, tids, do_aux_task=False):
        """
            Output a Q-value vector (a Q-value for each actions)
        """
            
        # Construct foveated feature from images and fixations as state representation
        with torch.no_grad():
            conv1_out = self.resnet.conv1(im_tensor)
            x = self.resnet.bn1(conv1_out)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            conv2_out = self.resnet.layer1(x)
            conv3_out = self.resnet.layer2(conv2_out)
            conv4_out = self.resnet.layer3(conv3_out)
            conv5_out = self.resnet.layer4(conv4_out)
            conv_outs = [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]

        foveal_featmaps = self.FFN(conv_outs, fixations, is_cumulative=self.is_cumulative)
        x = self.layer1(foveal_featmaps)
        x = self.layer2(x)
        x = self.layer3(x)
        bs, ch, h, w = x.size()
        curr_fix_pos_emb = get_sinuposition_encoding(
            fixations[:, -1], ch // 2, self.patch_num).unsqueeze(-1).unsqueeze(-1)
        xx = x + torch.cat([self.pos_emb_mat.to(im_tensor.device).expand(bs, ch // 2, h, w), 
                            curr_fix_pos_emb.expand(bs, ch // 2, h, w)], dim=1)
        xx, _ = self.self_attn_layer(xx)
        
        q_values = self.q_value_estimator(xx)
        q_values = q_values[torch.arange(bs), tids].view(bs, -1)
        
        if do_aux_task:
            centermaps = self.centermap_predictor(x)
            return q_values, centermaps

        return q_values

class FFNGeneratorCond_4Levels(nn.Module):
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
        super(FFNGeneratorCond_4Levels, self).__init__()
        self.is_cumulative = is_cumulative
        self.FFN = FeatureFovealNet_4Levels(foveal_feat_size)
        self.num_targets = num_targets
        self.num_actions = num_actions
        self.num_coco_classes = 80
        
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Actor

        self.layer2 = nn.Sequential(
            nn.Conv2d(foveal_feat_size,
                      hidden_size,
                      3,
                      stride=1,
                      padding=1,
                      bias=False), 
            nn.LayerNorm([hidden_size, 80, 128]), nn.ELU(),
            nn.Conv2d(hidden_size, hidden_size, 1, stride=2, bias=False),
            nn.LayerNorm([hidden_size, 40, 64]))
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_size,
                          hidden_size,
                          3,
                          stride=2,
                          padding=1,
                          bias=False), 
            nn.LayerNorm([hidden_size, 20, 32]), nn.ELU())

        
        self.q_value_estimator = nn.Sequential(
            nn.Conv2d(hidden_size,
                      hidden_size,
                      3,
                      stride=1,
                      padding=1,
                      bias=False), 
            nn.LayerNorm([hidden_size, 20, 32]), nn.ELU(),

            nn.Conv2d(hidden_size,
                      self.num_targets,
                      1,
                      bias=True)
        )
        
        self.centermap_predictor = nn.Sequential(
            nn.Conv2d(hidden_size,
                      2 * hidden_size,
                      3,
                      stride=1,
                      padding=1,
                      bias=False), 
            nn.LayerNorm([2 * hidden_size, 20, 32]), nn.ELU(),

            nn.Conv2d(2 * hidden_size,
                      self.num_coco_classes,
                      1,
                      bias=True),
            nn.Sigmoid()
        )
        
        self.self_attn_layer = Self_Attn(hidden_size)
        
    def forward(self, im_tensor, fixations, tids, do_aux_task=False):
        """
            Output a Q-value vector (a Q-value for each actions)
        """
            
        # Construct foveated feature from images and fixations as state representation
        with torch.no_grad():
            conv1_out = self.resnet.conv1(im_tensor)
            x = self.resnet.bn1(conv1_out)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            conv2_out = self.resnet.layer1(x)
            conv3_out = self.resnet.layer2(conv2_out)
            conv4_out = self.resnet.layer3(conv3_out)
            conv5_out = self.resnet.layer4(conv4_out)
            conv_outs = [conv2_out, conv3_out, conv4_out, conv5_out]

        x = self.FFN(conv_outs, fixations, is_cumulative=self.is_cumulative)
        x = self.layer2(x)
        x = self.layer3(x)
        bs, ch, h, w = x.size()
        
        q_values = self.q_value_estimator(x)
        q_values = q_values[torch.arange(bs), tids].view(bs, -1)
        
        if do_aux_task:
            centermaps = self.centermap_predictor(x)
            return q_values, centermaps

        return q_values
    
