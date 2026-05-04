import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

C = edict()
config = C
cfg = C

C.seed = 3407

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'LEVIR'
C.root_folder = osp.join(os.environ.get('DATASETS_ROOT', './datasets'), 'LEVIR-CD256')
C.A_format = '.png'
C.B_format = '.png'
C.gt_format = '.png'
C.is_test = False
C.num_train_imgs = 7120
C.num_eval_imgs = 1024
C.num_classes = 2
C.class_names =  ['background', 'change']

"""Image Config"""
C.background = 255
C.image_height = 256
C.image_width = 256
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'dinov3'
C.pretrained_model = None # do not need to change
C.decoder = 'Mask2Former' # MambaDecoder or MLPDecoder or Mask2Former
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

# Mask2Former training modes:
# 1 = logit decoder + unweighted CE
# 2 = Mask2Former set-based losses (default)
# 3 = logit decoder + weighted CE
# 4 = combined set-based losses + weighted CE
C.mask2former_train_mode = 4
C.mask2former_class_weights = [1.0, 1.0]  # e.g., [1.0, 5.0]
C.mask2former_class_weight = 1.0
C.mask2former_dice_weight = 1.0
C.mask2former_mask_weight = 1.0
C.mask2former_no_object_weight = 0.1
C.mask2former_num_points = 12544
C.mask2former_oversample_ratio = 3.0
C.mask2former_importance_sample_ratio = 0.75
C.mask2former_set_loss_weight = 0.1
C.mask2former_ce_loss_weight = 10

# DINOv3 options
# Example SAT-493M weights:
#   ViT-L/16: pretrained/DINOv3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
#   ViT-7B/16: pretrained/DINOv3/dinov3_vit7b16_pretrain_sat493m-a6675841.pth
C.dinov3_repo_dir = os.environ.get('DINOV3_REPO', './dinov3')
C.dinov3_model_name = 'dinov3_convnext_large'
C.dinov3_pretrained = os.path.join('pretrained', 'DINOv3', 'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth')
C.dinov3_out_indices = None  # auto-select 4 evenly spaced transformer blocks
C.dinov3_use_frm = True
C.dinov3_use_deform_attn = True
C.dinov3_deform_n_heads = 8
C.dinov3_deform_n_points = 6
C.dinov3_freeze_epochs = 0
C.dinov3_backbone_lr_mult = 0.1

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 150
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 16
C.train_scale_array = [1]
C.train_scale_array = None
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
# C.eval_iter = 1
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] 
C.eval_flip = False
C.eval_crop_size = [256, 256]

"""Store Config"""
C.checkpoint_start_epoch = 2
C.checkpoint_step = 2

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

# Build log directory path with decoder and specific backbone model name
backbone_str = C.backbone
if C.backbone == 'dinov3' and hasattr(C, 'dinov3_model_name'):
    backbone_str = C.dinov3_model_name
    # Extract pretrained dataset name from filename
    if hasattr(C, 'dinov3_pretrained'):
        filename = os.path.basename(C.dinov3_pretrained)
        if 'pretrain_' in filename:
            start = filename.find('pretrain_') + len('pretrain_')
            end = filename.find('-', start)
            if end != -1:
                pretrained_dataset = filename[start:end]
                backbone_str = backbone_str + '_' + pretrained_dataset
C.log_dir = osp.abspath('log_paper_final_8bs/log_LEVIR/' + 'log_' + C.dataset_name + '_' + backbone_str + '_' + C.decoder)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
