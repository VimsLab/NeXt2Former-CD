import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.changeDataset import ChangeDataset
from dataloader.dataloader import ValPre
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from eval import SegEvaluator
import shutil

from tensorboardX import SummaryWriter

from utils.config_utils import load_config_by_name

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '16005'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    print(args)
    
    config_name = args.config_name or args.dataset_name
    print("CONFIG NAME::  ", config_name)
    config = load_config_by_name(config_name)

    print("DINOv3 adapter:", getattr(config, 'dinov3_use_adapter', None))
    print("DINOv3 freeze backbone:", getattr(config, 'dinov3_freeze_backbone', False))
    logger.info(f"DINOv3 adapter: {getattr(config, 'dinov3_use_adapter', None)}")
    logger.info(f"DINOv3 freeze backbone: {getattr(config, 'dinov3_freeze_backbone', False)}")

    checkpoint_root = getattr(config, "checkpoint_dir", None)
    if checkpoint_root and osp.basename(checkpoint_root) == "checkpoint":
        run_id = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        config.checkpoint_dir = osp.join(checkpoint_root, run_id)

    print("=======================================")
    print(config.tb_dir)
    print("=======================================")

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Log all configuration parameters
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    
    # Get all config attributes and sort them for consistent logging
    config_dict = {}
    for key in dir(config):
        if not key.startswith('_'):
            try:
                value = getattr(config, key)
                # Skip methods and functions
                if not callable(value):
                    config_dict[key] = value
            except:
                pass
    
    # Log config parameters by category
    logger.info("\n--- Dataset Configuration ---")
    dataset_keys = ['dataset_name', 'root_folder', 'A_format', 'B_format', 'gt_format', 
                    'num_train_imgs', 'num_eval_imgs', 'num_classes', 'class_names',
                    'use_weather_data', 'weather_data_format']
    for key in dataset_keys:
        if key in config_dict:
            logger.info(f"  {key}: {config_dict[key]}")
    
    logger.info("\n--- Image Configuration ---")
    image_keys = ['image_height', 'image_width', 'background', 'norm_mean', 'norm_std']
    for key in image_keys:
        if key in config_dict:
            logger.info(f"  {key}: {config_dict[key]}")
    
    logger.info("\n--- Model Configuration ---")
    model_keys = ['backbone', 'decoder', 'decoder_embed_dim', 'pretrained_model', 'pretrained_path',
                  'dinov3_repo_dir', 'dinov3_model_name', 'dinov3_pretrained', 
                  'dinov3_out_indices', 'dinov3_use_frm']
    for key in model_keys:
        if key in config_dict:
            logger.info(f"  {key}: {config_dict[key]}")
    
    logger.info("\n--- Training Configuration ---")
    train_keys = ['optimizer', 'lr', 'lr_power', 'momentum', 'weight_decay', 'batch_size', 
                  'nepochs', 'niters_per_epoch', 'num_workers', 'train_scale_array', 
                  'warm_up_epoch', 'fix_bias', 'bn_eps', 'bn_momentum', 'seed',
                  'use_dice', 'dice_weight', 'use_CrossE', 'CrossE_weight', 'CrossE_class_weights',
                  'mask2former_train_mode', 'mask2former_class_weights',
                  'mask2former_class_weight', 'mask2former_dice_weight',
                  'mask2former_mask_weight', 'mask2former_no_object_weight',
                  'mask2former_num_points', 'mask2former_oversample_ratio',
                  'mask2former_importance_sample_ratio']
    for key in train_keys:
        if key in config_dict:
            logger.info(f"  {key}: {config_dict[key]}")
    
    logger.info("\n--- Evaluation Configuration ---")
    eval_keys = ['eval_stride_rate', 'eval_scale_array', 'eval_flip', 'eval_crop_size']
    for key in eval_keys:
        if key in config_dict:
            logger.info(f"  {key}: {config_dict[key]}")
    
    logger.info("\n--- Checkpoint Configuration ---")
    checkpoint_keys = ['checkpoint_start_epoch', 'checkpoint_step', 'checkpoint_dir']
    for key in checkpoint_keys:
        if key in config_dict:
            logger.info(f"  {key}: {config_dict[key]}")
    
    logger.info("\n--- Path Configuration ---")
    path_keys = ['root_dir', 'abs_dir', 'log_dir', 'tb_dir', 'log_file', 'val_log_file']
    for key in path_keys:
        if key in config_dict:
            logger.info(f"  {key}: {config_dict[key]}")
    
    # Log any remaining parameters not in the above categories
    all_logged_keys = set(dataset_keys + image_keys + model_keys + train_keys + 
                         eval_keys + checkpoint_keys + path_keys)
    remaining_keys = sorted([k for k in config_dict.keys() if k not in all_logged_keys])
    if remaining_keys:
        logger.info("\n--- Other Configuration ---")
        for key in remaining_keys:
            logger.info(f"  {key}: {config_dict[key]}")
    
    logger.info("=" * 80)
    logger.info("")

    # data loader
    train_loader, train_sampler = get_train_loader(engine, ChangeDataset, config)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)






    

    # config network and criterion

    mask2former_mode = getattr(config, 'mask2former_train_mode', None)
    if config.decoder == 'Mask2Former' and mask2former_mode == 2:
        logger.info("========================================")
        logger.info('Using Mask2Former SetCriterion loss')
        logger.info(f"mask2former_train_mode: {mask2former_mode}")
        logger.info("========================================")
        from utils.mask2former_loss import Mask2FormerSetLoss
        criterion = Mask2FormerSetLoss(
            num_classes=config.num_classes,
            ignore_index=getattr(config, 'ignore_index', 255),
            class_weight=getattr(config, 'mask2former_class_weight', 1.0),
            dice_weight=getattr(config, 'mask2former_dice_weight', 1.0),
            mask_weight=getattr(config, 'mask2former_mask_weight', 1.0),
            no_object_weight=getattr(config, 'mask2former_no_object_weight', 0.1),
            num_points=getattr(config, 'mask2former_num_points', 12544),
            oversample_ratio=getattr(config, 'mask2former_oversample_ratio', 3.0),
            importance_sample_ratio=getattr(config, 'mask2former_importance_sample_ratio', 0.75),
        )
    elif config.decoder == 'Mask2Former' and mask2former_mode == 3:
        class_weights = getattr(config, 'mask2former_class_weights', None) or getattr(config, 'CrossE_class_weights', None)
        logger.info("========================================")
        logger.info('Using weighted CrossEntropyLoss for Mask2Former')
        logger.info(f"mask2former_train_mode: {mask2former_mode}")
        logger.info(f"mask2former_class_weights: {class_weights}")
        logger.info("========================================")
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        else:
            criterion = nn.CrossEntropyLoss(reduction='mean')
    elif config.decoder == 'Mask2Former' and mask2former_mode == 4:
        class_weights = getattr(config, 'mask2former_class_weights', None) or getattr(config, 'CrossE_class_weights', None)
        logger.info("========================================")
        logger.info('Using combined Mask2Former SetCriterion + weighted CrossEntropyLoss')
        logger.info(f"mask2former_train_mode: {mask2former_mode}")
        logger.info(f"mask2former_class_weights: {class_weights}")
        logger.info("========================================")
        from utils.mask2former_loss import Mask2FormerCombinedLoss
        criterion = Mask2FormerCombinedLoss(
            num_classes=config.num_classes,
            ignore_index=getattr(config, 'ignore_index', 255),
            class_weight=getattr(config, 'mask2former_class_weight', 1.0),
            dice_weight=getattr(config, 'mask2former_dice_weight', 1.0),
            mask_weight=getattr(config, 'mask2former_mask_weight', 1.0),
            no_object_weight=getattr(config, 'mask2former_no_object_weight', 0.1),
            num_points=getattr(config, 'mask2former_num_points', 12544),
            oversample_ratio=getattr(config, 'mask2former_oversample_ratio', 3.0),
            importance_sample_ratio=getattr(config, 'mask2former_importance_sample_ratio', 0.75),
            ce_class_weights=class_weights,
            set_loss_weight=getattr(config, 'mask2former_set_loss_weight', 1.0),
            ce_loss_weight=getattr(config, 'mask2former_ce_loss_weight', 1.0),
        )
    else:
        logger.info("========================================")
        logger.info('Using unweighted CrossEntropyLoss as criterion')
        logger.info("========================================")
        
        criterion = nn.CrossEntropyLoss(reduction='mean')


    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

    def _set_backbone_requires_grad(target_model, requires_grad: bool) -> None:
        base_model = target_model.module if isinstance(target_model, DistributedDataParallel) else target_model
        if hasattr(base_model, "backbone"):
            for param in base_model.backbone.parameters():
                param.requires_grad = requires_grad

    freeze_epochs = int(getattr(config, 'dinov3_freeze_epochs', 0) or 0)
    if config.backbone == 'dinov3' and freeze_epochs > 0:
        _set_backbone_requires_grad(model, False)
        logger.info(f"Freezing DINOv3 backbone for first {freeze_epochs} epochs")
    


    # === Load pretrained model state_dict if provided ===
    if hasattr(config, 'pretrained_path') and config.pretrained_path and os.path.isfile(config.pretrained_path):
        logger.info(f"Loading pretrained weights from: {config.pretrained_path}")
        checkpoint = torch.load(config.pretrained_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix if loading DDP weights into a non-DDP model
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded pretrained model with missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
    else:
        logger.info("No pretrained model loaded.")




    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    
    params_list = []
    backbone_lr_mult = float(getattr(config, 'dinov3_backbone_lr_mult', 1.0) or 1.0)

    def _group_weight_excluding_modules(weight_group, module, norm_layer, lr, skip_modules=None):
        skip_modules = set(skip_modules or [])
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if m in skip_modules:
                continue
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                    or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.Parameter):
                group_decay.append(m)

        weight_group.append(dict(params=group_decay, lr=lr, lr_mult=1.0))
        weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr, lr_mult=1.0))
        return weight_group

    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    if config.backbone == 'dinov3' and backbone_lr_mult != 1.0 and hasattr(base_model, "backbone"):
        backbone_modules = set(base_model.backbone.modules())
        params_list = _group_weight_excluding_modules(params_list, base_model, BatchNorm2d, base_lr, backbone_modules)
        params_list = group_weight(params_list, base_model.backbone, BatchNorm2d, base_lr * backbone_lr_mult)
        for group in params_list[-2:]:
            group["lr_mult"] = backbone_lr_mult
        for group in params_list[:-2]:
            group.setdefault("lr_mult", 1.0)
    else:
        params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
        for group in params_list:
            group.setdefault("lr_mult", 1.0)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=True,
            )
            if hasattr(model, "_set_static_graph"):
                model._set_static_graph()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')
    
    # Initialize the evaluation dataset and evaluator
    val_setting = {'root': config.root_folder,
                    'A_format': config.A_format,
                    'B_format': config.B_format,
                    'gt_format': config.gt_format,
                    'class_names': config.class_names,
                    'use_weather_data': getattr(config, "use_weather_data", False),
                    'weather_data_format': getattr(config, "weather_data_format", None),}

    val_pre = ValPre()
    val_dataset = ChangeDataset(val_setting, 'val', val_pre)

    best_mean_iou = 0.0  # Track the best mean IoU for model saving
    best_epoch = 100000  # Track the epoch with the best mean IoU for model saving
    
    for epoch in range(engine.state.epoch, config.nepochs+1):
        if config.backbone == 'dinov3' and freeze_epochs > 0 and epoch == freeze_epochs + 1:
            _set_backbone_requires_grad(model, True)
            logger.info(f"Unfroze DINOv3 backbone at epoch {epoch}")
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)
            try:
                minibatch = next(dataloader)
            except StopIteration:
                break
            As = minibatch['A']
            Bs = minibatch['B']
            gts = minibatch['gt']


            As = As.cuda(non_blocking=True)
            Bs = Bs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            aux_rate = 0.2
            loss = model(As, Bs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                lr_mult = optimizer.param_groups[i].get('lr_mult', 1.0)
                optimizer.param_groups[i]['lr'] = lr * lr_mult

            if engine.distributed:
                if dist.get_rank() == 0:
                    sum_loss += reduce_loss.item()
                    print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
                    pbar.set_description(print_str, refresh=False)
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
                pbar.set_description(print_str, refresh=False)
            del loss
            
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            tb.add_scalar('lr', lr, epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
        
        # devices_val = [engine.local_rank] if engine.distributed else [0]
        torch.cuda.empty_cache()
        if engine.distributed:
            if dist.get_rank() == 0:
                # only test on rank 0, otherwise there would be some synchronization problems
                # evaluation to decide whether to save the model
                if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                    model.eval() 
                    with torch.no_grad():
                        all_dev = parse_devices(args.devices)
                        # network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d).cuda(all_dev[0])
                        segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                                                norm_mean=config.norm_mean, norm_std=config.norm_std,
                                                network=model, multi_scales=config.eval_scale_array,
                                                is_flip=config.eval_flip, devices=[model.device],
                                                verbose=False, config=config
                                                )
                        result_line, mean_IoU, dice = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                config.link_val_log_file)
                        print('mean_IoU:', mean_IoU)
                        
                        # Log validation metrics to TensorBoard
                        tb.add_scalar('val_mean_IoU', mean_IoU, epoch)                        # Log per-class dice scores
                        for class_idx, dice_score in enumerate(dice):
                            if class_idx < len(config.class_names):
                                tb.add_scalar(f'val_dice_{config.class_names[class_idx]}', dice_score, epoch)
                                tb.add_scalar(f'val_f1_{config.class_names[class_idx]}', dice_score, epoch)
                        if len(dice) > 0:
                            tb.add_scalar('val_f1_mean', sum(dice) / len(dice), epoch)
                        # Determine if the model performance improved
                        if mean_IoU > best_mean_iou:
                            # If the model improves, remove the saved checkpoint for this epoch
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                            best_epoch = epoch
                            best_mean_iou = mean_IoU
                        else:
                            # If the model does not improve, remove the saved checkpoint for this epoch
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                        
                    model.train()
        else:
            if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                model.eval() 
                with torch.no_grad():
                    devices_val = [engine.local_rank] if engine.distributed else [0]
                    segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                                            norm_mean=config.norm_mean, norm_std=config.norm_std,
                                            network=model, multi_scales=config.eval_scale_array,
                                            is_flip=config.eval_flip, 
                                            # devices=[1,2,3],
                                            devices=[0],
                                            verbose=False, config=config
                                            )
                    _, mean_IoU, dice = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                config.link_val_log_file)
                    print('mean_IoU:', mean_IoU)
                    
                    # Log validation metrics to TensorBoard
                    tb.add_scalar('val_mean_IoU', mean_IoU, epoch)                    # Log per-class dice scores
                    for class_idx, dice_score in enumerate(dice):
                        if class_idx < len(config.class_names):
                            tb.add_scalar(f'val_dice_{config.class_names[class_idx]}', dice_score, epoch)
                            tb.add_scalar(f'val_f1_{config.class_names[class_idx]}', dice_score, epoch)
                    if len(dice) > 0:
                        tb.add_scalar('val_f1_mean', sum(dice) / len(dice), epoch)
                    # Determine if the model performance improved
                    if mean_IoU > best_mean_iou:
                        # If the model improves, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        best_epoch = epoch
                        best_mean_iou = mean_IoU
                    else:
                        # If the model does not improve, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                model.train()
