import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.config_utils import load_config_by_name
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.changeDataset import ChangeDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
from PIL import Image

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device, config):
        As = data['A']
        Bs = data['B']
        label = data['gt']
        name = data['fn']
        pred = self.sliding_eval_rgbX(As, Bs, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            raw_dir = os.path.join(self.save_path, "raw")
            color_dir = os.path.join(self.save_path, "color")
            paper_qual_dir = os.path.join(self.save_path, "paper_qualitative")
            ensure_dir(raw_dir)
            ensure_dir(color_dir)
            ensure_dir(paper_qual_dir)

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8)*255, mode='P')
            # result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            # class_colors = self.dataset.get_class_colors()
            # palette_list = list(np.array(class_colors).flat)
            # if len(palette_list) < 768:
            #     palette_list += [0] * (768 - len(palette_list))
            # result_img.putpalette(palette_list)
            result_img.save(os.path.join(color_dir, fn))

            # save raw result
            cv2.imwrite(os.path.join(raw_dir, fn), pred)

            # save paper-style TP/TN/FP/FN qualitative result
            # TP: white, TN: black, FP: green, FN: red
            pred_pos = pred == 1
            gt_pos = label == 1
            tp = pred_pos & gt_pos
            tn = (~pred_pos) & (~gt_pos)
            fp = pred_pos & (~gt_pos)
            fn_mask = (~pred_pos) & gt_pos

            qual = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            qual[tn] = [0, 0, 0]
            qual[tp] = [255, 255, 255]
            qual[fp] = [0, 255, 0]
            qual[fn_mask] = [255, 0, 0]
            cv2.imwrite(os.path.join(paper_qual_dir, fn), cv2.cvtColor(qual, cv2.COLOR_RGB2BGR))
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, recall, precision, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc, dice_scalar, dice_per_class = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, recall, precision, freq_IoU, mean_pixel_acc, pixel_acc, dice_scalar,
                                self.dataset.class_names, show_no_back=False)
        return result_line, mean_IoU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument(
        '--config_name', '-n', default='levir', type=str,
        help='config name (configs[/dinov3_vX]/config_<name>.py)'
    )
    parser.add_argument(
        '--dataset_name', default=None, type=str,
        help='deprecated: use --config_name instead'
    )
    parser.add_argument('--split', '-c', default='test', type=str)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    
    config_name = args.config_name or args.dataset_name
    config = load_config_by_name(config_name)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    flops = network.flops()
    print("Gflops of the network: ", flops/(10**9))
    print("number of paramters: ", sum(p.numel() if p.requires_grad==True else 0 for p in network.parameters()))
    1/0
    data_setting = {'root': config.root_folder,
                    'A_format': config.A_format,
                    'B_format': config.B_format,
                    'gt_format': config.gt_format,
                    'class_names': config.class_names}
    val_pre = ValPre()
    dataset = ChangeDataset(data_setting, args.split, val_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image, config)
        _, mean_IoU = segmentor.run_eval(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)

    #visualize erf

    # with torch.enable_grad():
    #     segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, network,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, args.verbose, args.save_path,
    #                                 args.show_image, config)
        
    #     segmentor.get_erf(config.checkpoint_dir, args.epochs)
