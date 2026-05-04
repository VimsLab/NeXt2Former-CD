import os
import cv2
import argparse
import numpy as np
import time

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.config_utils import load_config_by_name, load_config_by_path
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.changeDataset import ChangeDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
from PIL import Image

logger = get_logger()


def _parse_bool_flag(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use T/F, True/False, or 1/0."
    )

class SegEvaluator(Evaluator):
    def __init__(self, *args, log_saved_every=0, time_warmup=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_saved_every = int(log_saved_every)
        self._num_saved = 0
        self.time_warmup = max(0, int(time_warmup))
        self._timed_samples = 0

    def func_per_iteration(self, data, device, config):
        As = data['A']
        Bs = data['B']
        label = data['gt']
        name = data['fn']

        infer_start = time.perf_counter()
        pred = self.sliding_eval_rgbX(As, Bs, config.eval_crop_size, config.eval_stride_rate, device)
        inference_time_ms = (time.perf_counter() - infer_start) * 1000.0
        self._timed_samples += 1
        should_count_time = self._timed_samples > self.time_warmup
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}
        if should_count_time:
            results_dict['inference_time_ms'] = inference_time_ms

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

            self._num_saved += 1
            if self.log_saved_every > 0 and self._num_saved % self.log_saved_every == 0:
                logger.info(f"Saved {self._num_saved} predictions")

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
        inference_times_ms = []
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1
            if 'inference_time_ms' in d:
                inference_times_ms.append(d['inference_time_ms'])

        print("correct: ", correct, " labeled: ", labeled, " count: ", count)

        iou, recall, precision, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc, dice_scalar, dice_per_class = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, recall, precision, freq_IoU, mean_pixel_acc, pixel_acc, dice_scalar,
                                self.dataset.class_names, show_no_back=False)
        if len(inference_times_ms) > 0:
            avg_ms = float(np.mean(inference_times_ms))
            std_ms = float(np.std(inference_times_ms))
            logger.info(
                "Average Inference Time Per Image Pair (ms): %.3f | Std (ms): %.3f | Timed Pairs: %d | Warmup Skipped: %d",
                avg_ms,
                std_ms,
                len(inference_times_ms),
                self.time_warmup,
            )
            result_line += (
                f"\nAverage Inference Time Per Image Pair (ms): {avg_ms:.3f}"
                f"\nInference Time Std (ms): {std_ms:.3f}"
                f"\nTimed Pairs: {len(inference_times_ms)}"
                f"\nTiming Warmup Skipped: {self.time_warmup}\n"
            )
        else:
            logger.info(
                "Average Inference Time Per Image Pair (ms): N/A | Timed Pairs: 0 | Warmup Skipped: %d",
                self.time_warmup,
            )
            result_line += (
                "\nAverage Inference Time Per Image Pair (ms): N/A"
                f"\nTimed Pairs: 0"
                f"\nTiming Warmup Skipped: {self.time_warmup}\n"
            )
        return result_line, mean_IoU, dice_per_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument(
        '--log_saved_every',
        type=int,
        default=0,
        help='log every N saved visualization files; 0 disables per-save logging',
    )
    parser.add_argument(
        '--save_visualizations',
        default=None,
        type=_parse_bool_flag,
        help='set True/False to enable or disable saving predicted masks',
    )
    parser.add_argument(
        '--config_name', '-n', default='levir', type=str,
        help='config name (configs[/dinov3_vX]/config_<name>.py)'
    )
    parser.add_argument(
        '--config_path', default=None, type=str,
        help='path to config python file that defines `config`'
    )
    parser.add_argument(
        '--dataset_name', default=None, type=str,
        help='deprecated: use --config_name instead'
    )
    parser.add_argument('--split', '-c', default='val', type=str)
    parser.add_argument(
        '--checkpoint_dir', '-k', default=None, type=str,
        help='path to checkpoint directory (overrides config.checkpoint_dir)'
    )
    parser.add_argument(
        '--checkpoint_path', default=None, type=str,
        help='path to a specific checkpoint file (.pth), overrides --checkpoint_dir/--epochs'
    )
    parser.add_argument(
        '--time_warmup',
        default=0,
        type=int,
        help='number of initial image pairs to skip when averaging inference time',
    )
    # torch.distributed.launch passes --local-rank; accept and ignore for single-GPU eval
    parser.add_argument('--local-rank', type=int, default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    
    if args.config_path:
        config = load_config_by_path(args.config_path)
        config_tag = os.path.splitext(os.path.basename(args.config_path))[0]
    else:
        config_name = args.config_name or args.dataset_name
        config = load_config_by_name(config_name)
        config_tag = (config_name or "config").replace("/", "_")

    checkpoint_dir = config.checkpoint_dir
    if args.checkpoint_dir:
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    checkpoint_path = None
    if args.checkpoint_path:
        checkpoint_path = os.path.abspath(args.checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint_tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
    else:
        checkpoint_tag = f"epoch_{args.epochs}"

    save_visualizations = args.save_visualizations
    if save_visualizations is None:
        save_visualizations = args.save_path is not None
    save_path = None
    if save_visualizations:
        if not args.save_path:
            raise ValueError("--save_visualizations is True but --save_path is not set.")
        run_name = f"{config_tag}__{checkpoint_tag}__{args.split}"
        save_path = os.path.join(os.path.abspath(args.save_path), run_name)
        logger.info(f"Visualization outputs will be saved under: {save_path}")

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    flops = network.flops()
    print("Gflops of the network: ", flops/(10**9))
    print("number of paramters: ", sum(p.numel() if p.requires_grad==True else 0 for p in network.parameters()))
    # 1/0
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
                                 all_dev, args.verbose, save_path,
                                 args.show_image, config,
                                 log_saved_every=args.log_saved_every,
                                 time_warmup=args.time_warmup)
        model_indice = checkpoint_path if checkpoint_path is not None else args.epochs
        _, mean_IoU = segmentor.run_eval(
            checkpoint_dir,
            model_indice,
            config.val_log_file,
            config.link_val_log_file,
        )

    #visualize erf

    # with torch.enable_grad():
    #     segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, network,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, args.verbose, args.save_path,
    #                                 args.show_image, config)
        
    #     segmentor.get_erf(config.checkpoint_dir, args.epochs)
