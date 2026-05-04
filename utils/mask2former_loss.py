import os
import sys
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_mask2former_repo_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mask2former_repo = os.path.join(repo_root, "Mask2Former")
    if mask2former_repo not in sys.path:
        sys.path.insert(0, mask2former_repo)


_ensure_mask2former_repo_on_path()
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher


class Mask2FormerSetLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        class_weight: float = 1.0,
        dice_weight: float = 1.0,
        mask_weight: float = 1.0,
        no_object_weight: float = 0.1,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=num_points,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=["labels", "masks"],
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
        )

    def _build_targets(self, labels: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        if labels.dim() != 3:
            raise ValueError("Mask2FormerSetLoss expects labels with shape [B, H, W].")

        targets: List[Dict[str, torch.Tensor]] = []
        batch_size = labels.shape[0]

        for b in range(batch_size):
            label_map = labels[b]
            classes = torch.unique(label_map)
            classes = classes[classes != self.ignore_index]

            if classes.numel() == 0:
                targets.append(
                    {
                        "labels": torch.zeros((0,), dtype=torch.int64, device=labels.device),
                        "masks": torch.zeros((0, label_map.shape[0], label_map.shape[1]), device=labels.device),
                    }
                )
                continue

            masks = []
            for class_id in classes.tolist():
                masks.append(label_map == class_id)

            mask_tensor = torch.stack(masks, dim=0).float()
            targets.append(
                {
                    "labels": classes.to(torch.int64),
                    "masks": mask_tensor,
                }
            )

        return targets

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        targets = self._build_targets(labels)
        loss_dict = self.criterion(outputs, targets)
        return sum(loss_dict.values())


def _semantic_logits_from_raw(outputs: Dict[str, torch.Tensor], logits: bool = True) -> torch.Tensor:
    pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
    pred_masks = outputs["pred_masks"]    # [B, Q, H, W]

    if logits:
        class_log_probs = F.log_softmax(pred_logits, dim=-1)[..., :-1]
        mask_log_probs = F.logsigmoid(pred_masks)
        return torch.logsumexp(
            class_log_probs[:, :, :, None, None] + mask_log_probs[:, :, None, :, :], dim=1
        )

    mask_cls = F.softmax(pred_logits, dim=-1)[..., :-1]
    mask_pred = pred_masks.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)


class Mask2FormerCombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        class_weight: float = 1.0,
        dice_weight: float = 1.0,
        mask_weight: float = 1.0,
        no_object_weight: float = 0.1,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        ce_class_weights: Optional[List[float]] = None,
        set_loss_weight: float = 1.0,
        ce_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.set_loss_weight = set_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.set_loss = Mask2FormerSetLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weight=class_weight,
            dice_weight=dice_weight,
            mask_weight=mask_weight,
            no_object_weight=no_object_weight,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
        )
        if ce_class_weights is not None:
            ce_weight_tensor = torch.tensor(ce_class_weights)
            self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight_tensor)
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        set_loss_val = self.set_loss(outputs, labels)
        logits = _semantic_logits_from_raw(outputs, logits=True)
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        ce_loss_val = self.ce_loss(logits, labels.long())
        return (self.set_loss_weight * set_loss_val) + (self.ce_loss_weight * ce_loss_val)