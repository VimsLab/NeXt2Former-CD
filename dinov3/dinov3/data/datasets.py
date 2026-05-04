# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Dataset stubs for the trimmed DINOv3 checkout.

The upstream DINOv3 repository includes dataset implementations under
``dinov3.data.datasets``. This workspace does not ship those datasets,
so importing them fails during model initialization even when the
training code does not actually use them.

These lightweight stubs satisfy imports and raise a clear error if a
caller tries to instantiate one of the missing datasets.
"""

from enum import Enum

import torch


class _MissingDataset(torch.utils.data.Dataset):
    """Placeholder dataset that raises on use."""

    class Split(Enum):
        TRAIN = "TRAIN"
        VAL = "VAL"
        TEST = "TEST"

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "dinov3.data.datasets is not included in this checkout. "
            "Please install/checkout the full DINOv3 datasets module if you "
            "need dataset loading via dinov3.data.make_dataset()."
        )

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise RuntimeError("Dataset stub does not provide samples.")


class ImageNet(_MissingDataset):
    pass


class ImageNet22k(_MissingDataset):
    pass


class ADE20K(_MissingDataset):
    pass


class CocoCaptions(_MissingDataset):
    pass


class NYU(_MissingDataset):
    pass
