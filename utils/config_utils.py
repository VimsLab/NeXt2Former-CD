import importlib
import importlib.util
import os
from types import ModuleType


def _build_module_candidates(config_name: str):
    normalized = config_name.strip()
    if normalized.endswith(".py"):
        normalized = normalized[:-3]

    # Allow users to pass module-like paths (e.g., dinov3_v2.levir_xxx)
    if "/" in normalized:
        normalized = normalized.replace("/", ".")

    if normalized.startswith("configs."):
        return [normalized]

    if normalized.startswith("config_"):
        prefixed = normalized
        unprefixed = normalized[len("config_") :]
    else:
        prefixed = f"config_{normalized}"
        unprefixed = normalized

    candidates = []
    for name in {prefixed, f"config_{unprefixed}"}:
        candidates.extend(
            [
                f"configs.{name}",
                f"configs.dinov3_v0.{name}",
                f"configs.dinov3_v1.{name}",
                f"configs.dinov3_v2.{name}",
                f"configs.dinov3_v3.{name}",
                f"configs.dinov3_v5.{name}",
            ]
        )

    return candidates


def _apply_config_defaults(cfg):
    if getattr(cfg, "decoder", None) == "Mask2Former":
        if not hasattr(cfg, "mask2former_train_mode"):
            cfg.mask2former_train_mode = 2
        if not hasattr(cfg, "mask2former_class_weights"):
            cfg.mask2former_class_weights = None
        if not hasattr(cfg, "mask2former_class_weight"):
            cfg.mask2former_class_weight = 1.0
        if not hasattr(cfg, "mask2former_dice_weight"):
            cfg.mask2former_dice_weight = 1.0
        if not hasattr(cfg, "mask2former_mask_weight"):
            cfg.mask2former_mask_weight = 1.0
        if not hasattr(cfg, "mask2former_no_object_weight"):
            cfg.mask2former_no_object_weight = 0.1
        if not hasattr(cfg, "mask2former_num_points"):
            cfg.mask2former_num_points = 12544
        if not hasattr(cfg, "mask2former_oversample_ratio"):
            cfg.mask2former_oversample_ratio = 3.0
        if not hasattr(cfg, "mask2former_importance_sample_ratio"):
            cfg.mask2former_importance_sample_ratio = 0.75
        if not hasattr(cfg, "mask2former_set_loss_weight"):
            cfg.mask2former_set_loss_weight = 1.0
        if not hasattr(cfg, "mask2former_ce_loss_weight"):
            cfg.mask2former_ce_loss_weight = 1.0
    return cfg


def _config_from_module(module: ModuleType, source: str):
    if not hasattr(module, "config"):
        raise ValueError(
            f"Config source '{source}' does not define a 'config' object."
        )
    return _apply_config_defaults(module.config)


def load_config_by_name(config_name: str):
    if not config_name:
        raise ValueError("Config name is required.")

    module_candidates = _build_module_candidates(config_name)
    last_error = None
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            return _config_from_module(module, module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                last_error = exc
                continue
            raise

    searched = ", ".join(f"{name}.py" for name in module_candidates)
    raise ValueError(
        f"Config '{config_name}' not found. Tried: {searched}"
    ) from last_error


def load_config_by_path(config_path: str):
    if not config_path:
        raise ValueError("Config path is required.")
    abs_path = os.path.abspath(config_path)
    if not os.path.isfile(abs_path):
        raise ValueError(f"Config file does not exist: {abs_path}")
    module_name = f"user_config_{abs(hash(abs_path))}"
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import config from path: {abs_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _config_from_module(module, abs_path)
