#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator
from PIL import Image, ImageDraw, ImageFont


MODEL_EPOCH_RE = re.compile(r"epoch-(\d+)\.pth")
CLASS_LINE_RE = re.compile(r"^\s*\d+\s+([^\t\s]+)\s+([0-9.]+)%")


def f1_to_iou(f1_value: float) -> float:
    if f1_value <= 0.0:
        return 0.0
    return f1_value / (2.0 - f1_value)


def load_change_iou_from_tb(tb_paths: list[Path]) -> dict[int, float]:
    if not tb_paths:
        return {}

    series: dict[int, tuple[float, float]] = {}
    for tb_dir in sorted(tb_paths):
        acc = event_accumulator.EventAccumulator(str(tb_dir))
        try:
            acc.Reload()
        except Exception:
            continue
        tags = set(acc.Tags().get("scalars", []))

        if "val_iou_change" in tags:
            for e in acc.Scalars("val_iou_change"):
                series[e.step] = (e.wall_time, float(e.value))
            continue

        if "val_f1_change" in tags:
            for e in acc.Scalars("val_f1_change"):
                series[e.step] = (e.wall_time, f1_to_iou(float(e.value)))
            continue

        if "val_dice_change" in tags:
            for e in acc.Scalars("val_dice_change"):
                series[e.step] = (e.wall_time, f1_to_iou(float(e.value)))

    return {k: v for k, (_, v) in sorted(series.items())}


def load_change_iou_from_logs(run_dir: Path) -> dict[int, float]:
    series: dict[int, float] = {}
    log_files = sorted(run_dir.glob("val_*.log"))

    for log_file in log_files:
        current_epoch = None
        try:
            lines = log_file.read_text(errors="ignore").splitlines()
        except Exception:
            continue

        for line in lines:
            if line.startswith("Model:"):
                match = MODEL_EPOCH_RE.search(line)
                if match:
                    current_epoch = int(match.group(1))
                continue

            if current_epoch is None:
                continue

            match = CLASS_LINE_RE.match(line)
            if not match:
                continue

            class_name = match.group(1).strip().lower()
            if class_name != "change":
                continue
            iou = float(match.group(2)) / 100.0
            series[current_epoch] = iou

    return dict(sorted(series.items()))


def resolve_run_path(input_path: Path) -> tuple[Path, list[Path]]:
    # Accept either run dir (.../log_xxx) or specific tb dir (.../log_xxx/tb/<session>)
    if input_path.name == "tb" and input_path.is_dir():
        tb_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        return input_path.parent, tb_dirs

    if input_path.parent.name == "tb" and input_path.is_dir():
        return input_path.parent.parent, [input_path]

    tb_root = input_path / "tb"
    tb_dirs = [d for d in tb_root.iterdir() if d.is_dir()] if tb_root.exists() else []
    return input_path, tb_dirs


def load_run_series(path_input: Path) -> tuple[dict[int, float], str]:
    run_dir, tb_dirs = resolve_run_path(path_input)
    tb_series = load_change_iou_from_tb(tb_dirs)
    log_series = load_change_iou_from_logs(run_dir)

    if tb_series and log_series:
        merged = dict(log_series)
        merged.update(tb_series)
        return dict(sorted(merged.items())), "tb+log"
    if tb_series:
        return tb_series, "tb"
    if log_series:
        return log_series, "log"
    return {}, "none"


def parse_run_arg(run_arg: str) -> tuple[str, Path]:
    if ":" not in run_arg:
        raise ValueError(f"Invalid --run '{run_arg}'. Expected format LABEL:PATH")
    label, path = run_arg.split(":", 1)
    label = label.strip()
    run_dir = Path(path).expanduser().resolve()
    if not label:
        raise ValueError(f"Invalid --run '{run_arg}'. LABEL is empty")
    return label, run_dir


def save_plot_with_pillow(
    plot_series: list[tuple[str, list[int], list[float]]],
    output_path: Path,
    title: str,
    dataset_label: str,
    x_label: str,
    min_epoch: int,
    max_epoch: int,
    y_min: float,
    y_max: float,
):
    scale = 3  # draw large then downsample for smoother lines/text
    width, height = 1200 * scale, 760 * scale
    left, right, top, bottom = 150 * scale, 60 * scale, 120 * scale, 120 * scale
    plot_w = width - left - right
    plot_h = height - top - bottom

    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_to_px(epoch: int) -> int:
        ratio = 0.0 if max_epoch == min_epoch else (epoch - min_epoch) / (max_epoch - min_epoch)
        return int(left + ratio * plot_w)

    def y_to_px(value: float) -> int:
        ratio = (value - y_min) / (y_max - y_min)
        return int(top + (1.0 - ratio) * plot_h)

    def clip_segment_to_ymin(x1: float, y1: float, x2: float, y2: float, ymin: float):
        # Keep only the part of a segment with y >= ymin.
        if y1 >= ymin and y2 >= ymin:
            return (x1, y1), (x2, y2)
        if y1 < ymin and y2 < ymin:
            return None
        if y2 == y1:
            return None

        t = (ymin - y1) / (y2 - y1)
        xi = x1 + t * (x2 - x1)
        yi = ymin
        if y1 < ymin <= y2:
            return (xi, yi), (x2, y2)
        return (x1, y1), (xi, yi)

    bg = (255, 255, 255)
    axis = (40, 40, 40)
    grid = (185, 185, 185)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 62 * scale)
        font_axis = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 54 * scale)
        font_tick = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 46 * scale)
        font_legend = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 50 * scale)
        font_dataset = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 66 * scale)
    except Exception:
        font_title = ImageFont.load_default()
        font_axis = ImageFont.load_default()
        font_tick = ImageFont.load_default()
        font_legend = ImageFont.load_default()
        font_dataset = ImageFont.load_default()

    if title:
        draw.text((left, 12 * scale), title, fill=axis, font=font_title)

    if dataset_label:
        ds_bbox = draw.textbbox((0, 0), dataset_label, font=font_dataset)
        ds_w = ds_bbox[2] - ds_bbox[0]
        ds_h = ds_bbox[3] - ds_bbox[1]
        x_ds = left + (plot_w - ds_w) // 2
        draw.text((x_ds, top - ds_h - 20 * scale), dataset_label, fill=axis, font=font_dataset)

    # grid first
    y_ticks = [y_min + (y_max - y_min) * k / 5 for k in range(6)]
    y_tick_labels = [f"{val:.0f}" for val in y_ticks]
    max_y_tick_w = 0
    max_y_tick_h = 0
    for t in y_tick_labels:
        tb = draw.textbbox((0, 0), t, font=font_tick)
        max_y_tick_w = max(max_y_tick_w, tb[2] - tb[0])
        max_y_tick_h = max(max_y_tick_h, tb[3] - tb[1])

    y_tick_x = left - max_y_tick_w - 14 * scale
    for val in y_ticks:
        y = y_to_px(val)
        draw.line([(left, y), (left + plot_w, y)], fill=grid, width=1 * scale)
        label = f"{val:.0f}"
        draw.text((y_tick_x, y - max_y_tick_h // 2), label, fill=axis, font=font_tick)

    x_ticks = [int(min_epoch + (max_epoch - min_epoch) * k / 6) for k in range(7)]
    x_tick_h = 0
    for ep in x_ticks:
        x = x_to_px(ep)
        draw.line([(x, top), (x, top + plot_h)], fill=grid, width=1 * scale)
        ep_text = str(ep)
        tb = draw.textbbox((0, 0), ep_text, font=font_tick)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        x_tick_h = max(x_tick_h, th)
        draw.text((x - tw // 2, top + plot_h + 12 * scale), ep_text, fill=axis, font=font_tick)

    # axis with arrow heads
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=axis, width=3 * scale)
    draw.line([(left, top + plot_h), (left, top)], fill=axis, width=3 * scale)
    draw.polygon(
        [
            (left + plot_w, top + plot_h),
            (left + plot_w - 20 * scale, top + plot_h - 9 * scale),
            (left + plot_w - 20 * scale, top + plot_h + 9 * scale),
        ],
        fill=axis,
    )
    draw.polygon(
        [
            (left, top),
            (left - 9 * scale, top + 20 * scale),
            (left + 9 * scale, top + 20 * scale),
        ],
        fill=axis,
    )

    colors = [
        (31, 119, 180),
        (220, 70, 130),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
    ]

    legend_y = top + int(plot_h * 0.58)
    for idx, (label, epochs, values) in enumerate(plot_series):
        color = colors[idx % len(colors)]
        for i in range(len(epochs) - 1):
            x1, y1 = float(epochs[i]), float(values[i])
            x2, y2 = float(epochs[i + 1]), float(values[i + 1])
            clipped = clip_segment_to_ymin(x1, y1, x2, y2, y_min)
            if clipped is None:
                continue
            (cx1, cy1), (cx2, cy2) = clipped
            draw.line(
                [(x_to_px(cx1), y_to_px(cy1)), (x_to_px(cx2), y_to_px(cy2))],
                fill=color,
                width=5 * scale,
            )

        lx = left + int(plot_w * 0.58)
        lb = draw.textbbox((0, 0), label, font=font_legend)
        lh = lb[3] - lb[1]
        draw.line([(lx, legend_y + lh // 2), (lx + 70 * scale, legend_y + lh // 2)], fill=color, width=5 * scale)
        draw.text((lx + 84 * scale, legend_y), label, fill=axis, font=font_legend)
        legend_y += lh + 12 * scale

    xlabel = x_label
    xb = draw.textbbox((0, 0), xlabel, font=font_axis)
    xw = xb[2] - xb[0]
    draw.text((left + (plot_w - xw) // 2, top + plot_h + x_tick_h + 22 * scale), xlabel, fill=axis, font=font_axis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = img.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot validation change-class IoU vs epoch for multiple runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run in LABEL:PATH format (PATH can be run dir, tb dir, or tb root). Repeat for comparisons.",
    )
    parser.add_argument("--output", required=True, help="Output plot path (.png/.pdf)")
    parser.add_argument("--title", default="Validation Change IoU vs Epoch")
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--x-label", default="Training iterations")
    parser.add_argument("--min-epoch", type=int, default=0)
    parser.add_argument("--max-epoch", type=int, default=150)
    parser.add_argument("--y-min", type=float, default=80.0)
    parser.add_argument("--y-max", type=float, default=100.0)
    args = parser.parse_args()

    run_specs = [parse_run_arg(x) for x in args.run]

    plotted = 0
    plot_series: list[tuple[str, list[int], list[float]]] = []

    for label, run_dir in run_specs:
        series, _source = load_run_series(run_dir)
        if not series:
            print(f"[WARN] No change IoU found for '{label}' in {run_dir}")
            continue

        epochs = [e for e in series.keys() if args.min_epoch <= e <= args.max_epoch]
        if not epochs:
            print(
                f"[WARN] '{label}' has data, but nothing in epoch range "
                f"[{args.min_epoch}, {args.max_epoch}]"
            )
            continue

        values = [series[e] * 100.0 for e in epochs]
        plot_series.append((label, epochs, values))
        plotted += 1

    if plotted == 0:
        raise SystemExit("No plottable series found.")

    output_path = Path(args.output)
    save_plot_with_pillow(
        plot_series=plot_series,
        output_path=output_path,
        title=args.title,
        dataset_label=args.dataset_label,
        x_label=args.x_label,
        min_epoch=args.min_epoch,
        max_epoch=args.max_epoch,
        y_min=args.y_min,
        y_max=args.y_max,
    )
    print(f"Saved plot: {output_path.resolve()}")


if __name__ == "__main__":
    main()
