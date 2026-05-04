#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate side-by-side montage pages for top improved CDD samples."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "paper_eval_visualizations/cdd_ours_vs_original_paper_qualitative_comparison.csv"
        ),
        help="Comparison CSV path.",
    )
    parser.add_argument(
        "--orig-dir",
        type=Path,
        default=Path("paper_eval_visualizations/config_cdd__cdd__test/paper_qualitative"),
        help="Original model paper_qualitative directory.",
    )
    parser.add_argument(
        "--ours-dir",
        type=Path,
        default=Path(
            "paper_eval_visualizations/config_cdd_dinov3_v5_convnext_Mask2Former_paperfinetune_bs8__epoch-140__test/paper_qualitative"
        ),
        help="Our model paper_qualitative directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper_eval_visualizations/cdd_ours_vs_original_montage_top_improved"),
        help="Output directory for montage pages.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top improved samples to visualize.",
    )
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=8,
        help="Rows (samples) per montage page.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="delta_iou",
        choices=["delta_iou", "delta_f1", "delta_err"],
        help="Sort key for ranking improved samples.",
    )
    parser.add_argument(
        "--a-dir",
        type=Path,
        default=Path("datasets/CDD-256/A"),
        help="Directory of pre-change (A) images.",
    )
    parser.add_argument(
        "--b-dir",
        type=Path,
        default=Path("datasets/CDD-256/B"),
        help="Directory of post-change (B) images.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("datasets/CDD-256/gt"),
        help="Directory of ground-truth (gt) masks.",
    )
    parser.add_argument(
        "--abgt-ext",
        type=str,
        default=".jpg",
        help="File extension for A/B/gt images (default: .jpg).",
    )
    return parser.parse_args()


def load_top_rows(csv_path: Path, top_n: int, sort_by: str) -> List[Dict[str, str]]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    improved = [r for r in rows if float(r["delta_iou"]) > 0]
    improved.sort(key=lambda r: float(r[sort_by]), reverse=True)
    return improved[:top_n]


def build_page(
    rows: List[Dict[str, str]],
    a_dir: Path,
    b_dir: Path,
    gt_dir: Path,
    abgt_ext: str,
    orig_dir: Path,
    ours_dir: Path,
    page_index: int,
    rows_per_page: int,
    out_dir: Path,
) -> None:
    first_img_name = rows[0]["image"]
    sample = Image.open(orig_dir / first_img_name).convert("RGB")
    w, h = sample.size
    sample.close()

    font = ImageFont.load_default()
    margin = 20
    row_gap = 16
    title_h = 36
    label_h = 18
    col_gap = 14

    panel_labels = ["A (pre)", "B (post)", "GT", "Original", "Ours"]
    n_cols = len(panel_labels)
    canvas_w = margin * 2 + w * n_cols + col_gap * (n_cols - 1)
    row_h = label_h + h
    canvas_h = margin * 2 + title_h + rows_per_page * row_h + (rows_per_page - 1) * row_gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    title = f"CDD Top Improved Samples: Page {page_index}"
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    draw.text(
        (margin, margin + 16),
        "Columns: A (pre) | B (post) | GT | Original | Ours",
        fill=(40, 40, 40),
        font=font,
    )

    y = margin + title_h
    for row in rows:
        img_name = row["image"]
        delta_iou = float(row["delta_iou"])
        delta_f1 = float(row["delta_f1"])
        delta_err = int(float(row["delta_err"]))

        orig_path = orig_dir / img_name
        ours_path = ours_dir / img_name
        base_name = Path(img_name).stem
        a_path = a_dir / f"{base_name}{abgt_ext}"
        b_path = b_dir / f"{base_name}{abgt_ext}"
        gt_path = gt_dir / f"{base_name}{abgt_ext}"
        if not (orig_path.exists() and ours_path.exists() and a_path.exists() and b_path.exists() and gt_path.exists()):
            continue

        a_img = Image.open(a_path).convert("RGB")
        b_img = Image.open(b_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        orig_img = Image.open(orig_path).convert("RGB")
        ours_img = Image.open(ours_path).convert("RGB")

        caption = (
            f"{img_name} | dIoU={delta_iou:+.4f} dF1={delta_f1:+.4f} "
            f"err_reduction={delta_err:+d}"
        )
        draw.text((margin, y), caption, fill=(0, 0, 0), font=font)

        img_y = y + label_h
        panels = [a_img, b_img, gt_img, orig_img, ours_img]
        for col_idx, (panel, panel_name) in enumerate(zip(panels, panel_labels)):
            x = margin + col_idx * (w + col_gap)
            canvas.paste(panel, (x, img_y))
            draw.rectangle((x - 1, img_y - 1, x + w, img_y + h), outline=(80, 80, 80), width=1)
            draw.text((x + 4, img_y + 4), panel_name, fill=(255, 255, 0), font=font)

        a_img.close()
        b_img.close()
        gt_img.close()
        orig_img.close()
        ours_img.close()
        y += row_h + row_gap

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"montage_page_{page_index:03d}.png"
    canvas.save(out_path)


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.orig_dir.exists():
        raise FileNotFoundError(f"Original dir not found: {args.orig_dir}")
    if not args.ours_dir.exists():
        raise FileNotFoundError(f"Ours dir not found: {args.ours_dir}")
    if not args.a_dir.exists():
        raise FileNotFoundError(f"A dir not found: {args.a_dir}")
    if not args.b_dir.exists():
        raise FileNotFoundError(f"B dir not found: {args.b_dir}")
    if not args.gt_dir.exists():
        raise FileNotFoundError(f"GT dir not found: {args.gt_dir}")

    rows = load_top_rows(args.csv, args.top_n, args.sort_by)
    if not rows:
        raise RuntimeError("No improved samples found in CSV (delta_iou > 0).")

    pages = math.ceil(len(rows) / args.rows_per_page)
    for i in range(pages):
        s = i * args.rows_per_page
        e = min((i + 1) * args.rows_per_page, len(rows))
        build_page(
            rows=rows[s:e],
            a_dir=args.a_dir,
            b_dir=args.b_dir,
            gt_dir=args.gt_dir,
            abgt_ext=args.abgt_ext,
            orig_dir=args.orig_dir,
            ours_dir=args.ours_dir,
            page_index=i + 1,
            rows_per_page=args.rows_per_page,
            out_dir=args.out_dir,
        )

    summary_path = args.out_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"top_n={len(rows)}\n")
        f.write(f"rows_per_page={args.rows_per_page}\n")
        f.write(f"sort_by={args.sort_by}\n")
        f.write(f"pages={pages}\n")
        f.write(f"csv={args.csv}\n")
        f.write(f"a_dir={args.a_dir}\n")
        f.write(f"b_dir={args.b_dir}\n")
        f.write(f"gt_dir={args.gt_dir}\n")
        f.write(f"abgt_ext={args.abgt_ext}\n")
        f.write(f"orig_dir={args.orig_dir}\n")
        f.write(f"ours_dir={args.ours_dir}\n")

    print(f"Generated {pages} montage page(s) under: {args.out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
