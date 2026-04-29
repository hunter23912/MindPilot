#!/usr/bin/env python3
"""从 CSV 中选出 top5 的 EEG 和 CLIP targets，提取对应方法目录中的生成图并拼接成结果图。

默认行为：
- 对每个 target，选出其具有最大 EEG_Score 的 image 作为该 target 的 EEG 分数代表；按该分数取前 5 个 target。
- 对每个 target，选出其具有最大 CLIP_Score 的 image 作为该 target 的 CLIP 分数代表；按该分数取前 5 个 target。
- 从 `eeg_guidance/target_{idx}_seed_0/final/final_generated_{image_idx}.png` 和
  `target_image_guidance/target_{idx}_seed_0/final/final_generated_{image_idx}.png` 中提取图片。

输出：保存为 `top5_collage_eeg_clip.png`（可通过 --out 指定）。
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import pandas as pd


def find_top5(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure numeric
    df['EEG_Score'] = pd.to_numeric(df['EEG_Score'], errors='coerce')
    df['CLIP_Score'] = pd.to_numeric(df['CLIP_Score'], errors='coerce')

    # For each target, get the image with max EEG_Score (record its Image_Idx)
    eeg_best = df.loc[df.groupby('Target_Idx')['EEG_Score'].idxmax()].set_index('Target_Idx')
    clip_best = df.loc[df.groupby('Target_Idx')['CLIP_Score'].idxmax()].set_index('Target_Idx')

    top5_eeg = eeg_best.sort_values('EEG_Score', ascending=False).head(5)
    top5_clip = clip_best.sort_values('CLIP_Score', ascending=False).head(5)

    eeg_list = [(int(idx), int(row['Image_Idx'])) for idx, row in top5_eeg.iterrows()]
    clip_list = [(int(idx), int(row['Image_Idx'])) for idx, row in top5_clip.iterrows()]
    return eeg_list, clip_list


def load_image_for(target_idx, image_idx, method_dir):
    # expected path pattern
    subdir = f"target_{target_idx}_seed_0"
    final_dir = Path(method_dir) / subdir / 'final'
    candidate = final_dir / f'final_generated_{image_idx}.png'
    if candidate.exists():
        return Image.open(candidate).convert('RGB')
    # fallback: try loop image names
    fallback_dir = Path(method_dir) / subdir
    for name in ['loop1.png', 'loop2.png', 'loop3.png', 'loop4.png', 'loop5.png']:
        p = fallback_dir / name
        if p.exists():
            return Image.open(p).convert('RGB')
    raise FileNotFoundError(f"No image found for target {target_idx} in {method_dir}")


def make_collage(eeg_items, clip_items, eeg_dir, clip_dir, out_path, img_size=(256,256), include_gt_dir=None):
    # rows: optional GT, CLIP (target_image_guidance), EEG (eeg_guidance)
    rows = []
    headers = []
    cols = max(len(eeg_items), len(clip_items))

    # load clip images (row 0)
    clip_imgs = []
    for idx, img_i in clip_items:
        try:
            img = load_image_for(idx, img_i, clip_dir)
        except FileNotFoundError:
            img = Image.new('RGB', img_size, (50,50,50))
        clip_imgs.append(img.resize(img_size))
    rows.append(clip_imgs)
    headers.append('Image')

    # load eeg images (row 1)
    eeg_imgs = []
    for idx, img_i in eeg_items:
        try:
            img = load_image_for(idx, img_i, eeg_dir)
        except FileNotFoundError:
            img = Image.new('RGB', img_size, (80,80,80))
        eeg_imgs.append(img.resize(img_size))
    rows.append(eeg_imgs)
    headers.append('EEG')

    # If counts differ, pad with blank images
    for r in rows:
        while len(r) < cols:
            r.append(Image.new('RGB', img_size, (0,0,0)))

    # canvas
    padding = 8
    w = cols * img_size[0] + (cols+1) * padding
    h = len(rows) * img_size[1] + (len(rows)+1) * padding
    canvas = Image.new('RGB', (w, h), (30,30,30))

    # paste images
    for ri, row in enumerate(rows):
        for ci, img in enumerate(row):
            x = padding + ci * (img_size[0] + padding)
            y = padding + ri * (img_size[1] + padding)
            canvas.paste(img, (x, y))

    canvas.save(out_path)
    print(f"Saved collage to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='detailed_results_20260428_190754.csv')
    p.add_argument('--eeg-dir', default='eeg_guidance')
    p.add_argument('--clip-dir', default='target_image_guidance')
    p.add_argument('--out', default='top5_collage_eeg_clip.png')
    p.add_argument('-n', type=int, default=5)
    args = p.parse_args()

    csv_path = Path(args.csv)
    base_dir = Path(csv_path.parent)
    eeg_dir = Path(args.eeg_dir)
    clip_dir = Path(args.clip_dir)
    # if relative paths, resolve relative to CSV folder
    if not eeg_dir.is_absolute():
        eeg_dir = base_dir / eeg_dir
    if not clip_dir.is_absolute():
        clip_dir = base_dir / clip_dir

    eeg_items, clip_items = find_top5(csv_path)
    # respect -n if user wants fewer
    eeg_items = eeg_items[:args.n]
    clip_items = clip_items[:args.n]

    out_path = Path(args.out)
    # default output in same folder as csv
    if not out_path.is_absolute():
        out_path = base_dir / out_path

    make_collage(eeg_items, clip_items, str(eeg_dir), str(clip_dir), str(out_path))


if __name__ == '__main__':
    main()
