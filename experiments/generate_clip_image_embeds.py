#!/usr/bin/env python3
"""
Generate CLIP (ViT-H-14) image embeddings for all test images.

These embeddings serve as optimization targets in the MindPilot benchmark.
Each test image gets a corresponding _embed.pt file, aligned by sorted filename.

Usage:
    python experiments/generate_clip_image_embeds.py
"""

import os
import sys

# Ensure project root and model/ are on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "model"))

import torch
import open_clip
from PIL import Image
from tqdm import tqdm

from mindpilot_paths import get_env_value, get_project_root, load_mindpilot_env

# 确保 PROJECT_ROOT 已设置（bash source 或直接 python 调用均可）
if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = _project_root

load_mindpilot_env()

# --- Helpers ---------------------------------------------------------------
def rename_images_with_prefix(image_dir):
    """
    Rename test images to `{idx:03d}_{original_name}` so that sorted order is
    explicit and robust across filesystems.
    """
    allowed = (".jpg", ".jpeg", ".png")
    files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(allowed)]
    )
    for idx, fname in enumerate(files):
        src = os.path.join(image_dir, fname)
        dst = os.path.join(image_dir, f"{idx:03d}_{fname}")
        if src != dst:
            os.rename(src, dst)
    print(f"Renamed {len(files)} images in {image_dir}")


def generate_clip_embeds(image_dir, embed_dir, vlmodel, preprocess, device):
    """
    For each image in *image_dir* (sorted), compute its CLIP ViT-H-14 embedding
    and save to *embed_dir* as `{idx:03d}_{basename_without_ext}_embed.pt`.
    """
    allowed = (".jpg", ".jpeg", ".png")
    files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(allowed)]
    )

    os.makedirs(embed_dir, exist_ok=True)

    for idx, fname in enumerate(tqdm(files, desc="Generating CLIP embeds")):
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embed = vlmodel.encode_image(img_tensor)

        name_no_ext = os.path.splitext(fname)[0]
        out_path = os.path.join(embed_dir, f"{name_no_ext}_embed.pt")
        torch.save(embed.cpu(), out_path)

    print(f"Saved {len(files)} embeddings to {embed_dir}")


# --- Main ------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = str(get_project_root())

    # ---- image directory ----
    image_dir = get_env_value(
        "TEST_IMAGE_DIR",
        default=os.path.join(project_root, "data", "things-eeg2", "test_images_flat"),
    )
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # ---- embed output directory ----
    # These are CLIP image embeddings (ViT-H-14), NOT EEG features.
    # The benchmark code calls them "eeg_feature" because the ATMS model maps
    # EEG signals into CLIP space and uses these as optimization targets,
    # but the data itself is CLIP-encoded images.
    embed_dir = os.path.join(project_root, "data", "things-eeg2", "clip_image_embeds")

    # ---- Step 1: rename images with numeric prefix for robust ordering ----
    print("Step 1: Renaming images with numeric prefix...")
    rename_images_with_prefix(image_dir)
    exit(0)
    # ---- Step 2: load OpenCLIP ViT-H-14 (same as benchmark) ----
    print("Step 2: Loading OpenCLIP ViT-H-14...")
    model_type = "ViT-H-14"
    local_weights = get_env_value("OPEN_CLIP_WEIGHTS", default="")
    if local_weights and os.path.exists(local_weights):
        print(f"  Loading local weights: {local_weights}")
        vlmodel, preprocess, feature_extractor = open_clip.create_model_and_transforms(
            model_type, pretrained=None, precision="fp32", device=device
        )
        state = torch.load(local_weights, map_location=device, weights_only=True)
        vlmodel.load_state_dict(state)
    else:
        print("  Local weights not found, downloading from Hugging Face...")
        vlmodel, preprocess, feature_extractor = open_clip.create_model_and_transforms(
            model_type, pretrained="laion2b_s32b_b79k", precision="fp32", device=device
        )
    vlmodel.to(device)
    vlmodel.eval()

    # ---- Step 3: generate and save CLIP embeddings ----
    print("Step 3: Generating CLIP image embeddings...")
    generate_clip_embeds(image_dir, embed_dir, vlmodel, preprocess, device)

    # ---- Verification ----
    allowed = (".jpg", ".jpeg", ".png")
    img_count = len(
        [f for f in os.listdir(image_dir) if f.lower().endswith(allowed)]
    )
    embed_count = len(
        [f for f in os.listdir(embed_dir) if f.endswith("_embed.pt")]
    )
    print(f"\nVerification: {img_count} images, {embed_count} embeddings")
    if img_count == embed_count:
        print("  OK — counts match.")
    else:
        print("  WARNING — count mismatch!")

    print(f"\nEmbed directory: {embed_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
