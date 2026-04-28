import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mindpilot_paths import get_env_value, get_project_root, load_mindpilot_env

load_mindpilot_env()

PROJECT_ROOT = get_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from ATMS_retrieval import ATMS, get_eeg_features
from util import get_gteeg, save_eeg

import logging

logging.getLogger("diffusers").setLevel(logging.WARNING)


def _load_tensor(path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    if isinstance(data, dict):
        for key in ("img_features", "features", "embed"):
            if key in data:
                data = data[key]
                break
        else:
            data = data[next(iter(data.keys()))]
    return data


def reward_function(eeg_model, image_path, encoder_model_path, yhat, sub, dnn, device):
    """Generate EEG signals for an image and compute similarity to the target EEG feature."""
    eeg_signal = torch.tensor(get_gteeg(image_path, encoder_model_path, dnn, device))
    eeg_feature = get_eeg_features(eeg_model, eeg_signal.unsqueeze(0), device, sub)
    similarity = torch.nn.functional.cosine_similarity(eeg_feature, yhat)
    similarity = (similarity + 1) / 2
    return similarity.item(), eeg_feature


def reward_function_clip_embed(embed1, embed2):
    cosine_sim = F.cosine_similarity(embed1, embed2, dim=1)
    normalized_sim = (cosine_sim + 1) / 2
    return normalized_sim.item()


def build_output_dirs(run_name):
    output_root = get_env_value(
        "EEG_LATENT_OUTPUT_DIR",
        default=os.path.join(str(PROJECT_ROOT), "benchmark_results", "exp_heuristic_generation_image_latent"),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_name, timestamp)
    gt_eeg_dir = os.path.join(run_dir, "syn_eeg_gt")
    pseudo_data_dir = os.path.join(run_dir, "pseudo_train_data")
    clip_embed_dir = os.path.join(run_dir, "clip_embed")
    plots_dir = os.path.join(run_dir, "plots")
    for folder in (gt_eeg_dir, pseudo_data_dir, clip_embed_dir, plots_dir):
        os.makedirs(folder, exist_ok=True)
    return run_dir, gt_eeg_dir, pseudo_data_dir, clip_embed_dir, plots_dir


def _parse_path_list_env(name, default=None):
    raw_value = get_env_value(name, default=None)
    if raw_value is None or raw_value.strip() == "":
        return default if default is not None else []

    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return values


def _resolve_path(path, default):
    if path:
        return path
    return default


def _load_image_embed_path(path, device):
    data = _load_tensor(path, device)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().view(-1, 1024)
    return torch.as_tensor(data).detach().cpu().view(-1, 1024)


def main():
    proxy = get_env_value("MINDPILOT_PROXY", default=get_env_value("HTTP_PROXY", default=None))
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(get_env_value("EEG_LATENT_SEED", default="4"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sub = get_env_value("EEG_LATENT_SUBJECT", default="sub-01")
    dnn = get_env_value("EEG_LATENT_DNN", default="alexnet")

    category_root = get_env_value(
        "EEG_LATENT_CATEGORY_ROOT",
        default=os.path.join(str(PROJECT_ROOT), "data", "things-eeg2", "test_images_flat"),
    )
    image_select_root = get_env_value(
        "EEG_LATENT_IMAGE_SELECT_ROOT",
        default=os.path.join(str(PROJECT_ROOT), "data", "things-eeg2", "image_select"),
    )
    image_pool_path = get_env_value(
        "EEG_LATENT_IMAGE_POOL_PATH",
        default=os.path.join(str(PROJECT_ROOT), "data", "clip_embed", "open_clip", "image50_embeddings.pt"),
    )
    target_image_path = get_env_value("EEG_LATENT_TARGET_IMAGE_PATH", default=None)
    target_image_paths = _parse_path_list_env("EEG_LATENT_TARGET_IMAGE_PATHS", default=[])
    target_image_embed_path = get_env_value("EEG_LATENT_TARGET_IMAGE_EMBED_PATH", default=None)
    encoder_model_path = get_env_value(
        "EEG_LATENT_ENCODER_MODEL_PATH",
        default=os.path.join(
            str(PROJECT_ROOT),
            "data",
            "encoding_models_checkpoints",
            sub,
            "synthetic_eeg_data",
            "encoding-end_to_end",
            f"dnn-{dnn}",
            "modeled_time_points-all",
            "pretrained-True",
            "lr-1e-05__wd-0e+00__bs-064",
            "model_state_dict.pt",
        ),
    )
    eeg2clip_model_path = get_env_value(
        "EEG_LATENT_EEG2CLIP_MODEL_PATH",
        default=os.path.join(
            str(PROJECT_ROOT),
            "data",
            "sub_model",
            sub,
            "diffusion_alexnet",
            "pretrained_True",
            "gene_gene",
            "ATM_S_reconstruction_scale_0_1000_40.pth",
        ),
    )
    eeg_features_path = get_env_value(
        "EEG_LATENT_EEG_FEATURES_PATH",
        default=os.path.join(
            str(PROJECT_ROOT),
            "data",
            "EEG_feature",
            "sub_features",
            sub,
            "eeg_features_ATMS_test.pt",
        ),
    )
    image_stride = int(get_env_value("EEG_LATENT_IMAGE_STRIDE", default="12"))
    max_iterations = int(get_env_value("EEG_LATENT_MAX_ITERATIONS", default="50"))
    num_images = int(get_env_value("EEG_LATENT_NUM_IMAGES", default="10"))

    if target_image_path is None:
        if not target_image_paths:
            raise ValueError(
                "Set EEG_LATENT_TARGET_IMAGE_PATH or EEG_LATENT_TARGET_IMAGE_PATHS in mindpilot.env"
            )
        target_index = int(get_env_value("EEG_LATENT_TARGET_INDEX", default="0"))
        if target_index >= len(target_image_paths):
            raise ValueError(
                f"EEG_LATENT_TARGET_INDEX={target_index} exceeds EEG_LATENT_TARGET_IMAGE_PATHS ({len(target_image_paths)})"
            )
        target_image_path = target_image_paths[target_index]
    else:
        target_index = 0

    if target_image_embed_path is None:
        target_stem = os.path.splitext(os.path.basename(target_image_path))[0]
        target_image_embed_path = os.path.join(
            str(PROJECT_ROOT),
            "data",
            "clip_embed",
            "open_clip",
            f"{target_stem}_image_embeds.pt",
        )

    text_list = sorted([name for name in os.listdir(category_root) if not name.startswith(".")])
    dir_name = os.path.basename(os.path.dirname(target_image_path))
    if dir_name in text_list:
        gt_category_id = text_list.index(dir_name)
    else:
        gt_category_id = -1

    run_dir, gt_eeg_dir, pseudo_data_dir, clip_embed_dir, plots_dir = build_output_dirs(dir_name)
    print(f"Output directory: {run_dir}")
    print(f"Target image: {target_image_path}")
    print(f"Target image embed: {target_image_embed_path}")
    print(f"Image pool: {image_pool_path}")

    image_pool = _load_image_embed_path(image_pool_path, device)
    target_image_embed = _load_tensor(target_image_embed_path, device)
    if isinstance(target_image_embed, dict):
        target_image_embed = _load_tensor(target_image_embed_path, device)
    target_image_embed = torch.as_tensor(target_image_embed).detach().to(device)
    if target_image_embed.dim() == 1:
        target_image_embed = target_image_embed.unsqueeze(0)

    checkpoint = torch.load(eeg2clip_model_path, map_location=device, weights_only=False)
    eeg_model = ATMS()
    eeg_model.load_state_dict(checkpoint["eeg_model_state_dict"])

    gt_eeg_path = os.path.join(gt_eeg_dir, f"gt_{os.path.splitext(os.path.basename(target_image_path))[0]}.npy")
    try:
        synthetic_eeg = torch.from_numpy(np.load(gt_eeg_path))
    except Exception:
        synthetic_eeg = torch.tensor(get_gteeg(target_image_path, encoder_model_path, dnn, device))
    gt_eeg_path = save_eeg(
        synthetic_eeg,
        gt_eeg_dir,
        file_name=f"gt_{os.path.splitext(os.path.basename(target_image_path))[0]}.npy",
    )

    tar_eeg_features = get_eeg_features(eeg_model, synthetic_eeg.unsqueeze(0), device, sub)

    data_x_list = []
    data_y_list = []
    sample_image_paths = []

    for idx, img_embed in enumerate(image_pool[::image_stride]):
        similarity = reward_function_clip_embed(img_embed.to(device).unsqueeze(0), target_image_embed)
        data_x_list.append(img_embed)
        data_y_list.append(-similarity * 100)
        sample_image_paths.append(f"image_pool_idx_{idx}")

    data = {
        "data_x": torch.stack(data_x_list),
        "data_y": torch.tensor(data_y_list),
    }

    pseudo_data_path = os.path.join(pseudo_data_dir, f"{dir_name}_data_scaling.pth")
    torch.save(data, pseudo_data_path)

    clip_embed_path = os.path.join(clip_embed_dir, f"{dir_name}_eeg_embeds.pt")
    torch.save(tar_eeg_features, clip_embed_path)

    summary = {
        "run_dir": run_dir,
        "seed": seed,
        "subject": sub,
        "dnn": dnn,
        "category_root": category_root,
        "image_select_root": image_select_root,
        "image_pool_path": image_pool_path,
        "target_image_path": target_image_path,
        "target_image_embed_path": target_image_embed_path,
        "eeg_features_path": eeg_features_path,
        "encoder_model_path": encoder_model_path,
        "eeg2clip_model_path": eeg2clip_model_path,
        "image_stride": image_stride,
        "max_iterations": max_iterations,
        "num_images": num_images,
        "gt_category_id": gt_category_id,
        "gt_eeg_path": gt_eeg_path,
        "pseudo_data_path": pseudo_data_path,
        "clip_embed_path": clip_embed_path,
        "num_data_points": len(data_x_list),
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved pseudo-train data to: {pseudo_data_path}")
    print(f"Saved EEG features to: {clip_embed_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
