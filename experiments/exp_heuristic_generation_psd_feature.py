import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from mne.time_frequency import psd_array_multitaper

from mindpilot_paths import get_env_value, get_project_root, load_mindpilot_env

load_mindpilot_env()

PROJECT_ROOT = get_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "model"))

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


def _parse_path_list_env(name, default=None):
    raw_value = get_env_value(name, default=None)
    if raw_value is None or raw_value.strip() == "":
        return default if default is not None else []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_int_list_env(name, default=None):
    raw_value = get_env_value(name, default=None)
    if raw_value is None or raw_value.strip() == "":
        return default if default is not None else []
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def _first_image_in_folder(folder_path):
    image_files = [file_name for file_name in sorted(os.listdir(folder_path)) if file_name.lower().endswith((".jpg", ".png", ".jpeg")) and not file_name.startswith("._")]
    if not image_files:
        raise ValueError(f"No image files found in folder: {folder_path}")
    return os.path.join(folder_path, image_files[0])


def _load_psd_from_eeg(target_signal, fs, selected_channel_idxes):
    selected_target_signal = target_signal[selected_channel_idxes, :]
    target_psd, _ = psd_array_multitaper(selected_target_signal, fs, adaptive=True, normalization='full', verbose=0)
    return torch.from_numpy(target_psd.flatten()).unsqueeze(0)


def reward_function(psd, target_psd):
    return F.cosine_similarity(target_psd, psd).item()


def build_output_dirs(run_name):
    output_root = get_env_value(
        "PSD_FEATURE_OUTPUT_DIR",
        default=os.path.join(str(PROJECT_ROOT), "benchmark_results", "exp_heuristic_generation_psd_feature"),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_name, timestamp)
    gt_eeg_dir = os.path.join(run_dir, "syn_eeg_gt")
    psd_dir = os.path.join(run_dir, "psd_feature")
    pseudo_train_dir = os.path.join(run_dir, "pseudo_train_data")
    plots_dir = os.path.join(run_dir, "plots")
    for folder in (gt_eeg_dir, psd_dir, pseudo_train_dir, plots_dir):
        os.makedirs(folder, exist_ok=True)
    return run_dir, gt_eeg_dir, psd_dir, pseudo_train_dir, plots_dir


def main():
    proxy = get_env_value("MINDPILOT_PROXY", default=get_env_value("HTTP_PROXY", default=None))
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy

    cuda_visible_devices = get_env_value("PSD_FEATURE_CUDA_VISIBLE_DEVICES", default=None)
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(get_env_value("PSD_FEATURE_SEED", default="4"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    image_folder = get_env_value(
        "PSD_FEATURE_IMAGE_FOLDER",
        default=os.path.join(str(PROJECT_ROOT), "data", "things-eeg2", "test_images_flat"),
    )
    image_select_root = get_env_value(
        "PSD_FEATURE_IMAGE_SELECT_ROOT",
        default=os.path.join(str(PROJECT_ROOT), "data", "things-eeg2", "image_select"),
    )
    image_pool_path = get_env_value(
        "PSD_FEATURE_IMAGE_POOL_PATH",
        default=os.path.join(str(PROJECT_ROOT), "data", "clip_embed", "open_clip", "image50_embeddings.pt"),
    )
    target_image_path = get_env_value("PSD_FEATURE_TARGET_IMAGE_PATH", default=None)
    target_image_paths = _parse_path_list_env("PSD_FEATURE_TARGET_IMAGE_PATHS", default=[])
    target_index = int(get_env_value("PSD_FEATURE_TARGET_INDEX", default="0"))
    encoder_model_path = get_env_value(
        "PSD_FEATURE_ENCODER_MODEL_PATH",
        default=os.path.join(
            str(PROJECT_ROOT),
            "data",
            "encoding_models_checkpoints",
            "sub-01",
            "synthetic_eeg_data",
            "encoding-end_to_end",
            "dnn-alexnet",
            "modeled_time_points-all",
            "pretrained-True",
            "lr-1e-05__wd-0e+00__bs-064",
            "model_state_dict.pt",
        ),
    )
    eeg2clip_model_path = get_env_value(
        "PSD_FEATURE_EEG2CLIP_MODEL_PATH",
        default=os.path.join(
            str(PROJECT_ROOT),
            "data",
            "sub_model",
            "sub-01",
            "diffusion_alexnet",
            "pretrained_True",
            "gene_gene",
            "ATM_S_reconstruction_scale_0_1000_40.pth",
        ),
    )
    eeg_features_path = get_env_value(
        "PSD_FEATURE_EEG_FEATURES_PATH",
        default=os.path.join(
            str(PROJECT_ROOT),
            "data",
            "EEG_feature",
            "sub_features",
            "sub-01",
            "eeg_features_ATMS_test.pt",
        ),
    )
    fs = int(get_env_value("PSD_FEATURE_FS", default="250"))
    selected_channel_idxes = _parse_int_list_env("PSD_FEATURE_SELECTED_CHANNEL_IDXS", default=[3, 4, 5])
    selected_channel_idxes = np.array(selected_channel_idxes, dtype=int)

    text_list = [name for name in sorted(os.listdir(image_folder)) if not name.startswith(".")]
    if target_image_path is None:
        if not target_image_paths:
            raise ValueError(
                "Set PSD_FEATURE_TARGET_IMAGE_PATH or PSD_FEATURE_TARGET_IMAGE_PATHS in mindpilot.env"
            )
        if target_index >= len(target_image_paths):
            raise ValueError(
                f"PSD_FEATURE_TARGET_INDEX={target_index} exceeds PSD_FEATURE_TARGET_IMAGE_PATHS ({len(target_image_paths)})"
            )
        target_image_path = target_image_paths[target_index]

    dir_name = os.path.basename(os.path.dirname(target_image_path))
    if dir_name not in text_list:
        raise ValueError(f"Target category '{dir_name}' not found in image folder '{image_folder}'")

    run_dir, gt_eeg_dir, psd_dir, pseudo_train_dir, plots_dir = build_output_dirs(dir_name)
    print(f"Output directory: {run_dir}")
    print(f"Target image: {target_image_path}")
    print(f"Image pool: {image_pool_path}")

    image_pool = _load_tensor(image_pool_path, device)
    if isinstance(image_pool, torch.Tensor):
        image_pool = image_pool.view(-1, 1024).detach().cpu()
    else:
        image_pool = torch.as_tensor(image_pool).view(-1, 1024).detach().cpu()

    checkpoint = torch.load(eeg2clip_model_path, map_location=device, weights_only=False)
    eeg_model = ATMS()
    eeg_model.load_state_dict(checkpoint["eeg_model_state_dict"])

    gt_eeg_cache = os.path.join(gt_eeg_dir, f"gt_{os.path.splitext(os.path.basename(target_image_path))[0]}.npy")
    try:
        synthetic_eeg = torch.from_numpy(np.load(gt_eeg_cache))
    except Exception:
        synthetic_eeg = torch.tensor(get_gteeg(target_image_path, encoder_model_path, "alexnet", device))
    gt_eeg_path = save_eeg(
        synthetic_eeg,
        gt_eeg_dir,
        file_name=f"gt_{os.path.splitext(os.path.basename(target_image_path))[0]}.npy",
    )

    target_psd_cache = os.path.join(psd_dir, f"{dir_name}_psd_o1o2oz.pt")
    if os.path.exists(target_psd_cache):
        target_psd = torch.load(target_psd_cache, weights_only=False).cpu()
    else:
        target_psd = _load_psd_from_eeg(synthetic_eeg.detach().cpu().numpy(), fs=fs, selected_channel_idxes=selected_channel_idxes)
        torch.save(target_psd, target_psd_cache)

    test_set_psd_cache = os.path.join(psd_dir, "test_set_psd_features_o1o2oz.pt")
    if os.path.exists(test_set_psd_cache):
        test_set_psd_features = torch.load(test_set_psd_cache, weights_only=False).cpu()
    else:
        test_set_psd_features = []
        for folder_name in text_list:
            folder_path = os.path.join(image_select_root, folder_name)
            image_path = _first_image_in_folder(folder_path)
            eeg_signal = get_gteeg(image_path, encoder_model_path, "alexnet", device)
            psd = _load_psd_from_eeg(eeg_signal.detach().cpu().numpy(), fs=fs, selected_channel_idxes=selected_channel_idxes)
            test_set_psd_features.append(psd)
        test_set_psd_features = torch.stack(test_set_psd_features)
        torch.save(test_set_psd_features, test_set_psd_cache)

    data_x_list = []
    data_y_list = []
    for idx, img_embed in enumerate(image_pool):
        similarity = reward_function(test_set_psd_features[idx], target_psd)
        data_x_list.append(img_embed)
        data_y_list.append(-similarity * 100)
        print(f"similarity {similarity}")

    data = {
        "data_x": torch.stack(data_x_list),
        "data_y": torch.tensor(data_y_list),
    }

    pseudo_train_path = os.path.join(pseudo_train_dir, f"{dir_name}_data_scaling.pth")
    torch.save(data, pseudo_train_path)

    eeg_feature_path = os.path.join(psd_dir, f"{dir_name}_eeg_features.pt")
    tar_eeg_features = get_eeg_features(eeg_model, synthetic_eeg.unsqueeze(0), device, "sub-01")
    torch.save(tar_eeg_features, eeg_feature_path)

    summary = {
        "run_dir": run_dir,
        "seed": seed,
        "image_folder": image_folder,
        "image_select_root": image_select_root,
        "image_pool_path": image_pool_path,
        "target_image_path": target_image_path,
        "encoder_model_path": encoder_model_path,
        "eeg2clip_model_path": eeg2clip_model_path,
        "eeg_features_path": eeg_features_path,
        "fs": fs,
        "selected_channel_idxes": selected_channel_idxes.tolist(),
        "gt_eeg_path": gt_eeg_path,
        "target_psd_cache": target_psd_cache,
        "test_set_psd_cache": test_set_psd_cache,
        "pseudo_train_path": pseudo_train_path,
        "eeg_feature_path": eeg_feature_path,
        "num_samples": len(data_x_list),
        "target_psd_shape": list(target_psd.shape),
        "test_set_psd_shape": list(test_set_psd_features.shape),
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved pseudo-train data to: {pseudo_train_path}")
    print(f"Saved target PSD cache to: {target_psd_cache}")
    print(f"Saved test PSD cache to: {test_set_psd_cache}")
    print(f"Saved EEG features to: {eeg_feature_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
