import json
import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mindpilot_paths import get_env_value, get_project_root, load_mindpilot_env

load_mindpilot_env()

PROJECT_ROOT = get_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from ATMS_retrieval import ATMS, get_eeg_features
from util import (
    get_gteeg,
    plot_similarity_and_mse_with_dual_axis,
    visualize_top_images,
)

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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_path_list_env(name, default=None):
    raw_value = get_env_value(name, default=None)
    if raw_value is None or raw_value.strip() == "":
        return default if default is not None else []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def get_image_path(category_idx, image_idx, text_list, image_select_root):
    category_folder = text_list[category_idx]
    folder_path = os.path.join(image_select_root, category_folder)
    image_file = [f for f in sorted(os.listdir(folder_path)) if f.endswith((".jpg", ".png", ".jpeg")) and not f.startswith("._")]
    if not image_file:
        raise ValueError(f"No images found in category folder: {folder_path}")
    return os.path.join(folder_path, image_file[image_idx])


def initial_policy(image_pool, num_images, history=None):
    total_images = image_pool.view(-1, 1024)
    if history is None:
        history = []
    available_indices = [i for i in range(len(total_images)) if i not in history]
    indices = np.random.choice(available_indices, num_images, replace=False)
    category_indices = indices // 12
    image_indices = indices % 12
    return list(zip(category_indices, image_indices))


def compute_embed_similarity(img_feature, all_features):
    dot_product = torch.matmul(all_features, img_feature)
    return dot_product


def reward_function(image_path, encoder_model_path, yhat, device, sub, dnn):
    eeg_signal = torch.tensor(get_gteeg(image_path, encoder_model_path, dnn, device))
    eeg_feature = get_eeg_features(eeg_model, eeg_signal.unsqueeze(0), device, sub)
    similarity = torch.nn.functional.cosine_similarity(eeg_feature, yhat)
    similarity = (similarity + 1) / 2
    return similarity.item(), eeg_feature


def policy_evaluation(policy, image_pool, visited_images, similarities, encoder_model_path, text_list, probabilities, historical_max_similarity, gamma, history=None):
    alpha_base = 0.1
    beta_base = 0.05
    epsilon = 1e-6

    if len(similarities) > 0:
        min_sim = min(similarities)
        max_sim = max(similarities)
        norm_similarities = [2 * ((s - min_sim) / (max_sim - min_sim + epsilon)) - 1 for s in similarities]
    else:
        norm_similarities = []

    for (cat, idx), similarity, norm_sim in zip(visited_images, similarities, norm_similarities):
        img_feature = image_pool[cat * 12 + idx]

        alpha = alpha_base * (0.5 + 0.5 * abs(norm_sim)) * (1 if norm_sim > 0 else -1)
        beta = beta_base * (0.5 + 0.5 * abs(norm_sim)) * (1 if norm_sim > 0 else -1)

        if norm_sim > 0:
            probabilities[cat * 12 + idx] = (1 - abs(alpha)) * probabilities[cat * 12 + idx] + abs(alpha)
        else:
            probabilities[cat * 12 + idx] = (1 - abs(alpha)) * probabilities[cat * 12 + idx]

        cosine_similarities = compute_embed_similarity(img_feature, image_pool)

        for i in range(len(image_pool)):
            if i != cat * 12 + idx:
                reward_to_spread = abs(beta) * cosine_similarities[i].item() * abs(norm_sim)
                if norm_sim > 0:
                    probabilities[i] = (1 - reward_to_spread) * probabilities[i] + reward_to_spread
                else:
                    probabilities[i] = (1 - reward_to_spread) * probabilities[i]

    probabilities = np.maximum(probabilities, epsilon)
    probabilities = probabilities / (np.sum(probabilities) + epsilon * len(image_pool))
    return probabilities


def policy_improvement(value_function, image_pool, iteration=None, save_folder=None, num_images=10, temperature=1.0):
    from scipy.special import softmax

    def improved_policy(image_pool, num_images, history=None):
        if history:
            available_indices = [i for i in range(len(image_pool)) if i not in history]
        else:
            available_indices = np.arange(len(image_pool))

        scaled_values = value_function[available_indices]
        value_probs = softmax(scaled_values)
        nominated_images = np.random.choice(available_indices, size=num_images, p=value_probs, replace=False)
        category_indices = nominated_images // 12
        image_indices = nominated_images % 12
        return list(zip(category_indices, image_indices))

    return improved_policy


def build_output_dirs(run_name):
    output_root = get_env_value(
        "INTERACTIVE_SEARCH_OUTPUT_DIR",
        default=os.path.join(str(PROJECT_ROOT), "benchmark_results", "exp_interactive_search"),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_name, timestamp)
    selected_images_dir = os.path.join(run_dir, "selected_images")
    plots_dir = os.path.join(run_dir, "plots")
    cache_dir = os.path.join(run_dir, "cache")
    for folder in (selected_images_dir, plots_dir, cache_dir):
        os.makedirs(folder, exist_ok=True)
    return run_dir, selected_images_dir, plots_dir, cache_dir


def save_similarities_json(similarities, save_folder, filename="similarities.json"):
    os.makedirs(save_folder, exist_ok=True)
    filepath = os.path.join(save_folder, filename)
    serializable = [[float(value) for value in iteration] for iteration in similarities]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Data saved to {filepath}")


def save_stacked_eeg_tensor(selected_eegs_per_iteration, save_folder, filename="stacked_eeg_features.pt"):
    stacked_per_iter = [torch.stack(sub_list, dim=0) for sub_list in selected_eegs_per_iteration]
    stacked_eegs = torch.stack(stacked_per_iter, dim=0)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)
    torch.save(stacked_eegs, save_path)
    print(f"Stacked EEG features saved to {save_path} (shape: {stacked_eegs.shape})")
    return stacked_eegs


def main():
    proxy = get_env_value("MINDPILOT_PROXY", default=get_env_value("HTTP_PROXY", default=None))
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy

    cuda_visible_devices = get_env_value("INTERACTIVE_SEARCH_CUDA_VISIBLE_DEVICES", default=None)
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(int(get_env_value("INTERACTIVE_SEARCH_SEED", default="4")))

    image_folder = get_env_value(
        "INTERACTIVE_SEARCH_IMAGE_FOLDER",
        default=os.path.join(str(PROJECT_ROOT), "data", "things-eeg2", "test_images_flat"),
    )
    image_select_root = get_env_value(
        "INTERACTIVE_SEARCH_IMAGE_SELECT_ROOT",
        default=os.path.join(str(PROJECT_ROOT), "data", "things-eeg2", "image_select"),
    )
    image_pool_path = get_env_value(
        "INTERACTIVE_SEARCH_IMAGE_POOL_PATH",
        default=os.path.join(str(PROJECT_ROOT), "data", "clip_embed", "open_clip", "image50_embeddings.pt"),
    )
    target_image_path = get_env_value("INTERACTIVE_SEARCH_TARGET_IMAGE_PATH", default=None)
    target_image_paths = parse_path_list_env("INTERACTIVE_SEARCH_TARGET_IMAGE_PATHS", default=[])
    target_index = int(get_env_value("INTERACTIVE_SEARCH_TARGET_INDEX", default="0"))
    encoder_model_path = get_env_value(
        "INTERACTIVE_SEARCH_ENCODER_MODEL_PATH",
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
        "INTERACTIVE_SEARCH_EEG2CLIP_MODEL_PATH",
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
    output_dir = get_env_value(
        "INTERACTIVE_SEARCH_OUTPUT_DIR",
        default=os.path.join(str(PROJECT_ROOT), "benchmark_results", "exp_interactive_search"),
    )
    max_iterations = int(get_env_value("INTERACTIVE_SEARCH_MAX_ITERATIONS", default="50"))
    num_images = int(get_env_value("INTERACTIVE_SEARCH_NUM_IMAGES", default="10"))
    gamma = float(get_env_value("INTERACTIVE_SEARCH_GAMMA", default="0.9"))
    top_n = int(get_env_value("INTERACTIVE_SEARCH_TOP_N", default="10"))
    sub = get_env_value("INTERACTIVE_SEARCH_SUBJECT", default="sub-01")
    dnn = get_env_value("INTERACTIVE_SEARCH_DNN", default="alexnet")

    text_list = [file_name for file_name in sorted(os.listdir(image_folder)) if not file_name.startswith(".")]

    if target_image_path is None:
        if not target_image_paths:
            raise ValueError(
                "Set INTERACTIVE_SEARCH_TARGET_IMAGE_PATH or INTERACTIVE_SEARCH_TARGET_IMAGE_PATHS in mindpilot.env"
            )
        if target_index >= len(target_image_paths):
            raise ValueError(
                f"INTERACTIVE_SEARCH_TARGET_INDEX={target_index} exceeds INTERACTIVE_SEARCH_TARGET_IMAGE_PATHS ({len(target_image_paths)})"
            )
        target_image_path = target_image_paths[target_index]

    dir_name = os.path.basename(os.path.dirname(target_image_path))
    if dir_name not in text_list:
        raise ValueError(f"Target category '{dir_name}' not found in image folder '{image_folder}'")
    gt_category_id = text_list.index(dir_name)

    run_dir, selected_images_dir, plots_dir, cache_dir = build_output_dirs(dir_name)
    print(f"Output directory: {run_dir}")
    print(f"Target image: {target_image_path}")
    print(f"Image pool: {image_pool_path}")

    image_pool = _load_tensor(image_pool_path, device)
    image_pool = image_pool.view(-1, 1024).detach().cpu()

    checkpoint = torch.load(eeg2clip_model_path, map_location=device, weights_only=False)
    eeg_model = ATMS()
    eeg_model.load_state_dict(checkpoint["eeg_model_state_dict"])

    yhat_path = os.path.join(cache_dir, f"{dir_name}_yhat.pt")
    if os.path.exists(yhat_path):
        yhat = torch.load(yhat_path, weights_only=False).to(device)
    else:
        synthetic_eeg = torch.tensor(get_gteeg(target_image_path, encoder_model_path, dnn, device))
        yhat = get_eeg_features(eeg_model, synthetic_eeg.unsqueeze(0), device, sub)
        torch.save(yhat, yhat_path)

    policy = initial_policy
    history = []
    max_similarities = []
    historical_max_similarity = -np.inf
    min_similarities = []
    variances = []
    selected_eegs_per_iteration = []
    similarities_per_iteration = []

    probabilities = np.ones(image_pool.shape[0], dtype=np.float64) / image_pool.shape[0]
    found_fg = False

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}...")
        temperature = max(1.0, 2.0 - iteration * 0.1)

        visited_images = policy(image_pool, num_images=num_images, history=history)
        history.extend([cat * 12 + idx for cat, idx in visited_images])

        selected_image_paths = []
        similarities = []
        choose_eeg_features = []
        print(f"visited_images {visited_images}")

        for cat, img in visited_images:
            image_path = get_image_path(cat, img, text_list, image_select_root)
            selected_image_paths.append(image_path)
            similarity, choose_eeg_feature = reward_function(image_path, encoder_model_path, yhat, device, sub, dnn)
            similarities.append(similarity)
            choose_eeg_features.append(choose_eeg_feature)

        average_sim = sum(similarities) / len(similarities)
        selected_eegs_per_iteration.append(choose_eeg_features)

        max_similarity = np.max(similarities)
        min_similarity = np.min(similarities)
        if max_similarity < historical_max_similarity:
            max_similarity = historical_max_similarity
        else:
            historical_max_similarity = max_similarity

        max_similarities.append(max_similarity)
        min_similarities.append(min_similarity)
        variances.append(np.var(similarities))

        probabilities = policy_evaluation(
            policy,
            image_pool,
            visited_images,
            similarities,
            encoder_model_path,
            text_list,
            probabilities,
            historical_max_similarity,
            gamma,
            history=history,
        )

        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        category_indices = top_indices // 12
        image_indices = top_indices % 12
        selected_pairs = list(zip(category_indices, image_indices))

        top_image_paths = []
        top_similaritys = []
        for cat, img in selected_pairs:
            image_path = get_image_path(cat, img, text_list, image_select_root)
            top_image_paths.append(image_path)
            similarity, _ = reward_function(image_path, encoder_model_path, yhat, device, sub, dnn)
            top_similaritys.append(similarity)
            if gt_category_id == cat:
                found_fg = True

        visualize_top_images(top_image_paths, top_similaritys, selected_images_dir, iteration)
        similarities_per_iteration.append([float(value) for value in top_similaritys])

        new_policy = policy_improvement(probabilities, image_pool, iteration, run_dir, num_images=num_images, temperature=temperature)
        if found_fg:
            print("Got it.")
            break

        policy = new_policy

    plot_similarity_and_mse_with_dual_axis(similarities_per_iteration, plots_dir, target_similarity=1.0)
    stacked_eegs = save_stacked_eeg_tensor(selected_eegs_per_iteration, plots_dir, filename=f"{dir_name}.pt")
    save_similarities_json(similarities_per_iteration, run_dir, filename=f"{dir_name}.json")

    summary = {
        "run_dir": run_dir,
        "seed": int(get_env_value("INTERACTIVE_SEARCH_SEED", default="4")),
        "image_folder": image_folder,
        "image_select_root": image_select_root,
        "image_pool_path": image_pool_path,
        "target_image_path": target_image_path,
        "encoder_model_path": encoder_model_path,
        "eeg2clip_model_path": eeg2clip_model_path,
        "max_iterations": max_iterations,
        "num_images": num_images,
        "gamma": gamma,
        "top_n": top_n,
        "target_category": dir_name,
        "gt_category_id": gt_category_id,
        "found_fg": found_fg,
        "similarities_per_iteration_shape": [len(item) for item in similarities_per_iteration],
        "stacked_eegs_shape": list(stacked_eegs.shape),
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
