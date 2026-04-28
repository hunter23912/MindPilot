import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

from mindpilot_paths import get_env_value, get_project_root, load_mindpilot_env

load_mindpilot_env()

PROJECT_ROOT = get_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from model.ATMS_retrieval import ATMS, get_eeg_features
from model.custom_pipeline_low_level import Generator4Embeds
from model.pseudo_target_model import PseudoTargetModel
from util import get_gteeg, save_eeg

import logging

logging.getLogger("diffusers").setLevel(logging.WARNING)


def reward_function_clip_embed(embed1, embed2):
    cosine_sim = F.cosine_similarity(embed1, embed2, dim=1)
    normalized_sim = (cosine_sim + 1) / 2
    return normalized_sim.item()


def reward_function(eeg_model, image_path, encoder_model_path, yhat, sub, dnn, device):
    eeg_signal = torch.tensor(get_gteeg(image_path, encoder_model_path, dnn, device))
    eeg_feature = get_eeg_features(eeg_model, eeg_signal.unsqueeze(0), device, sub)
    similarity = torch.nn.functional.cosine_similarity(eeg_feature, yhat)
    similarity = (similarity + 1) / 2
    return similarity.item(), eeg_feature


def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def get_image_path(category_idx, image_idx, text_list, image_select_root):
    category_folder = text_list[category_idx]
    folder_path = os.path.join(image_select_root, category_folder)
    image_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith((".jpg", ".png", ".jpeg")) and not f.startswith("._")]
    return os.path.join(folder_path, image_files[image_idx])


def parse_path_list_env(name, default_list):
    raw_value = get_env_value(name, default=None)
    if raw_value is None or raw_value.strip() == "":
        return default_list
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    return items if items else default_list


def build_run_dirs(dir_name):
    output_root = get_env_value(
        "EEG_LATENT_OUTPUT_DIR",
        default=os.path.join(str(PROJECT_ROOT), "benchmark_results", "exp_heuristic_generation_eeg_latent"),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, dir_name, timestamp)
    gt_eeg_dir = os.path.join(run_dir, "syn_eeg_gt")
    pseudo_data_dir = os.path.join(run_dir, "pseudo_train_data")
    clip_embed_dir = os.path.join(run_dir, "clip_embed")
    plots_dir = os.path.join(run_dir, "plots")
    for folder in [gt_eeg_dir, pseudo_data_dir, clip_embed_dir, plots_dir]:
        os.makedirs(folder, exist_ok=True)
    return run_dir, gt_eeg_dir, pseudo_data_dir, clip_embed_dir, plots_dir


def main():
    proxy = get_env_value("MINDPILOT_PROXY", default=get_env_value("HTTP_PROXY", default=None))
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub = get_env_value("EEG_LATENT_SUBJECT", default="sub-01")
    dnn = get_env_value("EEG_LATENT_DNN", default="alexnet")
    seed = int(get_env_value("EEG_LATENT_SEED", default="4"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_type = "ViT-H-14"
    vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
        model_type,
        pretrained="laion2b_s32b_b79k",
        precision="fp32",
        device=device,
    )
    vlmodel.to(device)

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
        default=os.path.join(str(PROJECT_ROOT), "data", "image50_embeddings.pt"),
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
    f_encoder = get_env_value(
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

    text_list = sorted([name for name in os.listdir(category_root) if not name.startswith(".")])
    single_target_image_path = get_env_value("EEG_LATENT_TARGET_IMAGE_PATH", default=None)
    if single_target_image_path:
        image_gt_paths = [single_target_image_path]
        target_index = 0
    else:
        image_gt_paths = parse_path_list_env("EEG_LATENT_TARGET_IMAGE_PATHS", default_list=[])
        if not image_gt_paths:
            raise ValueError(
                "Set EEG_LATENT_TARGET_IMAGE_PATH or EEG_LATENT_TARGET_IMAGE_PATHS in mindpilot.env"
            )
        target_index = int(get_env_value("EEG_LATENT_TARGET_INDEX", default="0"))
        if target_index >= len(image_gt_paths):
            raise ValueError(
                f"EEG_LATENT_TARGET_INDEX={target_index} exceeds available target paths ({len(image_gt_paths)})"
            )
    image_gt_path = image_gt_paths[target_index]
    dir_name = os.path.basename(os.path.dirname(image_gt_path))
    gt_category_id = text_list.index(dir_name)

    image_pool = torch.load(image_pool_path, map_location=device, weights_only=False)
    if isinstance(image_pool, dict):
        if "img_features" in image_pool:
            image_pool = image_pool["img_features"]
        elif "features" in image_pool:
            image_pool = image_pool["features"]
        elif "embed" in image_pool:
            image_pool = image_pool["embed"]
        else:
            first_key = next(iter(image_pool.keys()))
            image_pool = image_pool[first_key]
    image_pool = image_pool.view(-1, 1024).detach().cpu()

    checkpoint = torch.load(f_encoder, map_location=device, weights_only=False)
    eeg_model = ATMS()
    eeg_model.load_state_dict(checkpoint["eeg_model_state_dict"])

    run_dir, gt_eeg_dir, pseudo_data_dir, clip_embed_dir, plots_dir = build_run_dirs(dir_name)
    print(f"Output directory: {run_dir}")

    gt_eeg_path = os.path.join(gt_eeg_dir, f"gt_{os.path.splitext(os.path.basename(image_gt_path))[0]}.npy")
    print(f"gt_eeg_path {gt_eeg_path}")
    try:
        synthetic_eeg = torch.from_numpy(np.load(gt_eeg_path))
    except Exception:
        synthetic_eeg = torch.tensor(get_gteeg(image_gt_path, encoder_model_path, dnn, device))
    gt_eeg_path = save_eeg(synthetic_eeg, gt_eeg_dir, file_name=f"gt_{os.path.splitext(os.path.basename(image_gt_path))[0]}.npy")

    tar_eeg_features = get_eeg_features(eeg_model, synthetic_eeg.unsqueeze(0), device, sub)

    batch_size = 32
    alpha = 80
    max_iterations = int(get_env_value("EEG_LATENT_MAX_ITERATIONS", default="50"))
    gamma = 0.9
    num_images = int(get_env_value("EEG_LATENT_NUM_IMAGES", default="10"))
    num_inference_steps = int(get_env_value("EEG_LATENT_NUM_INFERENCE_STEPS", default="8"))
    guidance_scale = 0.0
    dimension = 1024
    self_improvement_ratio = 0.5
    reward_scaling_factor = 100
    initial_step_size = 30
    decay_rate = 0.1
    generate_batch_size = int(get_env_value("EEG_LATENT_GENERATE_BATCH_SIZE", default="3"))
    save_per = int(get_env_value("EEG_LATENT_SAVE_PER", default="5"))
    prompt = get_env_value("EEG_LATENT_PROMPT", default="")

    pipe = Generator4Embeds(device=device).pipe
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl_vit-h.bin",
        torch_dtype=torch.bfloat16,
    )
    pipe.set_ip_adapter_scale(0.5)

    generator = torch.Generator(device=device).manual_seed(seed)
    epsilon = torch.randn(
        num_inference_steps + 1,
        generate_batch_size,
        pipe.unet.config.in_channels,
        pipe.unet.config.sample_size,
        pipe.unet.config.sample_size,
        device=device,
        generator=generator,
    )

    epsilon_init = epsilon.clone()
    epsilon_init_norm = epsilon.flatten(2).norm(dim=-1)[:, :, None, None, None]
    all_images = []

    pseudo_target = torch.randn(generate_batch_size, dimension, device=device, generator=generator)

    data_x_list = []
    data_y_list = []

    for i, img_embed in enumerate(image_pool[::12]):
        image_path = get_image_path(i, 0, text_list, image_select_root)
        similarity, choose_eeg_feature = reward_function(
            eeg_model,
            image_path,
            encoder_model_path,
            tar_eeg_features,
            sub,
            dnn,
            device,
        )
        data_x_list.append(img_embed)
        data_y_list.append(-similarity * 100)

    data = {
        "data_x": torch.stack(data_x_list),
        "data_y": torch.tensor(data_y_list),
    }

    pseudo_target_model = PseudoTargetModel(dimension=dimension, noise_level=1e-4).to(device)
    pseudo_target_model.add_model_data(data["data_x"].to(device), data["data_y"].to(device))

    pseudo_data_path = os.path.join(pseudo_data_dir, f"{dir_name}_data_scaling.pth")
    torch.save(data, pseudo_data_path)

    clip_embed_path = os.path.join(clip_embed_dir, f"{dir_name}_eeg_embeds.pt")
    torch.save(tar_eeg_features, clip_embed_path)

    for step in range(max_iterations):
        latents = pipe(
            [prompt] * generate_batch_size,
            ip_adapter_image_embeds=[pseudo_target.unsqueeze(0).type(torch.bfloat16).to(device)],
            latents=epsilon[0].type(torch.bfloat16),
            given_noise=epsilon[1:].type(torch.bfloat16),
            output_type="latent",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=1.0,
        ).images

        images = pipe.vae.decode(
            (latents / pipe.vae.config.scaling_factor) + (pipe.vae.config.shift_factor if pipe.vae.config.shift_factor else 0.0),
            return_dict=False,
        )[0]
        images = pipe.image_processor.postprocess(images.detach())

        image_inputs = torch.stack([preprocess_train(img) for img in images])
        with torch.no_grad():
            image_features = vlmodel.encode_image(image_inputs.to(device))

        scaled_similarity = reward_function_clip_embed(
            image_features,
            tar_eeg_features.expand(generate_batch_size, 1024),
        ) * reward_scaling_factor

        step_size = initial_step_size / (1 + decay_rate * step)
        pseudo_target, _ = pseudo_target_model.estimate_pseudo_target(image_features, step_size=step_size)

        if step % save_per == 0:
            print(f"scaled_similarity {scaled_similarity}")
            all_images.append(images)

    merged_image = Image.new("RGB", (all_images[0][0].size[0] * len(all_images[0]), all_images[0][0].size[1] * len(all_images)))
    for row_idx, row in enumerate(all_images):
        for col_idx, img in enumerate(row):
            merged_image.paste(img, (col_idx * img.size[0], row_idx * img.size[1]))

    merged_path = os.path.join(plots_dir, "output.jpg")
    merged_image.save(merged_path)
    print(f"Merged image saved to: {merged_path}")

    if len(images) > 0:
        preview_path = os.path.join(run_dir, "preview.png")
        images[0].save(preview_path)
        print(f"Preview image saved to: {preview_path}")

    summary = {
        "dir_name": dir_name,
        "gt_category_id": gt_category_id,
        "seed": seed,
        "max_iterations": max_iterations,
        "num_images": num_images,
        "generate_batch_size": generate_batch_size,
        "save_per": save_per,
        "image_pool_path": image_pool_path,
        "gt_eeg_path": gt_eeg_path,
        "pseudo_data_path": pseudo_data_path,
        "clip_embed_path": clip_embed_path,
        "final_scaled_similarity": scaled_similarity,
        "epsilon_init_norm_mean": float(epsilon_init_norm.mean().item()),
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
