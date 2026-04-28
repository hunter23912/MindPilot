import os
import random
import sys
from datetime import datetime
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from IPython.display import display

from mindpilot_paths import get_env_value, get_project_root, load_mindpilot_env

load_mindpilot_env()

PROJECT_ROOT = get_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from model.ATMS_retrieval import ATMS, get_eeg_features
from model.custom_pipeline_low_level import Generator4Embeds
from model.pseudo_target_model import PseudoTargetModel
from model.utils import load_model_encoder, generate_eeg, save_eeg_signal
from util import save_eeg, get_gteeg

import logging

logging.getLogger("diffusers").setLevel(logging.WARNING)


def reward_function_clip_embed(embed1, embed2):
    cosine_sim = F.cosine_similarity(embed1, embed2, dim=1)
    return (cosine_sim + 1) / 2


def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def latents_to_images(pipe, latents):
    shift_factor = pipe.vae.config.shift_factor if pipe.vae.config.shift_factor else 0.0
    latents = (latents / pipe.vae.config.scaling_factor) + shift_factor
    images = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(images)


def x_flatten(pipe, x):
    return einops.rearrange(
        x,
        "... C W H -> ... (C W H)",
        C=pipe.unet.config.in_channels,
        W=pipe.unet.config.sample_size,
        H=pipe.unet.config.sample_size,
    )


def x_unflatten(pipe, x):
    return einops.rearrange(
        x,
        "... (C W H) -> ... C W H",
        C=pipe.unet.config.in_channels,
        W=pipe.unet.config.sample_size,
        H=pipe.unet.config.sample_size,
    )


def get_norm(pipe, epsilon):
    return x_flatten(pipe, epsilon).norm(dim=-1)[:, :, None, None, None]


def merge_images_grid(image_grid):
    rows = len(image_grid)
    cols = len(image_grid[0])
    img_width, img_height = image_grid[0][0].size
    merged_image = Image.new("RGB", (cols * img_width, rows * img_height))

    for row_idx, row in enumerate(image_grid):
        for col_idx, img in enumerate(row):
            merged_image.paste(img, (col_idx * img_width, row_idx * img_height))

    return merged_image


def _load_tensor_path(path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    if isinstance(data, dict):
        if "data_x" in data and "data_y" in data:
            return data["data_x"].to(device), data["data_y"].to(device)
        if "img_features" in data:
            return data["img_features"].to(device)
        if "features" in data:
            return data["features"].to(device)
        if "embed" in data:
            return data["embed"].to(device)
        first_key = next(iter(data.keys()))
        return data[first_key].to(device)
    return data.to(device)


def build_output_dirs(target_tag):
    save_root = get_env_value(
        "GRADIENT_FREE_OUTPUT_DIR",
        default=os.path.join(str(PROJECT_ROOT), "benchmark_results", "exp_gradient_free_guidance"),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_root, target_tag, timestamp)
    plots_dir = os.path.join(run_dir, "plots")
    images_dir = os.path.join(run_dir, "generated_imgs")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    return run_dir, plots_dir, images_dir


def main():
    proxy = get_env_value("HTTP_PROXY", default=None)
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "ViT-H-14"

    vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
        model_type,
        pretrained="laion2b_s32b_b79k",
        precision="fp32",
        device=device,
    )
    vlmodel.to(device)

    pipe = Generator4Embeds(device=device).pipe

    sub = get_env_value("GRADIENT_FREE_SUBJECT", default="sub-01")
    dnn = get_env_value("GRADIENT_FREE_DNN", default="alexnet")
    seed_value = int(get_env_value("GRADIENT_FREE_SEED", default="42"))
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    target_data_path = get_env_value(
        "GRADIENT_FREE_DATA_PATH",
        default=os.path.join(
            str(PROJECT_ROOT),
            "data",
            "pseudo_train_data",
            "open_clip",
            "clip_embed_tar",
            "00135_pie_data_scaling.pth",
        ),
    )
    target_image_embed_path = get_env_value(
        "GRADIENT_FREE_TARGET_IMAGE_EMBED_PATH",
        default=os.path.join(
            str(PROJECT_ROOT),
            "data",
            "clip_embed",
            "open_clip",
            "00135_pie_image_embeds.pt",
        ),
    )
    eeg_feature_path = get_env_value(
        "GRADIENT_FREE_EEG_FEATURE_PATH",
        default=os.path.join(
            str(PROJECT_ROOT),
            "data",
            "EEG_feature",
            "sub_features",
            sub,
            "eeg_features_ATMS_test.pt",
        ),
    )

    data = torch.load(target_data_path, map_location=device, weights_only=False)
    data_x, data_y = data["data_x"].to(device), data["data_y"].to(device)
    tar_image_embed = _load_tensor_path(target_image_embed_path, device)
    eeg_embeds = _load_tensor_path(eeg_feature_path, device)

    pseudo_target_model = PseudoTargetModel(dimension=1024, noise_level=1e-4).to(device)
    pseudo_target_model.add_model_data(data_x, data_y)

    batch_size = 32
    alpha = 80
    total_steps = int(get_env_value("GRADIENT_FREE_TOTAL_STEPS", default="30"))
    max_inner_steps = 10
    num_inference_steps = 8
    guidance_scale = 0.0
    dimension = 1024
    self_improvement_ratio = 0.5
    reward_scaling_factor = 100
    initial_step_size = 30
    decay_rate = 0.1
    is_train = False
    prompt = get_env_value("GRADIENT_FREE_PROMPT", default="")
    generate_batch_size = int(get_env_value("GRADIENT_FREE_GENERATE_BATCH_SIZE", default="3"))
    save_per = int(get_env_value("GRADIENT_FREE_SAVE_PER", default="5"))

    run_dir, plots_dir, images_dir = build_output_dirs(Path(target_image_embed_path).stem)
    print(f"Output directory: {run_dir}")

    generator = torch.Generator(device=device).manual_seed(0)

    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl_vit-h.bin",
        torch_dtype=torch.bfloat16,
    )
    pipe.set_ip_adapter_scale(0.5)

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
    epsilon_init_norm = get_norm(pipe, epsilon_init)
    all_images = []

    pseudo_target = torch.randn(generate_batch_size, dimension, device=device, generator=generator)

    for step in range(total_steps):
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

        images = latents_to_images(pipe, latents)
        image_inputs = torch.stack([preprocess_train(img) for img in images])

        with torch.no_grad():
            image_features = vlmodel.encode_image(image_inputs.to(device))

        scaled_similarity = reward_function_clip_embed(
            image_features,
            tar_image_embed.expand(generate_batch_size, 1024),
        ) * reward_scaling_factor

        step_size = initial_step_size / (1 + decay_rate * step)
        pseudo_target, _ = pseudo_target_model.estimate_pseudo_target(image_features, step_size=step_size)

        if step % save_per == 0:
            print(f"scaled_similarity {scaled_similarity}")
            all_images.append(images)

    merged_image = merge_images_grid(all_images)
    merged_path = os.path.join(plots_dir, "output.jpg")
    merged_image.save(merged_path)
    print(f"Merged image saved to: {merged_path}")

    if images:
        display(images[0])

    summary = {
        "run_dir": run_dir,
        "target_data_path": target_data_path,
        "target_image_embed_path": target_image_embed_path,
        "eeg_feature_path": eeg_feature_path,
        "seed": seed_value,
        "total_steps": total_steps,
        "generate_batch_size": generate_batch_size,
        "save_per": save_per,
        "final_step_similarity": float(scaled_similarity.detach().cpu().mean().item()) if torch.is_tensor(scaled_similarity) else float(np.mean(scaled_similarity)),
        "epsilon_init_norm": float(epsilon_init_norm.mean().item()),
    }
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
