from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class DummyPreprocess:
    def __call__(self, image):
        image = image.convert('RGB').resize((224, 224))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return (image_tensor - mean) / std


class DummyVLModel(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.image_proj = nn.Parameter(torch.randn(3, embed_dim, generator=generator), requires_grad=False)
        self.text_proj = nn.Parameter(torch.randn(1, embed_dim, generator=generator), requires_grad=False)

    def to(self, device):
        super().to(device)
        return self

    def eval(self):
        return self

    def encode_image(self, image_tensor):
        image_tensor = image_tensor.float()
        pooled = image_tensor.mean(dim=(2, 3))
        embeds = torch.tanh(pooled @ self.image_proj)
        return embeds

    def encode_text(self, text_tokens):
        text_tokens = text_tokens.float()
        pooled = text_tokens.mean(dim=1, keepdim=True)
        embeds = torch.tanh(pooled @ self.text_proj)
        return embeds.repeat(1, self.image_proj.shape[1])


class DummyVAEConfig:
    shift_factor = 0.0
    scaling_factor = 1.0


class DummyVAE:
    config = DummyVAEConfig()

    def decode(self, latents, return_dict=False):
        latents = latents.float()
        image = latents[:, :3, :, :]
        image = torch.nn.functional.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image = torch.tanh(image)
        return (image,)


class DummyImageProcessor:
    def postprocess(self, images):
        images = images.detach().cpu().clamp(-1, 1)
        images = (images + 1.0) / 2.0
        pil_images = []
        for image in images:
            array = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            pil_images.append(Image.fromarray(array))
        return pil_images


class DummyUNetConfig:
    in_channels = 4
    sample_size = 8


class DummyUNet:
    config = DummyUNetConfig()


class DummyPipe:
    def __init__(self):
        self.unet = DummyUNet()
        self.vae = DummyVAE()
        self.image_processor = DummyImageProcessor()

    def load_ip_adapter(self, *args, **kwargs):
        return None

    def set_ip_adapter_scale(self, *args, **kwargs):
        return None

    def to(self, device):
        return self

    def __call__(self, prompts, ip_adapter_image_embeds=None, latents=None, given_noise=None, output_type='pil', num_inference_steps=None, guidance_scale=None, eta=None):
        batch_size = len(prompts)
        if latents is None:
            latents = torch.randn(
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            latents = latents.clone().float()

        if ip_adapter_image_embeds is not None:
            embeds = ip_adapter_image_embeds[0].float()
            if embeds.dim() == 3:
                embeds = embeds.squeeze(0)
            bias = embeds.mean(dim=-1, keepdim=True).view(-1, 1, 1, 1)
            latents = latents + bias

        if output_type == 'latent':
            return SimpleNamespace(images=latents)

        decoded = self.vae.decode(latents, return_dict=False)[0]
        images = self.image_processor.postprocess(decoded)
        return SimpleNamespace(images=images)


class HeuristicGenerator:
    def __init__(self, pipe, vlmodel, preprocess_train, device='cpu', seed=42, load_ip_adapter=False, min_data_threshold=10):
        self.pipe = pipe
        self.vlmodel = vlmodel
        self.preprocess_train = preprocess_train
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)
        self.min_data_threshold = min_data_threshold
        self.reward_scaling_factor = 100
        self.total_steps = 15
        self.initial_step_size = 30
        self.decay_rate = 0.1
        self.num_inference_steps = 8
        self.generate_batch_size = 1
        self.dimension = 1024
        self.guidance_scale = 0.0
        self.pseudo_target_model = None

    def generate(self, data_x, data_y, tar_image_embed, prompt='', save_path=None, start_embedding=None):
        if start_embedding is not None:
            pseudo_target = start_embedding.expand(self.generate_batch_size, self.dimension).to(self.device)
        else:
            pseudo_target = torch.randn(self.generate_batch_size, self.dimension, generator=self.generator)

        latents = torch.randn(
            self.generate_batch_size,
            self.pipe.unet.config.in_channels,
            self.pipe.unet.config.sample_size,
            self.pipe.unet.config.sample_size,
            generator=self.generator,
        )
        result = self.pipe(
            [prompt] * self.generate_batch_size,
            ip_adapter_image_embeds=[pseudo_target.unsqueeze(0).float()],
            latents=latents,
            output_type='pil',
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            eta=1.0,
        )
        return result.images


vlmodel = DummyVLModel().eval()
preprocess_train = DummyPreprocess()
pipe = DummyPipe()
