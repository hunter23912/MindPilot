import torch
import os
import numpy as np
from PIL import Image
import torch.nn as nn


def _preprocess_image(path, device):
    image = Image.open(path).convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    return image_tensor.unsqueeze(0).to(device)


class _AlexNet(nn.Module):
    def __init__(self, num_outputs=4250):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def create_model(device, dnn):
    if dnn == 'alexnet':
        model = _AlexNet(num_outputs=4250)
    if dnn == 'cornet_s':
        from CORnet.cornet import CORnet_S
        model = CORnet_S()
        model.decoder = nn.Sequential(
            model.decoder.avgpool,
            model.decoder.flatten,
            model.decoder.linear,
            nn.Linear(in_features=1000, out_features=4250), 
            model.decoder.output 
        )
    model = model.to(device)
    return model

def load_model_encoder(model_path, device):
    model = create_model(device, 'alexnet')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['best_model'])
    model.eval()
    return model

def preprocess_image(path, device):
    return _preprocess_image(path, device)

def generate_eeg(model, image_tensor, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        eeg_output = model(image_tensor).detach().cpu().numpy()
        eeg_output = np.reshape(eeg_output, (17, 250))
    return eeg_output

def save_eeg_signal(eeg_signal, save_dir, idx, category):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"{category}_{idx + 1}.npy" 
    file_path = os.path.join(save_dir, file_name)
    np.save(file_path, eeg_signal)

