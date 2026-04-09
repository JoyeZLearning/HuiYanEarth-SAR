import os
import json  
import pandas as pd
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F

import torchvision.transforms.functional as TF
import random

# 2. 修复 XPU 报错
# --- [修复 1] 防止 Windows 下报 xpu 错误 ---
if not hasattr(torch, "xpu"):
    class DummyXPU:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            pass

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def manual_seed(seed):
            # 假装设置了种子
            return

        @staticmethod
        def manual_seed_all(seed):
            # 假装给所有 XPU 设备设置了种子
            return

        @staticmethod
        def synchronize():
            # 假装同步了设备
            return

        @staticmethod
        def set_device(device):
            pass

        @staticmethod
        def current_device():
            return 0


    torch.xpu = DummyXPU()
# ------------------------------------------

from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import os
from accelerate import Accelerator
from diffusers import (
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel



class AlphaEarthSARDataset(Dataset):
    def __init__(self, data_root, size=512):
        if isinstance(data_root, str):
            self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.tif')]
        elif isinstance(data_root, list):
            self.files = []
            for root in data_root:
                files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.tif')]
                self.files.extend(files)
        else:
            raise ValueError("data_root must be string or list of strings")

        self.size = size
        print(f"Loaded {len(self.files)} files from {len(data_root)} directories")

    def __len__(self):
        return len(self.files)

    def apply_augmentations(self, sar_3ch, control_signal):
        """
        应用随机翻转和微小旋转
        """

        sar_tensor = torch.from_numpy(sar_3ch) if isinstance(sar_3ch, np.ndarray) else sar_3ch
        control_tensor = torch.from_numpy(control_signal) if isinstance(control_signal, np.ndarray) else control_signal

        if random.random() > 0.5:
            sar_tensor = TF.hflip(sar_tensor)
            control_tensor = TF.hflip(control_tensor)

        if random.random() > 0.5:
            sar_tensor = TF.vflip(sar_tensor)
            control_tensor = TF.vflip(control_tensor)

        rot_k = random.choice([0, 1, 2, 3])  # 0=0°, 1=90°, 2=180°, 3=270°
        if rot_k != 0:
            sar_tensor = torch.rot90(sar_tensor, k=rot_k, dims=[1, 2])
            control_tensor = torch.rot90(control_tensor, k=rot_k, dims=[1, 2])


        if random.random() > 0.5:  
            angle = random.uniform(-5, 5)
            sar_tensor = TF.rotate(sar_tensor.unsqueeze(0), angle, fill=0).squeeze(0)
            control_tensor = TF.rotate(control_tensor.unsqueeze(0), angle, fill=0).squeeze(0)

        return sar_tensor, control_tensor

    def __getitem__(self, idx):
        try:
            with rasterio.open(self.files[idx]) as src:
                raw_data = src.read()

            raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)

            # === SAR (Target) ===
            sar_data = raw_data[0:2, :, :].astype(np.float32)

            sar_data = np.clip(sar_data, sar_min, sar_max)
            sar_data = (sar_data - sar_min) / (sar_max - sar_min)  # [0, 1]
            sar_data = (sar_data * 2.0) - 1.0  # [-1, 1]

            # 扩充到 3 通道 (SD requirement)
            sar_3ch = np.concatenate([sar_data, sar_data[0:1, :, :]], axis=0)

            # === Condition ===
            angle_map = raw_data[2:3, :, :].astype(np.float32)
            angle_map = (angle_map - 35.0) / 15.0  # Center around 0
            angle_map = np.clip(angle_map, -1.0, 1.0)  # Clip


            emb_map = raw_data[3:, :, :].astype(np.float32)


            if emb_map.shape[0] > 0:
                mean = np.mean(emb_map)
                std = np.std(emb_map) + 1e-5
                emb_map = (emb_map - mean) / std
                emb_map = np.clip(emb_map, -3.0, 3.0)

            control_signal = np.concatenate([angle_map, emb_map], axis=0)


            sar_3ch, control_signal = self.apply_augmentations(sar_3ch, control_signal)

            return {
                "pixel_values": sar_3ch if isinstance(sar_3ch, torch.Tensor) else torch.from_numpy(sar_3ch),
                "conditioning_pixel_values": control_signal if isinstance(control_signal,
                                                                          torch.Tensor) else torch.from_numpy(
                    control_signal),
                "prompt_ids": torch.zeros(1)  
            }
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))


# === 配置参数 ===
MODEL_ID = "runwayml/stable-diffusion-v1-5" 
DATA_DIR = "/root/autodl-tmp/Patches_67_Cleaned"  # 单个目录的数据路径

# 多个目录的数据路径
DATA_DIRS = [
    "/root/autodl-tmp/Patches_67_Cleaned",           # 第一个数据目录
    "/root/autodl-tmp/Patches_67_Cleaned_last39",          # 第二个数据目录
         # 第三个数据目录
]
OUTPUT_DIR = "./controlnet_alphaearth_sar_lora_regray"
BATCH_SIZE = 4  # 单卡Batch，8卡就是32
EPOCHS = 80

LR_RATE = 2e-5
EMBEDDING_DIM = 64 + 1  


def main():
    accelerator = Accelerator(mixed_precision="fp16")  # 自动处理多卡

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_info = {
        "losses": [],
        "steps": [],
        "epochs_completed": 0,
        "config": {
            "model_id": MODEL_ID,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR_RATE,
            "epochs": EPOCHS,
            "embedding_dim": EMBEDDING_DIM
        }
    }



    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    print("ending tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    print("ending text_encoder")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    print("ending vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,   #  lora_alpha=16 (Scaling = 0.5)。让 LoRA 更新更温和一点
        target_modules=["to_k", "to_q", "to_v", "to_out.0","conv1", "conv2"],
        lora_dropout=0.05,
        bias="none",
    )


    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters() 

    print("ending unet")

    scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    print("ending scheduler")

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    print("Loading Baseline ControlNet weights...")

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")


    temp_dataset = AlphaEarthSARDataset(DATA_DIRS)  # 传递目录列表而不是单个目录

    sample = temp_dataset[0]["conditioning_pixel_values"]
    actual_cond_channels = sample.shape[0]  # 比如 65 或 67
    print(f"Detected Condition Channels from dataset: {actual_cond_channels}")

    old_conv = controlnet.controlnet_cond_embedding.conv_in
    print(f"Original Conv Layer: {old_conv}")  

    new_conv = nn.Conv2d(
        in_channels=actual_cond_channels,  
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding
    )

    old_weights = old_conv.weight.data

    avg_weights = torch.mean(old_weights, dim=1, keepdim=True)

    new_weights = avg_weights.repeat(1, actual_cond_channels, 1, 1)

    new_conv.weight.data = new_weights
    new_conv.bias.data = old_conv.bias.data

    controlnet.controlnet_cond_embedding.conv_in = new_conv

    # 同时必须更新 config，否则保存时会报错/下次加载出错
    controlnet.config.conditioning_channels = actual_cond_channels

    controlnet.train()
    unet.train()  

    controlnet.enable_gradient_checkpointing()
    unet.enable_gradient_checkpointing()

    params_to_optimize = list(controlnet.parameters()) + list(unet.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=LR_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # 数据集
    # dataset = AlphaEarthSARDataset(DATA_DIR)
    dataset = AlphaEarthSARDataset(DATA_DIRS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # Prepare
    controlnet, optimizer, dataloader = accelerator.prepare(
        controlnet, optimizer, dataloader
    )

    vae.to(accelerator.device)
    unet.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # === 训练循环 ===
    global_step = 0
    SAR_PROMPT = "satellite sar image, Sentinel-1, radar map, top down view, raw data, grayscale, grainy texture, speckle noise"

    prompt_ids = tokenizer(
            [SAR_PROMPT] * BATCH_SIZE,  
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True
    ).input_ids.to(accelerator.device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(prompt_ids)[0]

    for epoch in range(EPOCHS):
        for batch in dataloader:
            with accelerator.accumulate(controlnet):
                latents = vae.encode(
                    batch["pixel_values"].to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                offset_noise_scale = 0.2 
                noise += offset_noise_scale * torch.randn(latents.shape[0], latents.shape[1], 1, 1,
                                                          device=latents.device)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],),
                                          device=latents.device)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                control_cond = batch["conditioning_pixel_values"].to(
                    device=accelerator.device,
                    dtype=accelerator.unwrap_model(controlnet).dtype
                )

                down_block_res_samples, mid_block_res_sample = controlnet(
                    controlnet_cond=control_cond,
                    conditioning_scale=1.0,
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                timesteps_indices = timesteps.long()
                alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device)
                sqrt_alphas_cumprod = alphas_cumprod[timesteps_indices] ** 0.5
                sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod[timesteps_indices]) ** 0.5

                snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
                snr_gamma = 5.0 
                weights = torch.minimum(snr, torch.tensor(snr_gamma, device=accelerator.device)) / snr

                raw_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                loss = (raw_loss.mean(dim=[1, 2, 3]) * weights).mean()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            if global_step % 500 == 0: 
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)

                accelerator.unwrap_model(controlnet).save_pretrained(save_path)

                unet_to_save = accelerator.unwrap_model(unet)
                unet_to_save.save_pretrained(os.path.join(save_path, "unet_lora"))

                training_info["losses"].append(float(loss.item()))
                training_info["steps"].append(global_step)

                with open(f"{OUTPUT_DIR}/training_progress.json", 'w', encoding='utf-8') as f:
                    json.dump(training_info, f, indent=2)


            global_step += 1
        training_info["epochs_completed"] = epoch + 1


    if accelerator.is_main_process:
        with open(f"{OUTPUT_DIR}/final_training_history.json", 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2)

        df = pd.DataFrame({
            'step': training_info["steps"],
            'loss': training_info["losses"]
        })
        df.to_csv(f"{OUTPUT_DIR}/training_losses.csv", index=False)

if __name__ == "__main__":
    main()