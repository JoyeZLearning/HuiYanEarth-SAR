import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from peft import PeftModel
from diffusers import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
import numpy as np
import matplotlib.pyplot as plt

if not hasattr(torch, "xpu"):
    class DummyXPU:
        @staticmethod
        def is_available(): return False

        @staticmethod
        def empty_cache(): pass

        @staticmethod
        def device_count(): return 0

        @staticmethod
        def manual_seed(seed): pass


    torch.xpu = DummyXPU()

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"


CONTROLNET_PATH = "./controlnet_alphaearth_sar_lora_regray/checkpoint-7500"


TEST_INPUT_TIF = "/root/autodl-tmp/Patches_67_Cleaned/NewYork_SAR_AE_S1A_IW_GRDH_1SDV_20200520T225111_20200520T225136_032655_03C836_4E08-0000000000-0000000000_x1024_y1024.tif"

COND_CHANNELS = 65

def load_and_preprocess_condition(tif_path):
    print(f"Reading Tiff: {tif_path}")
    with rasterio.open(tif_path) as src:
        raw_data = src.read()  # (C, H, W)

    raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)

    cond_data = raw_data[2:, :, :].astype(np.float32)

    cond_data[0] = (cond_data[0] - 35.0) / 15.0
    cond_data[0] = np.clip(cond_data[0], -1.0, 1.0)

    if cond_data.shape[0] > 1:
        emb_layers = cond_data[1:]

        mean = np.mean(emb_layers)
        std = np.std(emb_layers)

        if std < 1e-4:
            emb_layers[:] = 0.0
        else:
            emb_layers = (emb_layers - mean) / std

        emb_layers = np.clip(emb_layers, -3.0, 3.0)
        cond_data[1:] = emb_layers

    tensor = torch.from_numpy(cond_data).unsqueeze(0).float()
    downscaled = F.interpolate(tensor, scale_factor=0.5, mode='bilinear', align_corners=False)
    tensor = F.interpolate(downscaled, size=(512, 512), mode='bicubic', align_corners=False)

    return tensor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    cond_tensor = load_and_preprocess_condition(TEST_INPUT_TIF).to(device, dtype=dtype)

    from safetensors.torch import load_file as load_safetensors
    import glob
    found_weights = None
    weight_type = None  # 'safe' or 'bin'

    candidates = [
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin"
    ]

    for fname in candidates:
        p = os.path.join(CONTROLNET_PATH, fname)
        if os.path.exists(p):
            found_weights = p
            if fname.endswith(".safetensors"):
                weight_type = 'safe'
            else:
                weight_type = 'bin'
            break

    if found_weights is None:
        raise FileNotFoundError("无法定位权重文件")

    if weight_type == 'safe':
        state_dict = load_safetensors(found_weights)
    else:
        state_dict = torch.load(found_weights, map_location="cpu")

    new_state_dict = {}
    for k, v in state_dict.items():
        # 清洗 module.
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    try:
        ctrl_config = ControlNetModel.load_config("lllyasviel/sd-controlnet-canny")
    except:
        ctrl_config = ControlNetModel.load_config(CONTROLNET_PATH)

    ctrl_config['conditioning_channels'] = COND_CHANNELS
    controlnet = ControlNetModel.from_config(ctrl_config)

    missing, unexpected = controlnet.load_state_dict(new_state_dict, strict=False)

    controlnet.to(dtype=dtype)
    params = list(controlnet.parameters())
    param_mean = torch.mean(torch.cat([p.flatten().float() for p in params])).item()

    
    output_conv_weight = controlnet.controlnet_mid_block.weight

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        vae=vae, 
        torch_dtype=torch.float32
    ).to(device)

    LORA_PATH = "./controlnet_alphaearth_sar_lora_regray/checkpoint-7500/unet_lora"

    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++"  # 开启 SDE 随机噪声模式
    )
    generator = torch.manual_seed(42)

    image = pipe(
        prompt="satellite sar image, Sentinel-1, radar map, top down view, raw data, grayscale, grainy texture, speckle noise",
        negative_prompt="cloud, color, low resolution, pixelated, blocky, square artifacts, smooth, flat shading, blurring, painting, cartoon",
        image=cond_tensor,
        num_inference_steps=50,  
        controlnet_conditioning_scale=1.1,  
        guidance_scale=2.5,
        generator=generator,
        height=512,
        width=512
    ).images[0]

    img_np = np.array(image)
    if len(img_np.shape) == 3:
        sar_result = img_np[:, :, 0]
    else:
        sar_result = img_np

    Image.fromarray(sar_result.astype(np.uint8)).save("output_raw_data4.png")

    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(sar_result, cmap='gray')  # 它是自动归一化的，会自动拉开灰度
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("output_visual_compare3.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    print("Done! Check output_visual_compare4.png")

    vv_channel = img_np[:, :, 0]
    vh_channel = img_np[:, :, 1]


    # 3. 保存
    Image.fromarray(np.clip(vv_channel, 0, 255).astype(np.uint8)).save("output_vv_9500_ny4.png")
    Image.fromarray(np.clip(vh_channel, 0, 255).astype(np.uint8)).save("output_vh_9500_ny4.png")

    print("Done!")


if __name__ == "__main__":
    main()