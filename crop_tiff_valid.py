import os
import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm  # 建议安装 tqdm 显示进度条: pip install tqdm


input_folder = r"E:\datasets\AlphaEarth_SAR_67_all\Unzip_all_last39"

output_folder = r"E:\datasets\AlphaEarth_SAR_67_all\Patches_67_Cleaned_last39"


patch_size = 512
stride = 256  
valid_threshold = 0.999  

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有 tif 文件
tiff_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]


for filename in tqdm(tiff_files):
    src_path = os.path.join(input_folder, filename)

    try:
        with rasterio.open(src_path) as src:
            width = src.width
            height = src.height

            # 读取 Profile 用于保存
            meta = src.meta.copy()

            # 双重循环切片
            for i in range(0, width, stride):
                for j in range(0, height, stride):
                    if i + patch_size > width or j + patch_size > height:
                        continue

                    window = Window(i, j, patch_size, patch_size)
                    data = src.read(window=window, masked=True)
                    sar_part = data[0:2, :, :]
                    if np.nanmax(sar_part) == 0 and np.nanmin(sar_part) == 0:
                        continue  # 全黑，丢弃
                    data_filled = data.filled(np.nan)

                    invalid_mask = np.isnan(data_filled).any(axis=0)  
                    valid_pixels = np.sum(~invalid_mask)
                    total_pixels = patch_size * patch_size

                    ratio = valid_pixels / total_pixels

                  
                    if ratio < valid_threshold:
                    
                        continue

                    if data.shape[0] > 3:
                        emb_part = data_filled[3:, :, :]
                        if np.all(emb_part == 0):
                            continue

                    transform = src.window_transform(window)
                    meta.update({
                        "driver": "GTiff",
                        "height": patch_size,
                        "width": patch_size,
                        "transform": transform,
                        "count": src.count,  # 保持波段数不变
                        "compress": 'lzw'  # 压缩节省空间
                    })

                    save_name = f"{os.path.splitext(filename)[0]}_x{i}_y{j}.tif"
                    save_path = os.path.join(output_folder, save_name)

                    final_data = data.filled(0)  # 将 NaN 填为 0 保存

                    with rasterio.open(save_path, "w", **meta) as dest:
                        dest.write(final_data)

    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")

print("切片清洗完成！")