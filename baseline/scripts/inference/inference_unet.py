"""FaceFlux UNet 通用图片推理。

推理实现直接复用 Facescape 评测脚本，确保与最优流程一致：
- UNet-B
- 256×256
- 1-step Euler
- Geom-CFG
- no-TTA
- seed=42


cd ${FACEFLUX_BEST_ROOT}

CUDA_VISIBLE_DEVICES=0 python \
  scripts/inference/inference_unet.py \
  --input data/test_images_1/imgs \
  --output_dir outputs/general_inference \
  --ckpt checkpoints/repro_p05_full_exact/epoch_500.pth



"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from scripts.eval.eval_facescape_modes import infer_flow, load_unet


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def main():
    parser = argparse.ArgumentParser(
        description="FaceFlux UNet 通用法向量推理"
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints/repro_p05_full_exact/epoch_500.pth",
        help="UNet checkpoint",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入图片目录",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="输出 normal 图片目录",
    )
    parser.add_argument(
        "--model_size",
        choices=["S", "B", "L"],
        default="B",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--inference_mode",
        choices=["no_tta", "ours", "no_cfg", "naive_tta"],
        default="no_tta",
        help="最优复现配置为 no_tta",
    )
    args = parser.parse_args()

    ckpt_path = resolve_path(args.ckpt)
    input_dir = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 不存在：{ckpt_path}")

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"输入目录不存在：{input_dir}")

    image_names = sorted(
        name
        for name in os.listdir(input_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not image_names:
        raise RuntimeError(f"输入目录中没有图片：{input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    model = load_unet(ckpt_path, args.model_size)

    times = []

    for image_name in tqdm(image_names, desc="Inference"):
        image_path = os.path.join(input_dir, image_name)
        rgb = Image.open(image_path).convert("RGB")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        prediction = infer_flow(
            model=model,
            rgb_pil=rgb,
            num_steps=args.steps,
            size=args.image_size,
            inference_mode=args.inference_mode,
            legacy_steps=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

        output = np.clip(
            np.rint(prediction * 255.0),
            0,
            255,
        ).astype(np.uint8)

        output_name = os.path.splitext(image_name)[0] + ".png"
        Image.fromarray(output).save(
            os.path.join(output_dir, output_name)
        )

    mean_time = float(np.mean(times))

    print(f"完成：{len(image_names)} 张图片")
    print(f"输出目录：{output_dir}")
    print(f"平均耗时：{mean_time:.4f} 秒/张")
    print(f"推理模式：{args.inference_mode}")
    print(f"分辨率：{args.image_size}")
    print(f"Steps：{args.steps}")
    print("随机种子：42（由共享推理实现固定）")


if __name__ == "__main__":
    main()