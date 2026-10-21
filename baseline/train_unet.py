"""
🚀 UNet Flow Matching 训练脚本

使用 ADM-style UNet 替代 DiT 进行 flow matching 训练。
训练逻辑与 train_ablation.py 完全一致，仅替换模型骨干。

用法:
    # Base 模型 (与 FaceNormalFlux 参数量相近)
    python train_unet.py --output_dir checkpoints/repro_p05_full_exact

    # 指定模型大小
    python train_unet.py --model_size S --output_dir checkpoints/unet_small
    python train_unet.py --model_size B --output_dir checkpoints/repro_p05_full_exact
    python train_unet.py --model_size L --output_dir checkpoints/unet_large

    # 自定义 channels
    python train_unet.py --channels 64 128 256 512 --output_dir checkpoints/unet_custom
"""
import os, sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from src.models.model_unet import FaceNormalUNet, FaceNormalUNet_S, FaceNormalUNet_B, FaceNormalUNet_L


# =============================================================
# 数据集 (与 train_ablation.py 完全一致)
# =============================================================

class AugmentedDataset(Dataset):
    def __init__(self, rgb_dir, normal_dir, size):
        self.rgb_files, self.normal_files, self.size = [], [], size
        raw_rgb = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for f in raw_rgb:
            name = os.path.splitext(f)[0]
            rgb_path = os.path.join(rgb_dir, f)
            normal_path = os.path.join(normal_dir, name + ".png")
            if os.path.exists(normal_path):
                self.rgb_files.append(rgb_path)
                self.normal_files.append(normal_path)
        print(f"✅ 数据集加载: {len(self.rgb_files)} 对")
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.color_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.1)
        ])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_files[idx]).convert("RGB").resize((self.size, self.size))
        normal = Image.open(self.normal_files[idx]).convert("RGB").resize((self.size, self.size))
        i, j, h, w = transforms.RandomResizedCrop.get_params(rgb, scale=(0.7, 1.0), ratio=(0.9, 1.1))
        rgb = TF.resized_crop(rgb, i, j, h, w, (self.size, self.size))
        normal = TF.resized_crop(normal, i, j, h, w, (self.size, self.size))
        if random.random() > 0.5:
            rgb = TF.hflip(rgb)
            normal = TF.hflip(normal)
            normal_np = np.array(normal)
            normal_np[:, :, 0] = 255 - normal_np[:, :, 0]
            normal = Image.fromarray(normal_np)
        rgb_aug = self.color_aug(rgb)
        return self.to_tensor(rgb_aug), self.to_tensor(normal)


# =============================================================
# 推理函数
# =============================================================

@torch.no_grad()
def infer_no_cfg(model, cond_img, num_steps, device):
    z = torch.randn_like(cond_img)
    t_seq = torch.linspace(0, 1, num_steps, device=device)
    dt = 1.0 / max(num_steps - 1, 1)
    for t in t_seq:
        t_batch = t.expand(cond_img.shape[0])
        v = model(z, t_batch, cond_img)
        z = z + v * dt
    return z


@torch.no_grad()
def infer_geom_cfg(model, cond_img, num_steps, device, base_scale=1.5):
    z = torch.randn_like(cond_img)
    t_seq = torch.linspace(0, 1, num_steps, device=device)
    dt = 1.0 / max(num_steps - 1, 1)
    zero_cond = torch.zeros_like(cond_img)
    for t in t_seq:
        t_batch = t.expand(cond_img.shape[0])
        v_cond = model(z, t_batch, cond_img)
        v_uncond = model(z, t_batch, zero_cond)
        t_view = t_batch.view(-1, 1, 1, 1)
        n_cond = F.normalize(z + v_cond * (1 - t_view), dim=1)
        n_uncond = F.normalize(z + v_uncond * (1 - t_view), dim=1)
        diff = 1.0 - F.cosine_similarity(n_cond, n_uncond, dim=1)
        scale = (base_scale * (1.0 - diff)).clamp(0.0, base_scale).unsqueeze(1)
        v = v_uncond + scale * (v_cond - v_uncond)
        z = z + v * dt
    return z


# =============================================================
# 训练中评估
# =============================================================

@torch.no_grad()
def evaluate(model, epoch, args, device):
    print(f"\n🧪 评估 Epoch {epoch}...")
    model.eval()
    output_dir = os.path.join(args.output_dir, "eval_output_dir")
    raw_dir = os.path.join(output_dir, "raw")
    compare_dir = os.path.join(output_dir, "compare")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.eval_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    pre = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])

    for img_name in tqdm(image_files, desc="  评估中", ncols=100):
        raw_pil = Image.open(os.path.join(args.eval_img_dir, img_name)).convert("RGB")
        inp = pre(raw_pil).unsqueeze(0).to(device)

        with torch.cuda.amp.autocast(enabled=True):
            if args.drop_prob == 0.0:
                out = infer_no_cfg(model, inp, args.eval_steps, device)
            else:
                out = infer_geom_cfg(model, inp, args.eval_steps, device, base_scale=1.5)

        out_np = ((out.clamp(-1, 1) + 1) / 2).cpu().squeeze(0).permute(1, 2, 0).numpy()
        out_np = (out_np * 255).astype(np.uint8)
        normal_pil = Image.fromarray(out_np).resize((args.image_size, args.image_size), Image.BICUBIC)
        name = os.path.splitext(img_name)[0]
        normal_pil.save(os.path.join(raw_dir, f"{name}_raw.png"))
        rgb_vis = raw_pil.resize((args.image_size, args.image_size), Image.BICUBIC)
        cmp = Image.new("RGB", (args.image_size * 2, args.image_size))
        cmp.paste(rgb_vis, (0, 0))
        cmp.paste(normal_pil, (args.image_size, 0))
        cmp.save(os.path.join(compare_dir, f"{name}_compare.png"))

    print(f"✅ 评估完成: {output_dir}\n")


# =============================================================
# 损失函数 (与 train_ablation.py 一致)
# =============================================================

def compute_loss_flow(pred_v, target_v, z_t, normal_gt, t_view, loss_mode, weight_cos):
    loss_mse = F.mse_loss(pred_v, target_v)

    if loss_mode == 'no_flow':
        # True w/o L_flow: remove velocity-matching supervision.
        pred_x0 = z_t + pred_v * (1 - t_view)

        loss_unit = (
            (pred_x0.norm(dim=1) - 1.0) ** 2
        ).mean() * weight_cos

        loss_cos = (
            1.0
            - F.cosine_similarity(
                pred_x0, normal_gt, dim=1
            ).mean()
        ) * weight_cos

        return loss_unit + loss_cos, {
            'unit': loss_unit.item(),
            'cos': loss_cos.item(),
        }

    elif loss_mode == 'l2_only':
        return loss_mse, {'mse': loss_mse.item()}
    elif loss_mode == 'l2_norm':
        pred_x0 = z_t + pred_v * (1 - t_view)
        loss_unit = ((pred_x0.norm(dim=1) - 1.0) ** 2).mean() * weight_cos
        return loss_mse + loss_unit, {'mse': loss_mse.item(), 'unit': loss_unit.item()}
    elif loss_mode == 'l2_cos':
        pred_x0 = z_t + pred_v * (1 - t_view)
        loss_cos = (1.0 - F.cosine_similarity(pred_x0, normal_gt, dim=1).mean()) * weight_cos
        return loss_mse + loss_cos, {'mse': loss_mse.item(), 'cos': loss_cos.item()}
    else:  # full = L_flow + L_unit + L_ang (l2_norm + l2_cos 的完整组合)
        pred_x0 = z_t + pred_v * (1 - t_view)          # 不预先 normalize
        loss_unit = ((pred_x0.norm(dim=1) - 1.0) ** 2).mean() * weight_cos
        loss_cos  = (1.0 - F.cosine_similarity(pred_x0, normal_gt, dim=1).mean()) * weight_cos
        return loss_mse + loss_unit + loss_cos, {'mse': loss_mse.item(), 'unit': loss_unit.item(), 'cos': loss_cos.item()}


# =============================================================
# 主训练循环
# =============================================================

def set_seed(seed=1247):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(args):
    """根据参数构建 UNet 模型"""
    if args.channels:
        # 用户自定义 channels
        channels = tuple(args.channels)
        model = FaceNormalUNet(channels=channels, num_res_blocks=args.num_res_blocks)
        print(f"📐 模型: FaceNormalUNet (custom channels={channels}, blocks={args.num_res_blocks})")
    else:
        # 预设大小
        builders = {'S': FaceNormalUNet_S, 'B': FaceNormalUNet_B, 'L': FaceNormalUNet_L}
        model = builders[args.model_size]()
        print(f"📐 模型: FaceNormalUNet_{args.model_size}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {total_params/1e6:.2f}M ({total_params:,})")
    return model


def train(args):
    set_seed(1247)
    accelerator = Accelerator(mixed_precision="fp16")
    os.makedirs(args.output_dir, exist_ok=True)
    device = accelerator.device

    # --- 数据集 ---
    dataset = AugmentedDataset(args.rgb_dir, args.normal_dir, args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=8, pin_memory=True, prefetch_factor=2)

    # --- 模型 ---
    model = build_model(args).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- 加载 checkpoint ---
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"✅ 权重加载: {args.checkpoint}")
    else:
        print("🆕 从头开始训练")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # --- 打印配置 ---
    print(f"🚀 UNet Flow Matching 训练开始")
    print(f"   loss={args.loss}, drop_prob={args.drop_prob}")
    print(f"   batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}")

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    ncols=120, disable=not accelerator.is_local_main_process)

        for i, (rgb_img, normal_gt) in enumerate(pbar):
            normal_gt = F.normalize(normal_gt, dim=1, eps=1e-6)

            # ===== Flow Matching =====
            bs = rgb_img.shape[0]
            t = torch.rand((bs,), device=device)
            noise = torch.randn_like(normal_gt)
            t_view = t.view(bs, 1, 1, 1)
            z_t = t_view * normal_gt + (1 - t_view) * noise
            target_v = normal_gt - noise

            # CFG dropout
            if random.random() < args.drop_prob:
                cond_input = torch.zeros_like(rgb_img)
            else:
                cond_input = rgb_img

            pred_v = model(z_t, t, cond_input)

            total_loss, info = compute_loss_flow(
                pred_v, target_v, z_t, normal_gt, t_view,
                args.loss, args.weight_cos)

            accelerator.backward(total_loss / args.grad_accum_steps)
            if (i + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(**{k: f"{v:.4f}" for k, v in info.items()})

        # --- 保存 + 评估 ---
        if (epoch + 1) % args.save_interval == 0 and (epoch + 1) > args.start_save_epoch:
            sd = model.state_dict()
            torch.save(sd, os.path.join(args.output_dir, f"epoch_{epoch+1}.pth"))
            print(f"💾 保存: epoch_{epoch+1}.pth")


def main():
    p = argparse.ArgumentParser(description="🚀 UNet Flow Matching 训练")
    # 模型
    p.add_argument("--model_size", choices=["S", "B", "L"], default="B",
                   help="预设模型大小: S/B/L")
    p.add_argument("--channels", type=int, nargs='+', default=None,
                   help="自定义 channels (覆盖 model_size), e.g. --channels 64 128 256 512")
    p.add_argument("--num_res_blocks", type=int, default=2,
                   help="每个 level 的 ResBlock 数量")
    # 训练参数
    p.add_argument("--loss", choices=["no_flow", "l2_only", "l2_norm", "l2_cos", "full"], default="full",
                   help="Loss 组合")
    p.add_argument("--drop_prob", type=float, default=0.5,
                   help="CFG dropout 概率")
    # 路径
    p.add_argument("--rgb_dir", default=os.path.join(PROJECT_ROOT, "datasets/ffhq_rgb_crop256"))
    p.add_argument("--normal_dir", default=os.path.join(PROJECT_ROOT, "datasets/ffhq_rgb_crop256N"))
    p.add_argument("--output_dir", required=True, help="输出目录")
    p.add_argument("--eval_img_dir", default=os.path.join(PROJECT_ROOT, "data/test_images_1/imgs"))
    p.add_argument("--checkpoint", default=None, help="预训练 checkpoint 路径")
    # 训练超参
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--weight_cos", type=float, default=0.25)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--start_save_epoch", type=int, default=200)
    p.add_argument("--start_epoch", type=int, default=0)
    p.add_argument("--eval_steps", type=int, default=1, help="评估时 ODE 步数")

    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()
