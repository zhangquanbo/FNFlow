"""
📊 FacescapeRender1 法向量评估脚本 (Face Mask 版)

Mask 策略: dlib 面部轮廓凸包（不挖眼睛，不截下巴）∩ GT 前景
支持 _NormalGL_ / _NormalCV_ 后缀自动识别。

输出: 指定输出目录下的 eval_results.json + eval_results.md

用法:
    python scripts/eval/eval_facescape_modes.py --model dsine
    python scripts/eval/eval_facescape_modes.py --model marigold
    python scripts/eval/eval_facescape_modes.py --model unet --ckpt checkpoints/repro_p05_full_exact/epoch_500.pth --steps 1
    python scripts/eval/eval_facescape_modes.py --model all --unet_ckpt checkpoints/repro_p05_full_exact/epoch_500.pth
"""
import os, sys, argparse, json, time, itertools
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================
# 1. 数据集
# =============================================================

def collect_pairs(dataset_dir):
    rgb_dir    = os.path.join(dataset_dir, "RImage")
    normal_dir = os.path.join(dataset_dir, "RNormal")
    pairs = []
    if not os.path.isdir(rgb_dir):
        return pairs

    normal_tag = "_NormalGL_"
    try:
        sample_subj  = sorted(os.listdir(normal_dir))[0]
        sample_files = os.listdir(os.path.join(normal_dir, sample_subj))
        if any("_NormalCV_" in f for f in sample_files):
            normal_tag = "_NormalCV_"
    except Exception:
        pass
    print(f"   Normal suffix: {normal_tag}")

    for subj in sorted(os.listdir(rgb_dir)):
        subj_rgb    = os.path.join(rgb_dir, subj)
        subj_normal = os.path.join(normal_dir, subj)
        if not os.path.isdir(subj_rgb) or not os.path.isdir(subj_normal):
            continue
        for fname in sorted(os.listdir(subj_rgb)):
            if not fname.endswith('.png'):
                continue
            normal_fname = fname.replace('_Image_', normal_tag)
            normal_path  = os.path.join(subj_normal, normal_fname)
            if os.path.exists(normal_path):
                expr = fname.split('_Image_')[0] if '_Image_' in fname else fname
                pairs.append({
                    'subject':     subj,
                    'expression':  expr,
                    'rgb_path':    os.path.join(subj_rgb, fname),
                    'normal_path': normal_path,
                })
    print(f"   {len(pairs)} pairs collected")
    return pairs


# =============================================================
# 2. dlib Face Hull Mask（不挖眼睛，不截下巴）
# =============================================================

_dlib_detector  = None
_dlib_predictor = None

def _get_dlib():
    global _dlib_detector, _dlib_predictor
    if _dlib_detector is None:
        import dlib
        _dlib_detector = dlib.get_frontal_face_detector()
        candidates = [
            os.path.join(PROJECT_ROOT, "data/shape_predictor_68_face_landmarks.dat"),
            os.path.join(PROJECT_ROOT, "assets/shape_predictor_68_face_landmarks.dat"),
            os.path.expanduser("~/shape_predictor_68_face_landmarks.dat"),
            "$HOME/shape_predictor_68_face_landmarks.dat",
        ]
        for p in candidates:
            if os.path.exists(p):
                _dlib_predictor = dlib.shape_predictor(p)
                print(f"   dlib predictor: {p}")
                return _dlib_detector, _dlib_predictor
        raise FileNotFoundError("未找到 shape_predictor_68_face_landmarks.dat")
    return _dlib_detector, _dlib_predictor


def create_face_mask(rgb_pil, size=256):
    """dlib 面部凸包 mask（下巴线 + 眉毛轮廓，不剔除眼睛/嘴巴）"""
    import cv2
    detector, predictor = _get_dlib()

    img_gray = np.array(rgb_pil.resize((size, size), Image.BICUBIC).convert("L"))
    dets = detector(img_gray, 1)
    if len(dets) == 0:
        return None

    shape = predictor(img_gray, dets[0])
    pts   = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

    # 面部凸包：下巴线(0-16) + 眉毛(17-26) 反向
    face_contour = np.concatenate([pts[0:17], pts[26:16:-1]])
    hull = cv2.convexHull(face_contour.astype(np.int32))
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


# =============================================================
# 3. Angular Error 计算
# =============================================================

def compute_angular_error(pred_np, gt_np, face_mask=None, size=256):
    import cv2
    pred_np = pred_np.astype(np.float64)
    gt_np   = gt_np.astype(np.float64)

    if pred_np.max() > 1.5: pred_np /= 255.0
    if gt_np.max()   > 1.5: gt_np   /= 255.0

    # GT 前景 mask
    fg_uint8 = (gt_np.sum(axis=-1) > 0.1).astype(np.uint8)
    # 闭运算填充嘴巴空洞
    close_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fg_closed = cv2.morphologyEx(fg_uint8, cv2.MORPH_CLOSE, close_k)
    # 腐蚀去边界伪影
    erode_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_hard  = cv2.erode(fg_closed, erode_k, iterations=1).astype(bool)

    # 与 dlib 面部轮廓 mask 取交集
    if face_mask is not None:
        fg_hard = fg_hard & face_mask

    if not fg_hard.any():
        return None

    # 转 [-1,1] 并归一化
    pred_f = pred_np * 2.0 - 1.0
    gt_f   = gt_np   * 2.0 - 1.0
    pred_n = pred_f / (np.linalg.norm(pred_f, axis=-1, keepdims=True) + 1e-8)
    gt_n   = gt_f   / (np.linalg.norm(gt_f,   axis=-1, keepdims=True) + 1e-8)

    cos_sim = np.clip((pred_n * gt_n).sum(axis=-1), -1.0, 1.0)
    angles  = np.degrees(np.arccos(cos_sim))[fg_hard]
    angles  = angles[np.isfinite(angles)]

    if len(angles) == 0:
        return None

    return {
        'mean':   float(np.mean(angles)),
        'std':    float(np.std(angles)),
        'lt_20':  float(np.mean(angles < 20.0) * 100),
        'lt_25':  float(np.mean(angles < 25.0) * 100),
        'lt_30':  float(np.mean(angles < 30.0) * 100),
        'n_px':   int(len(angles)),
    }


# =============================================================
# 4. 模型推理
# =============================================================

def get_preprocess(size=256):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def load_unet(ckpt_path, model_size='B'):
    from src.models.model_unet import FaceNormalUNet_S, FaceNormalUNet_B, FaceNormalUNet_L
    builders = {'S': FaceNormalUNet_S, 'B': FaceNormalUNet_B, 'L': FaceNormalUNet_L}
    model = builders[model_size]()
    sd = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in sd: sd = sd["model_state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    print(f"✅ UNet_{model_size}: {ckpt_path}")
    return model


def load_dsine():
    dsine_root = os.path.join(PROJECT_ROOT, "baselines/DSINE/repo")
    sys.path.insert(0, dsine_root)
    from models.dsine.v02 import DSINE_v02
    import utils.utils as dsine_utils

    class DSINEArgs:
        NNET_encoder_B=5; NNET_decoder_NF=2048; NNET_decoder_BN=False
        NNET_decoder_down=8; NNET_learned_upsampling=True
        NRN_prop_ps=5; NRN_num_iter_train=5; NRN_num_iter_test=5
        NRN_ray_relu=True; NNET_output_dim=3; NNET_feature_dim=64
        NNET_hidden_dim=64; NNET_architecture='v02'
        ckpt_path=os.path.join(dsine_root, "projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt")

    import logging; logging.basicConfig(level=logging.WARNING)
    args = DSINEArgs()
    model = DSINE_v02(args).to(DEVICE).eval()
    if os.path.exists(args.ckpt_path):
        model = dsine_utils.load_checkpoint(args.ckpt_path, model)
        print(f"✅ DSINE: {args.ckpt_path}")
    else:
        print(f"⚠️  DSINE ckpt not found: {args.ckpt_path}")
    model.pixel_coords = model.pixel_coords.to(DEVICE)
    return model


def load_marigold():
    from diffusers import MarigoldNormalsPipeline
    print("⏳ 加载 Marigold...")
    pipe = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-lcm-v0-1",
        torch_dtype=torch.float16, variant="fp16"
    ).to(DEVICE)
    print("✅ Marigold loaded")
    return pipe


def infer_flow(
    model,
    rgb_pil,
    num_steps=1,
    size=256,
    inference_mode="ours",
    legacy_steps=False,
):
    pre = get_preprocess(size)
    inp = pre(rgb_pil).unsqueeze(0).to(DEVICE)

    @torch.no_grad()
    def _ode(cond_img, steps, use_cfg=True, seed=42):
        if steps < 1:
            raise ValueError("steps must be >= 1")

        torch.manual_seed(seed)
        z = torch.randn_like(cond_img)
        zero_cond = torch.zeros_like(cond_img)

        if legacy_steps:
            # 原项目实现。steps>1 时总积分长度为 steps/(steps-1)。
            t_seq = torch.linspace(0, 1, steps, device=DEVICE)
            dt = 1.0 / max(steps - 1, 1)
        else:
            # 标准 Euler：steps 次更新，总积分长度恰好为1。
            t_seq = (
                torch.arange(
                    steps, device=DEVICE, dtype=torch.float32
                ) / steps
            )
            dt = 1.0 / steps

        for t in t_seq:
            t_b = t.expand(cond_img.shape[0])

            if use_cfg:
                with torch.cuda.amp.autocast(enabled=True):
                    v_c = model(z, t_b, cond_img)
                    v_u = model(z, t_b, zero_cond)

                t_v = t_b.view(-1, 1, 1, 1)
                n_c = F.normalize(z + v_c * (1 - t_v), dim=1)
                n_u = F.normalize(z + v_u * (1 - t_v), dim=1)
                diff = 1.0 - F.cosine_similarity(
                    n_c, n_u, dim=1
                )
                scale = (
                    1.5 * (1.0 - diff)
                ).clamp(0.0, 1.5).unsqueeze(1)
                v = v_u + scale * (v_c - v_u)
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    v = model(z, t_b, cond_img)

            z = z + v * dt

        return z

    if inference_mode == "no_cfg":
        pred = _ode(
            inp, num_steps, use_cfg=False, seed=42
        )

    elif inference_mode == "no_tta":
        pred = _ode(
            inp, num_steps, use_cfg=True, seed=42
        )

    elif inference_mode == "naive_tta":
        pred1 = _ode(
            inp, num_steps, use_cfg=True, seed=42
        )
        pred2_flip = _ode(
            torch.flip(inp, [3]),
            num_steps,
            use_cfg=True,
            seed=42,
        )
        pred2 = torch.flip(pred2_flip, [3])
        # Naive TTA故意不修正normal x分量。
        pred = 0.5 * (pred1 + pred2)

    elif inference_mode == "ours":
        pred1 = _ode(
            inp, num_steps, use_cfg=True, seed=42
        )
        pred2_flip = _ode(
            torch.flip(inp, [3]),
            num_steps,
            use_cfg=True,
            seed=42,
        )
        pred2 = torch.flip(pred2_flip, [3])
        pred2[:, 0] = -pred2[:, 0]
        pred = 0.5 * (pred1 + pred2)

    else:
        raise ValueError(
            f"Unknown inference_mode: {inference_mode}"
        )

    return (
        (pred.clamp(-1, 1) + 1) / 2
    ).cpu().squeeze(0).permute(1, 2, 0).numpy()


def infer_dsine(model, rgb_pil, size=256):
    dsine_root = os.path.join(PROJECT_ROOT, "baselines/DSINE/repo")
    sys.path.insert(0, dsine_root)
    from utils.projection import intrins_from_fov
    pre = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    inp    = pre(rgb_pil).unsqueeze(0).to(DEVICE)
    intrins = intrins_from_fov(new_fov=60.0, H=size, W=size, device=DEVICE).unsqueeze(0)
    # Accuracy evaluation uses FP32 to prevent occasional NaN predictions.
    with torch.no_grad():
        pred = model(inp, intrins=intrins.clone(), mode='test')[-1]
    return ((pred.clamp(-1, 1) + 1) / 2).cpu().squeeze(0).permute(1, 2, 0).numpy()


def infer_marigold(pipe, rgb_pil, size=256):
    with torch.no_grad():
        output = pipe(rgb_pil, num_inference_steps=4,
                      processing_resolution=768, match_input_resolution=True)
    vis = pipe.image_processor.visualize_normals(output.prediction)
    return np.array(vis[0].resize((size, size), Image.BICUBIC)).astype(np.float32) / 255.0


# =============================================================
# 5. 坐标系校准
# =============================================================

def auto_calibrate_axes(infer_fn, pairs, num_samples=5):
    preds, gts, fgs = [], [], []
    for pair in pairs[:num_samples]:
        try:
            rgb_pil = Image.open(pair['rgb_path']).convert("RGB")
            gt_pil  = Image.open(pair['normal_path']).convert("RGB")
            pred_np = infer_fn(rgb_pil)
            gt_np   = np.array(gt_pil.resize(
                (pred_np.shape[1], pred_np.shape[0]), Image.BICUBIC
            )).astype(np.float64) / 255.0
            fg = gt_np.sum(axis=-1) > 0.01
            preds.append(pred_np.astype(np.float64) * 2.0 - 1.0)
            gts.append(gt_np * 2.0 - 1.0)
            fgs.append(fg)
        except:
            continue
    if not preds:
        return (False, False, False)

    best_err, best_flip = 999, (False, False, False)
    for fx, fy, fz in itertools.product([False, True], repeat=3):
        total = []
        for p, g, fg in zip(preds, gts, fgs):
            pp = p.copy()
            if fx: pp[:,:,0] = -pp[:,:,0]
            if fy: pp[:,:,1] = -pp[:,:,1]
            if fz: pp[:,:,2] = -pp[:,:,2]
            pn = pp / (np.linalg.norm(pp, axis=-1, keepdims=True) + 1e-8)
            gn = g  / (np.linalg.norm(g,  axis=-1, keepdims=True) + 1e-8)
            cos_sim = np.clip((pn * gn).sum(axis=-1), -1.0, 1.0)
            angles  = np.degrees(np.arccos(cos_sim))
            angles  = angles[fg & np.isfinite(angles)]
            total.extend(angles.tolist())
        err = np.mean(total)
        if err < best_err:
            best_err, best_flip = err, (fx, fy, fz)

    fx, fy, fz = best_flip
    print(f"  🔧 坐标系校准: flip_X={fx}, flip_Y={fy}, flip_Z={fz} → {best_err:.1f}°")
    return best_flip


# =============================================================
# 6. 评估单个模型
# =============================================================

def evaluate_model(model_name, infer_fn, pairs, args):
    print(f"\n{'='*60}")
    print(f"📐 评估: {model_name}")
    print(f"{'='*60}")

    # Fixed coordinate conventions; never select transforms from test GT.
    fixed_axes = {
        "DSINE": (True, False, False),
        "Ours": (False, False, False),
        "Marigold": (False, False, False),
    }
    flip_x, flip_y, flip_z = fixed_axes.get(
        model_name, (False, False, False)
    )
    print(
        f"  固定坐标变换: flip_X={flip_x}, "
        f"flip_Y={flip_y}, flip_Z={flip_z}"
    )

    # 官方评测必须使用 dlib face-hull mask。
    # 缺少 dlib 或 landmark predictor 时直接失败，
    # 防止静默切换评测协议。
    _get_dlib()

    all_errors, failed = [], 0
    mask_fallbacks = 0

    for pair in tqdm(pairs, desc=f"  {model_name}", ncols=100):
        try:
            rgb_pil = Image.open(pair['rgb_path']).convert("RGB")
            gt_pil  = Image.open(pair['normal_path']).convert("RGB")

            pred_np = infer_fn(rgb_pil)

            if flip_x: pred_np[:, :, 0] = 1.0 - pred_np[:, :, 0]
            if flip_y: pred_np[:, :, 1] = 1.0 - pred_np[:, :, 1]
            if flip_z: pred_np[:, :, 2] = 1.0 - pred_np[:, :, 2]

            gt_size    = (pred_np.shape[1], pred_np.shape[0])
            gt_resized = np.array(gt_pil.resize(gt_size, Image.BICUBIC)).astype(np.float32) / 255.0

            face_mask = create_face_mask(
                rgb_pil,
                args.eval_size,
            )

            if face_mask is None:
                mask_fallbacks += 1

                if mask_fallbacks <= 5:
                    print(
                        "  ⚠️ dlib 未检测到人脸，"
                        "该图片回退到 GT 前景 mask："
                        f"{pair['rgb_path']}"
                    )

            err = compute_angular_error(
                pred_np,
                gt_resized,
                face_mask,
                args.eval_size,
            )
            if err:
                err['subject']    = pair['subject']
                err['expression'] = pair['expression']
                all_errors.append(err)
            else:
                print(
                    f"  NO_VALID_METRIC: {pair['rgb_path']} "
                    f"subject={pair['subject']} "
                    f"expression={pair['expression']}"
                )

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  ⚠️ 失败: {pair['rgb_path']}: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()

    if mask_fallbacks > 0:
        print(
            f"  ⚠️ dlib face-mask fallback："
            f"{mask_fallbacks}/{len(pairs)} 张使用 GT 前景 mask"
        )

    if failed > 0:
        raise RuntimeError(
            f"评测失败：共有 {failed} 张图片处理异常，"
            "拒绝生成部分数据指标"
        )

    if len(all_errors) != len(pairs):
        raise RuntimeError(
            "评测结果数量不完整："
            f"expected={len(pairs)}, actual={len(all_errors)}"
        )

    if not all_errors:
        raise RuntimeError("评测失败：没有有效结果")

    means = [e['mean'] for e in all_errors]
    result = {
        'model':      model_name,
        'mean':       float(np.mean(means)),
        'std':        float(np.std(means)),
        'lt_20':      float(np.mean([e['lt_20'] for e in all_errors])),
        'lt_25':      float(np.mean([e['lt_25'] for e in all_errors])),
        'lt_30':      float(np.mean([e['lt_30'] for e in all_errors])),
        'num_images': len(all_errors),
        'face_mask_successes': len(all_errors) - mask_fallbacks,
        'face_mask_fallbacks': mask_fallbacks,
        'mask_policy': (
            'dlib_face_hull_intersect_gt; '
            'gt_foreground_fallback_on_detection_failure'
        ),
    }

    print(f"\n  📊 {model_name}: {result['mean']:.2f} ± {result['std']:.2f}° | "
          f"<20°: {result['lt_20']:.1f}% | <25°: {result['lt_25']:.1f}% | <30°: {result['lt_30']:.1f}%")
    return result


# =============================================================
# 7. 保存结果
# =============================================================

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "eval_results.json")
    md_path   = os.path.join(output_dir, "eval_results.md")

    # 合并已有结果
    existing = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                for r in json.load(f):
                    existing[r['model']] = r
        except:
            pass

    for r in results:
        existing[r['model']] = r
    all_results = list(existing.values())

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Normal Reconstruction Error — FacescapeRender1 (Face Hull Mask)\n\n")
        f.write(
            "Mask: dlib 面部凸包 ∩ GT 前景；"
            "dlib 单图检测失败时回退到 GT 前景 mask。\\n\\n"
        )
        f.write(f"最后更新: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(
            "| Method | Mean ± Std | < 20° | < 25° | < 30° | "
            "Mask fallback |\\n"
        )
        f.write(
            "|:------:|:----------:|:-----:|:-----:|:-----:|"
            ":-------------:|\\n"
        )
        for r in all_results:
            fallback = r.get("face_mask_fallbacks", "N/A")
            f.write(
                f"| {r['model']} | "
                f"{r['mean']:.1f} ± {r['std']:.1f} | "
                f"{r['lt_20']:.1f}% | "
                f"{r['lt_25']:.1f}% | "
                f"{r['lt_30']:.1f}% | "
                f"{fallback} |\\n"
            )

    print(f"\n💾 结果保存: {md_path}")
    print(f"\n{'='*65}")
    print(f"  {'Method':<16}| Mean ± Std     | <20°    | <25°    | <30°")
    print(f"  {'-'*64}")
    for r in all_results:
        print(f"  {r['model']:<16}| {r['mean']:5.1f} ± {r['std']:4.1f}°  | "
              f"{r['lt_20']:5.1f}%  | {r['lt_25']:5.1f}%  | {r['lt_30']:5.1f}%")
    print(f"{'='*65}")


# =============================================================
# 8. Main
# =============================================================

def main():
    p = argparse.ArgumentParser(description="📊 FacescapeRender1 法向量评估 (Face Hull Mask)")
    p.add_argument("--model", required=True,
                   choices=["unet", "dsine", "marigold", "all"])
    p.add_argument("--dataset_dir",
                   default=os.path.join(PROJECT_ROOT, "datasets/FacescapeRender1"))
    p.add_argument("--output_dir",
                   default=os.path.join(PROJECT_ROOT, "outputs/facescape_eval"))
    p.add_argument("--ckpt",       default=None)
    p.add_argument("--unet_ckpt",  default=None)
    p.add_argument("--model_name", default=None,
                   help="自定义输出模型名称（覆盖默认的 UNet_B (1step) 等）")
    p.add_argument("--unet_size",  default="B", choices=["S", "B", "L"])
    p.add_argument("--steps",      type=int, default=1)
    p.add_argument("--eval_size",  type=int, default=256)
    p.add_argument(
        "--inference_mode",
        choices=["ours", "no_cfg", "no_tta", "naive_tta"],
        default="no_tta",
    )
    p.add_argument(
        "--legacy_steps",
        action="store_true",
        help="使用原项目的多步积分实现",
    )
    args = p.parse_args()

    pairs = collect_pairs(args.dataset_dir)
    if not pairs:
        print("❌ 没有找到数据对")
        return

    results = []

    def _run(name, infer_fn):
        r = evaluate_model(name, infer_fn, pairs, args)
        if r:
            results.append(r)

    if args.model in ("unet", "all"):
        ckpt = args.ckpt or args.unet_ckpt
        if ckpt and os.path.exists(ckpt):
            model = load_unet(ckpt, args.unet_size)
            default_name = f"UNet_{args.unet_size} (1step)" if args.steps == 1 else f"UNet_{args.unet_size} ({args.steps}steps)"
            name = args.model_name if args.model_name else default_name
            _run(
                name,
                lambda img: infer_flow(
                    model,
                    img,
                    num_steps=args.steps,
                    size=args.eval_size,
                    inference_mode=args.inference_mode,
                    legacy_steps=args.legacy_steps,
                ),
            )
            del model; torch.cuda.empty_cache()
        else:
            print(f"⚠️  UNet ckpt 不存在: {ckpt}")

    if args.model in ("dsine", "all"):
        model = load_dsine()
        _run("DSINE", lambda img: infer_dsine(model, img, args.eval_size))
        del model; torch.cuda.empty_cache()

    if args.model in ("marigold", "all"):
        model = load_marigold()
        _run("Marigold", lambda img: infer_marigold(model, img, args.eval_size))
        del model; torch.cuda.empty_cache()

    if results:
        save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
