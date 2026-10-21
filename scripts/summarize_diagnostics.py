"""Write an evidence-linked first diagnostic readout after completed runs."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from src.research_io import ROOT, BEST, now, write_json, output_path, sha256


def main():
    full = json.loads((ROOT / "outputs/D1_full/summary.json").read_text())
    pilot = json.loads((ROOT / "outputs/P0_D1_D5_pilot/summary.json").read_text())
    if full["status"] != "completed" or pilot["status"] != "completed":
        raise RuntimeError("Readout requires both runs to complete")
    historical_path = BEST / "outputs/clean_experiment_finals_20260801_212431/eval_results.json"
    historical = {row["model"]: row for row in json.loads(historical_path.read_text())}
    mapping = {"p0": ("p0_full", "no_cfg"), "p03": ("p03_full", "geom_cfg"),
               "p05_original": ("p05_full", "geom_cfg"), "p07": ("p07_full", "geom_cfg"),
               "p05_best": ("repro_p05_full_epoch500", "geom_cfg")}
    checks = []
    for model, (name, mode) in mapping.items():
        row = next(r for r in full["rows"] if r["model"] == model and r["mode"] == mode)
        reference = historical[name]
        checks.append({"model": model, "historical": reference["mean"], "current": row["legacy"]["mean"],
                       "abs_difference_deg": abs(reference["mean"] - row["legacy"]["mean"]),
                       "num_images": row["legacy"]["num_images"],
                       "same_fallback_count": row["mask_fallbacks"] == reference["face_mask_fallbacks"]})
    write_json(ROOT / "provenance/historical_reproduction.json",
               {"created_at": now(), "source": str(historical_path),
                "source_sha256": sha256(historical_path), "checks": checks,
                "all_passed": all(c["abs_difference_deg"] < 1e-6 and c["same_fallback_count"]
                                  and c["num_images"] == 1998 for c in checks)})
    if not all(c["abs_difference_deg"] < 1e-6 and c["same_fallback_count"] for c in checks):
        raise RuntimeError("Historical reproduction differs; inspect before interpreting")
    def row(model, mode):
        return next(r for r in full["rows"] if r["model"] == model and r["mode"] == mode)
    baseline = row("p05_best", "geom_cfg")
    rows = [
        "# 首轮诊断结论", "", f"整理时间：{now()}。", "",
        "本轮完成 P0 的首批协议组件、D0 兼容检查、D1 全量对照及 D2–D5 小样本诊断。"
        "未进行新训练；通用数据协议与后续训练入口仍需继续实现。", "",
        "## 已完成与数据范围", "",
        "- 协议测试 13 项通过：积分总长度、局部随机数、引导分支、有效像素、单位项、法线插值与几何变换。",
        "- D0：no-CFG/Geom-CFG 与复制的历史推理输出最大绝对差为 0，首张样例的 mask/角误差也一致。",
        "- D1：1,998 张 / 100 个身份，11 组配置；每组 38 张 dlib 失败后按历史规则回退 GT 前景。",
        "- D2–D4：固定 32 个身份，每个身份一张图片，8 个推理 seed；这一组有 1 张 mask 回退。",
        "- D5：原有 33 张无 GT 图片，仅作引导分布和可视化诊断，不代表新增物体/室内量化测试。", "",
        "## 历史复现", "",
        f"最佳历史权重：{baseline['legacy']['mean']:.9f}°；"
        f"浮点单位法线协议：{baseline['float']['mean']:.9f}°。",
        "p0、p0.3、原 p0.5、p0.7 和最佳复现这五个历史参考，"
        "均与已有结果的 MAE 在 1e-6° 内一致，图片数量与 fallback 数量一致。",
        "[复现核对明细](${FNFLOW_ROOT}/provenance/historical_reproduction.json)", "",
        "## D1：训练权重与推理引导分开看", "",
        "| checkpoint | 纯条件 MAE° | fixed-CFG 1.5 MAE° | Geom-CFG MAE° | Geom 相对纯条件变化° |",
        "|---|---:|---:|---:|---:|"]
    for name in ["p03", "p05_original", "p07"]:
        no, fixed, geom = [row(name, mode)["legacy"]["mean"]
                           for mode in ["no_cfg", "fixed_cfg", "geom_cfg"]]
        rows.append(f"| {name} | {no:.4f} | {fixed:.4f} | {geom:.4f} | {geom-no:+.4f} |")
    rows.extend([
        "",
        "- 原 p0.5 的 Geom-CFG 改善约 0.04°；p0.3 和 p0.7 的 Geom-CFG 在本协议下变差，固定 1.5 引导对三者均变差。",
        "- p0 仅评纯条件：26.4529°。它没有受过零条件训练，不加入正常 CFG 优劣排名。",
        "- 约 8.5° 的旧配置差距不能归因于推理 CFG。更大差距已经存在于不同训练权重的纯条件输出中。",
        "- 这不是对 dropout 的多次独立训练因果实验，也不是通用域结论；后续新域仍保留必要 CFG 候选。", "",
        "![D1 全量结果](${FNFLOW_ROOT}/figures/D1_full.png)", "",
        "## D2–D4：实际模型并没有消除噪声依赖", "",
        "| Euler 步数 | MAE°：浮点协议、seed 42、32 身份 |", "|---:|---:|"])
    for steps in [1, 2, 4, 8]:
        r = next(r for r in pilot["rows"] if r["model"] == "p05_best" and
                 r["mode"] == "geom_cfg" and r["seed"] == 42 and r["steps"] == steps)
        rows.append(f"| {steps} | {r['float']['mean']:.4f} |")
    singles = [r["float"]["mean"] for r in pilot["rows"] if r["model"] == "p05_best" and
               r["mode"] == "geom_cfg" and r["steps"] == 1 and isinstance(r["seed"], int)]
    ensemble = next(r for r in pilot["rows"] if r["task"] == "D4")["float"]["mean"]
    rows.extend([
        "",
        f"- 八个推理 seed 的 MAE 范围：{min(singles):.4f}°–{max(singles):.4f}°；"
        f"平均 {np.mean(singles):.4f}°，seed 间标准差 {np.std(singles):.4f}°（不是训练 seed 标准差）。",
        f"- 八个单位方向样本求均值再单位化：{ensemble:.4f}°。"
        "它优于八个单样本误差的平均值，但不优于本组的 seed 42；不能用最好的噪声 seed 代替正式平均表现。",
        f"- 逐像素平均角度离散度 {pilot['noise']['mean_angular_dispersion_deg']:.4f}°，"
        f"与方向均值误差的 Pearson 相关约 {pilot['noise']['pearson_dispersion_error']:.4f}。"
        "该相关按像素汇总且仅来自 32 张，不能当作独立样本显著性或校准证据。",
        "- 本组中增加步数未改善 MAE；结论限于当前权重、引导和子集，不推广为“所有 FM 多步无效”。",
        "- 当前证据不足以优先启动歧义/校准不确定性路线，先保留为备选。", "",
        "![D2–D4 小样本结果](${FNFLOW_ROOT}/figures/D2_D4_pilot.png)", "",
        "## D5：未观察到普遍的零引导权重", "",
        f"33 张图的整图平均 s={pilot['ood']['mean_guidance']:.4f}，"
        f"s=0 的平均像素比例为 {pilot['ood']['mean_zero_fraction']*100:.5f}%。",
        "这一集合没有显示大面积 s≈0 的现象；统计包含背景，且没有 GT，不能据此证明通用法线准确。",
        "对照图和浮点法线位于 outputs/P0_D1_D5_pilot/ood。", "",
        "## 对后续工作的影响", "",
        "1. 继续采用修订方案的主线：可靠监督下的跨域几何细节；保留高斯起点、线性路径和速度匹配。",
        "2. 参数化主对照优先纯条件，另列必要 CFG 配置；不再根据旧的 8.5° 差距预设 CFG 带来大收益。",
        "3. 保留可配置推理 seed，正式比较报告固定的一组 seed，避免挑选噪声。",
        "4. 先核查现有 Hypersim 与真实室内标签的转换/划分，再补物体训练数据；完成通用训练入口后运行 R/V/Xv/Xn。",
        "5. 浮点输出修正影响很小，但作为协议保留；不能把这项修正包装为精度贡献。", "",
        "## 证据位置", "",
        "- [代码与全部结果记录](${FNFLOW_ROOT}/docs/代码与实验记录.md)",
        "- [D1 全量机器可读结果](${FNFLOW_ROOT}/outputs/D1_full/summary.json)",
        "- [D2–D5 固定小样本结果](${FNFLOW_ROOT}/outputs/P0_D1_D5_pilot/summary.json)",
        "- [数据来源初查](${FNFLOW_ROOT}/docs/数据来源初查.md)",
        "- [Best 最终完整性核对](${FNFLOW_ROOT}/provenance/best_after.json)", "",
        "本轮未使用 FaceScape 选择新的通用训练超参；它只用于历史诊断。"
        "“最佳复现”指原有固定权重，不是从本轮测试重新选出的训练模型。", ""])
    output_path(ROOT / "docs/首轮诊断结论.md").write_text("\n".join(rows), encoding="utf-8")
    print("Readout written; all five historical reference results reproduced.", flush=True)


if __name__ == "__main__":
    main()
