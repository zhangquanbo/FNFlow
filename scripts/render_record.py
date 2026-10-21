"""Regenerate the durable code/result record from recorded run state."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
from src.research_io import ROOT, now, output_path


def render():
    files = sorted((ROOT / "outputs").glob("*/summary.json"))
    env_file = ROOT / "provenance/environment.json"
    env = json.loads(env_file.read_text()) if env_file.exists() else {}
    lines = [
        "# FaceFlux-Research：代码与实验记录", "",
        f"更新：{now()}。此文档由运行状态自动整理，方案保存在 [研究方案](${FNFLOW_ROOT}/docs/研究方案.md)。", "",
        "阅读入口：[首轮诊断结论](${FNFLOW_ROOT}/docs/首轮诊断结论.md) · "
        "[数据来源初查](${FNFLOW_ROOT}/docs/数据来源初查.md)", "",
        "## 工作边界", "",
        "- 新工作目录：${FNFLOW_ROOT}；服务器对应 ${FNFLOW_ROOT}。",
        "- FaceFlux-Best 仅作为只读源码、权重与数据参考；所有新输出和缓存写入新目录。",
        "- 本轮先做 P0 协议检查及 D1–D5 无需重训诊断。FaceScape 是历史开发域。",
        "- 小样本、全量、历史 clamp 协议与浮点单位法线协议分列；推理 seed 变异不是训练 seed 变异。",
        "- 本轮没有进行新训练，尚不能支持通用域优越性或论文创新结论。", "",
        "## 运行环境", "",
        f"- Python：{env.get('python', '待检查')}",
        f"- PyTorch：{env.get('packages', {}).get('torch', '待检查')}；CUDA：{env.get('torch_cuda', '待检查')}",
        f"- GPU：{env.get('gpu', '待检查')}",
        "- [环境原始记录](${FNFLOW_ROOT}/provenance/environment.json)",
        "- [数据来源清单](${FNFLOW_ROOT}/provenance/data_inventory.json)", "",
        "- [Best 最终核对（含使用权重内容哈希）](${FNFLOW_ROOT}/provenance/best_after.json)",
        "- [历史结果复现核对](${FNFLOW_ROOT}/provenance/historical_reproduction.json)", "",
        "## 代码索引", "",
        "| 文件 | 职责 |", "|---|---|",
        "| [model_unet.py](${FNFLOW_ROOT}/src/models/model_unet.py) | 从 Best 原样复制的 UNet；结构与参数未改 |",
        "| [历史评测快照](${FNFLOW_ROOT}/scripts/legacy/eval_facescape_modes.py) | 原推理、mask 和指标的兼容对照，不直接导入 Best |",
        "| [flow_inference.py](${FNFLOW_ROOT}/src/flow_inference.py) | 正确 Euler、局部 RNG、no/fixed/Geom-CFG、FP32 单位法线和引导统计 |",
        "| [normal_geometry.py](${FNFLOW_ROOT}/src/normal_geometry.py) | 有效区域法线插值、镜像及连续坐标 K 变换 |",
        "| [losses.py](${FNFLOW_ROOT}/src/losses.py) | masked 速度与法线损失、raw endpoint unit 项、无效批次跳过 |",
        "| [run_diagnostics.py](${FNFLOW_ROOT}/scripts/run_diagnostics.py) | D0 历史兼容、D1–D5、逐图结果、失败与 Best 审计 |",
        "| [run_multiseed_diagnostics.py](${FNFLOW_ROOT}/scripts/run_multiseed_diagnostics.py) | 全量 8 seed 的单步 CFG 及 no-CFG 步数配对补做 |",
        "| [补做配置](${FNFLOW_ROOT}/configs/diagnostics_full_multiseed.json) | 固定参考权重、8 seed、4 步数与无筛选协议 |",
        "| [反馈核对与补实验](${FNFLOW_ROOT}/docs/反馈核对与补实验.md) | 修正解释边界、训练重复设计与发布范围记录 |",
        "| [test_protocol.py](${FNFLOW_ROOT}/tests/test_protocol.py) | 有明确几何/数值期望的协议测试 |",
        "| [pilot 配置](${FNFLOW_ROOT}/configs/diagnostics_pilot.json) | 固定 checkpoint、子集选择、seed 与步数 |",
        "| [preflight.py](${FNFLOW_ROOT}/scripts/preflight.py) | 环境、来源哈希和 Best 改动前清单 |",
        "| [summarize_diagnostics.py](${FNFLOW_ROOT}/scripts/summarize_diagnostics.py) | 由完成结果生成首轮结论并核对历史参考 |",
        "| [plot_diagnostics.py](${FNFLOW_ROOT}/scripts/plot_diagnostics.py) | 全量 CFG 与小样本步数/seed 图表 |",
        "| [verify_best.py](${FNFLOW_ROOT}/scripts/verify_best.py) | 最终检查 Best 条目与使用过的 checkpoint 内容 |", "",
        "## 基础检查", "",
    ]
    test_file = ROOT / "outputs/protocol_tests.txt"
    if test_file.exists():
        useful = [line for line in test_file.read_text(errors="replace").splitlines()
                  if "passed" in line or "failed" in line or "error" in line.lower()]
        lines.extend(["~~~text", *useful[-4:], "~~~", ""])
    else:
        lines.extend(["协议测试：待运行。", ""])
    for file in files:
        run = json.loads(file.read_text())
        lines.extend([f"## {run['run_id']}", "",
                      f"- 状态：{run['status']}；阶段：{run['stage']}；记录时间：{run.get('updated_at', '')}",
                      f"- 数据：{run.get('num_images', 0)} 张 / {run.get('num_subjects', 0)} 个身份；{run.get('scope', '')}",
                      f"- 当前任务：{run.get('current_task', '—')}",
                      "- GT mask 沿用人脸历史策略；浮点指标使用相同 mask，只隔离输出方向处理差异。",
                      f"- [完整结果](${FNFLOW_ROOT}/outputs/{run['run_id']}/summary.json) · "
                      f"[样本清单](${FNFLOW_ROOT}/outputs/{run['run_id']}/subset.json)", ""])
        if run.get("error"):
            lines.extend([f"运行错误：{run['error']}", ""])
        parity = run.get("parity", {})
        if parity:
            lines.extend([f"D0：历史输出最大绝对差 {parity.get('max_abs_encoded_difference')}；"
                          f"mask/指标角误差差 {parity.get('angular_mean_difference_deg')}°。", ""])
        lines.extend(["| 任务 | 模型 | 模式 | 步数 | seed | 历史 MAE° | 浮点 MAE° | 标准差：逐图 MAE | 记录耗时 ms/图 |",
                      "|---|---|---|---:|---|---:|---:|---:|---:|"])
        for row in run.get("rows", []):
            timing = "—" if row.get("timing_note") else f"{row.get('mean_ms', 0):.2f}"
            lines.append(
                f"| {row['task']} | {row['model']} | {row['mode']} | {row['steps']} | {row['seed']} | "
                f"{row['legacy']['mean']:.4f} | {row['float']['mean']:.4f} | "
                f"{row['float']['std_image_mean']:.4f} | {timing} |")
        lines.extend(["",
                      "耗时为缓存输入后、包含诊断统计的推理耗时，未包含数据读取和 mask 预处理；不是正式部署端到端基准。", ""])
        if run.get("multiseed"):
            lines.extend(["### 推理 seed 汇总", "",
                          "标准差在数据集 MAE 的 8 个推理 seed 间计算，采用样本标准差（ddof=1）；不代表训练 seed 方差。", "",
                          "| 模式与步数 | 已完成 seed | 浮点 MAE 均值° | seed 标准差° |",
                          "|---|---:|---:|---:|"])
            for name, group in run["multiseed"].get("groups", {}).items():
                score = group["protocols"]["float"]
                lines.append(f"| {name} | {group['num_seeds']} | {score['mean_mae_deg']:.4f} | {score['std_seed_ddof1_deg']:.4f} |")
            lines.extend(["", "只列已完成全部预定 seed 的组；完整步数矩阵是否完成："
                          + str(run["multiseed"].get("matrix_completed", False)) + "。", ""])
        if run.get("noise"):
            noise = run["noise"]
            lines.extend([f"D3：平均角度离散度 {noise['mean_angular_dispersion_deg']:.4f}°；"
                          f"逐像素离散度–误差相关 {noise['pearson_dispersion_error']}；"
                          f"相关的有效像素数 {noise['num_pixels']}。",
                          "这只是同一已训练模型的噪声敏感度，不证明多模态后验或校准不确定性。", ""])
        if run.get("ood"):
            ood = run["ood"]
            lines.extend([f"D5：{ood['num_images']} 张无 GT 通用图片；平均 s={ood['mean_guidance']:.4f}；"
                          f"s=0 像素比例={ood['mean_zero_fraction']:.2%}。",
                          "统计区域为整张图，包含背景；不据此声称通用几何准确度或有/无人脸结构。", ""])
        audit = run.get("best_audit")
        if audit:
            lines.extend([f"Best 审计：{'未检测到改动' if audit['verified_unchanged'] else '检测到差异，需检查'}；"
                          f"新增 {len(audit['added'])}，删除 {len(audit['removed'])}，变化 {len(audit['changed'])}。",
                          "审计范围：不跟随目录软链接的条目大小/修改时间/类型，以及代码、配置、文档内容哈希；"
                          "使用的 checkpoint 内容哈希另行记录。", ""])
        lines.extend(["初步观察："])
        baseline = next((row for row in run.get("rows", [])
                         if row["model"] == "p05_best" and row["steps"] == 1
                         and row["mode"] == "geom_cfg" and row["seed"] == 42), None)
        if baseline:
            delta = baseline["float"]["mean"] - baseline["legacy"]["mean"]
            lines.append(f"- 同一 raw 预测改用浮点单位方向后，MAE 变化为 {delta:+.4f}°；不涉及重训。")
        lines.extend(["- 仅解释已完成的表格行；未完成项不当作结果。",
                      "- 旧 checkpoint 之间的差距仍包含训练变异，不能仅靠此表推出 dropout 的独立因果收益。", ""])
    lines.extend(["## 下一步", "",
                  "先完成已启动的历史诊断并核对兼容误差；再做数据坐标与来源清点、通用训练/验证划分，"
                  "随后运行 R/V/Xv/Xn 参数化对照。当前 masked 损失是基础组件，尚未用于新训练。", "",
                  "脚本由固定配置启动；每完成一个配方写入 JSON/逐图 CSV 并刷新此文档。"
                  "进程失败会记录错误，重新运行可跳过已经验证签名一致的结果。", ""])
    target = output_path(ROOT / "docs/代码与实验记录.md")
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, target)


if __name__ == "__main__":
    render()
