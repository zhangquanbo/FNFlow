from pathlib import Path
import json,statistics,datetime,hashlib
R=Path(__file__).resolve().parents[1];O=R/'outputs/T6_S1_locked_20261112/timing'
def read(p):return json.loads(Path(p).read_text())
def write(p,a):Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
reg=read(O/'registration_private.json');cases={};native={};optimized={}
for job in reg['jobs']:
 p=O/'cases'/job['case']/'summary.json';a=read(p);assert a['status']=='completed' and a['CPU_affinity']==[8,9,10,11] and not a['telemetry_summary']['training_seen'];assert a['primary']['warmup']==5 and a['primary']['timed']==30;assert not a['primary']['global_training_scan_before']['matches'] and not a['primary']['global_training_scan_after']['matches'];cases[job['case']]=a
for name in ['S1','MoGe-S','MoGe-B','E1_B']:
 group=[cases[f'{name}_native_rep{n}'] for n in range(3)];med=[a['primary']['wall_median_ms'] for a in group];evt=[a['primary']['GPU_event_median_ms'] for a in group];native[name]={'wall_median_of_process_medians_ms':statistics.median(med),'wall_process_medians_ms':med,'GPU_event_median_of_process_medians_ms':statistics.median(evt),'GPU_event_process_medians_ms':evt,'between_process_range_ms':[min(med),max(med)],'CPU_four_core_percent':[a['primary']['telemetry']['CPU_four_core_percent']['mean'] for a in group],'GPU_power_W':[a['primary']['telemetry']['GPU_power_W']['mean'] for a in group],'GPU_temp_C_range':[min(a['primary']['telemetry']['GPU_temp_C']['min'] for a in group),max(a['primary']['telemetry']['GPU_temp_C']['max'] for a in group)],'GPU_SM_MHz_range':[min(a['primary']['telemetry']['GPU_SM_MHz']['min'] for a in group),max(a['primary']['telemetry']['GPU_SM_MHz']['max'] for a in group)]}
for name in ['S1','MoGe-S']:
 a=cases[f'{name}_compile_channels_last_rep0'];optimized[name]={'wall_median_ms':a['primary']['wall_median_ms'],'GPU_event_median_ms':a['primary']['GPU_event_median_ms'],'parity':a['numerical_parity'],'compile_first_call_seconds_excluded':a['compile_first_call_seconds_excluded']}
controls={}
for name in ['E1_B','MoGe-S']:
 arr=cases[f'{name}_native_rep0']['affinity_control'];aa=[x['wall_median_ms'] for x in arr if 'pinned_' in x['label']];bb=[x['wall_median_ms'] for x in arr if 'legacy_all_' in x['label']];controls[name]={'ABBA_blocks':[{k:v for k,v in b.items() if k in ['label','wall_median_ms','GPU_event_median_ms','CPU_affinity','telemetry']} for b in arr],'pinned_median_ms':statistics.median(aa),'all_allowed_median_ms':statistics.median(bb),'mask_change_fraction':statistics.median(bb)/statistics.median(aa)-1,'interpretation':'separatecontrolledmaskcomparison,notmainrow;doesnotbyitselfprovehistoricalrootcause'}
old=read(O/'old_formal_efficiency_snapshot_private.json');legacy_e=read(O/'old_legacy_E1_B_snapshot_private.json');legacy_s=read(O/'old_legacy_MoGe-S_snapshot_private.json');parts=cases['S1_native_rep0']['components'];assert parts['head_output_abs_difference']==0
result={'status':'completed','protocol':reg,'native':native,'compile_channels_last':optimized,'S1_components':parts,'affinity_controls':controls,'old_formal_efficiency':old,'old_legacy_E1_B':legacy_e,'old_legacy_MoGe_S':legacy_s,'cases':cases,'cause_conclusion':'CPUaffinityeffectquantifiedbyABBA;noactivebackgroundtrainingorIOinmeasure;wallandeventreported;historicalrootcausecannotbeclaimedwithoutadditionalhistoricalruntimeevidence','all_old_and_new_retained':True,'no_recipe_or_weight_change':True,'no_fullGSO_accuracy_claim_for_compile':True,'test_images_scored':0,'updated_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()};write(O/'summary.json',result)
lines=['# S1锁定后：全机无训练、CPU绑定测速','','固定CPU8–11核，监控核12；每个原生模型3个独立进程，各5次预热30次计时。汇总取三次进程中位数的中位数，不挑最快。所有主测窗口全机训练入口扫描为空，3090独占；采样原始CPU/NVML状态在每个case的telemetry_private。','','|模型|墙钟中位ms|GPU事件中位ms|三次墙钟中位ms|','|---|---:|---:|---|']
for name,a in native.items():lines.append(f'|{name}|{a["wall_median_of_process_medians_ms"]:.4f}|{a["GPU_event_median_of_process_medians_ms"]:.4f}|'+', '.join(f'{v:.4f}' for v in a['wall_process_medians_ms'])+'|')
lines+=['','|compile+channels_last，单独变体|墙钟ms|GPU事件ms|编译首调用秒（未计入稳态）|','|---|---:|---:|---:|']
for n,a in optimized.items():lines.append(f'|{n}|{a["wall_median_ms"]:.4f}|{a["GPU_event_median_ms"]:.4f}|{a["compile_first_call_seconds_excluded"]:.2f}|')
lines+=['','优化仅改变执行方式，不改锁定权重/配方；单图数值差见summary，不把优化延迟与原生GSO精度默默混成一行。','','S1组件：']
for label,key in [('编码器（含518预处理）','standalone_encoder'),('预测头（UNet+融合，缓存中间特征）','standalone_prediction_head')]:
 a=parts[key];lines.append(f'- {label}：墙钟{a["wall_median_ms"]:.4f}ms，GPU事件{a["GPU_event_median_ms"]:.4f}ms。')
lines+=['- 完整前向内GPU阶段事件中位数：'+str(parts['stage_GPU_medians_ms'])+'。缓存特征只用于单独头部计时；完整主测始终运行编码器。','','CPU/功耗/温度：']
for n,a in native.items():lines.append(f'- {n}：4核CPU占用各进程平均{a["CPU_four_core_percent"]}%；GPU功耗平均{a["GPU_power_W"]}W，温度{a["GPU_temp_C_range"]}℃，SM频率{a["GPU_SM_MHz_range"]}MHz。原始样本与限频位全部保留。')
lines+=['','差异原因复核：']
for n,a in controls.items():lines.append(f'- {n} ABBA固定核/原全核掩码对照：固定核{a["pinned_median_ms"]:.4f}ms，全核{a["all_allowed_median_ms"]:.4f}ms，变化{a["mask_change_fraction"]*100:.2f}%。该对照隔离本次CPU掩码效应，不等于证明整个历史差异的唯一根因。')
lines+=['- 旧E1_B38.08ms/MoGe-S33.91ms、上一轮原生计时、独立复核均保存于JSON；不覆盖旧记录。若本次仍不能复现旧低值，继续明确标注根因未知，不归因给未经证实的网络/读写或训练并发。','- 事件与墙钟每轮都记录；CPU占用、GPU功耗/温度/频率/限频位可按测量窗口对齐。数据支持的因素与未确认原因分开报告。','- CUDA输入/噪声/t均预载入；排除PNG读写、CPU/GPU输入传输、噪声生成与首次编译；TF32关闭，原生S1/E1_B为BF16 autocast，MoGe为FP16 autocast，参数FP32。','','模型：锁定S1 EMA检查点ebf6a113cd9e6a15edba0e1282fd469550f8d33f33012e8818e11b8dcfda9822。GSO-test评分0，配方/权重未改；训练队列将在本轮完整测量结束后启动。'];(O/'REPORT.md').write_text('\n'.join(lines)+'\n');print('LOCKED_PINNED_TIMING_COMPLETE',json.dumps({n:a['wall_median_of_process_medians_ms'] for n,a in native.items()}))
