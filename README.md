# DGST Standalone Project

`DGST` 是一个独立的对象级幻觉检测项目，当前可以在不依赖 `vicr.*` 运行时代码的前提下，独立完成以下链路：

- 单图分析
- 数据集 zero-shot 评测
- probe 数据导出
- probe 训练
- probe 评测
- 数据集 cache 构建

## Entry Points

```bash
python -m dgst.cli single
python -m dgst.cli evaluate
python -m dgst.cli export-probe
python -m dgst.cli train-probe
python -m dgst.cli eval-probe
python -m dgst.cli build-cache
```

## Install

```bash
pip install -r requirements.txt
```

## Defaults

- 默认本地模型：
  - `/home/apulis-dev/userdata/models/llava/llava-hf/llava-1___5-7b-hf`
- 默认 `local_files_only=True`
- 默认 DGST 参数：
  - `tau=0.07`
  - `transport_top_k=32`
  - `gamma1=1.0`
  - `gamma2=1.0`
  - `baseline_layers=10`
  - `risk_start_layer=15`
  - `alpha=2.0`
  - `ot_solver=linprog`

## Outputs

默认输出都写入仓库自己的工作目录：

- `outputs/`
- `analysis_plots/`
- `probe_data/`
- `probe_runs/`
- `probe_evals/`
- `experiment_log.jsonl`
- `coco_ground_truth.json`
- `chair.pkl`

## Example Workflow

构建 cache：

```bash
python -m dgst.cli build-cache --dataset coco
```

zero-shot 评测：

```bash
python -m dgst.cli evaluate \
  --dataset coco \
  --gpus 0,1 \
  --num-data 100 \
  --auto-name
```

导出 probe 数据：

```bash
python -m dgst.cli export-probe \
  --dataset coco \
  --gpus 0,1 \
  --num-data 500 \
  --run-name coco_dgst_probe_500
```

训练 probe：

```bash
python -m dgst.cli train-probe \
  --dataset-file probe_data/coco_dgst_probe_500/probe_dataset.jsonl \
  --run-name coco_dgst_probe_500_v1
```

评测 probe：

```bash
python -m dgst.cli eval-probe \
  --probe-run coco_dgst_probe_500_v1 \
  --dataset-file probe_data/coco_dgst_probe_500/probe_dataset.jsonl \
  --input-format probe_dataset
```
