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

AMBER 生成式幻觉评测需要把官方 AMBER `data/` 目录和图片目录放好，默认路径为：

- `/home/apulis-dev/userdata/AMBER/data`
- `/home/apulis-dev/userdata/AMBER/images`

M-HalDetect 评测需要官方 `train_raw.json` / `val_raw.json` 和 COCO 2014 val 图片，默认路径为：

- `/home/apulis-dev/userdata/M-HalDetect`
- `/home/apulis-dev/userdata/val2014`

AMBER Yes/No 判别式评测使用官方 discriminative query 和 truth，默认路径为：

- `/home/apulis-dev/userdata/AMBER/data`
- `/home/apulis-dev/userdata/AMBER/images`

也可以显式指定：

```bash
python -m dgst.cli build-cache \
  --dataset amber \
  --dataset-root /path/to/AMBER/images \
  --annotation-path /path/to/AMBER/data
```

```bash
python -m dgst.cli build-cache --dataset mhaldetect
```

```bash
python -m dgst.cli build-cache --dataset amber_discriminative
```

```bash
python -m dgst.cli build-cache --dataset amber_discriminative_paired
```

zero-shot 评测：

```bash
python -m dgst.cli evaluate \
  --dataset coco \
  --gpus 0,1 \
  --num-data 100 \
  --auto-name
```

AMBER 只接入生成式任务（`AMBER_1.jpg` 到 `AMBER_1004.jpg`）：

```bash
python -m dgst.cli evaluate \
  --dataset amber \
  --dataset-root /path/to/AMBER/images \
  --annotation-path /path/to/AMBER/data \
  --num-data 100 \
  --auto-name
```

M-HalDetect 使用 benchmark 自带 response 做 teacher-forced forward，并按人工 span 标签
`ACCURATE` / `INACCURATE` 做 segment-level 幻觉检测：

```bash
python -m dgst.cli evaluate \
  --dataset mhaldetect \
  --gpus 0,1 \
  --num-data 3164 \
  --auto-name
```

AMBER Yes/No 判别式任务会让模型生成 Yes/No 回答，并对生成回答中的
Yes/No token 做 DGST probe 特征；标签为 `1` 表示该回答与 AMBER truth 不一致，即幻觉正样本：

```bash
python -m dgst.cli export-probe \
  --dataset amber_discriminative \
  --gpus 0,1 \
  --num-data 14216 \
  --max-new-tokens 8 \
  --run-name amber_discriminative_yesno_probe6_full
```

如果要复现 reference-free 论文式的 paired Yes/No 设置，使用
`amber_discriminative_paired`。每个 AMBER query 会导出两条样本：固定
`Yes` 和固定 `No`，其中与 truth 不一致的一条是幻觉正样本；`--num-data`
表示原始 query 数，导出的 probe 样本数会翻倍：

```bash
python -m dgst.cli export-probe \
  --dataset amber_discriminative_paired \
  --gpus 0,1 \
  --num-data 5000 \
  --max-new-tokens 8 \
  --run-name amber_discriminative_paired_probe6_prompt_5000q
```

论文里的训练/验证比例约为 2:1，可以在训练时显式设置
`--test-size 0.3333333333333333`：

```bash
python -m dgst.cli train-probe \
  --dataset-file probe_data/amber_discriminative_paired_probe6_prompt_5000q/probe_dataset.jsonl \
  --run-name amber_discriminative_paired_probe6_prompt_5000q_train \
  --feature-set probe6_prompt \
  --split-by group \
  --test-size 0.3333333333333333 \
  --paper-config
```

如果要更严格地避免同一张图的不同 query 同时出现在 train/val，可以改成
`--split-by image`。这会按真实图片路径分组切分。

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

probe 训练默认以 `hallucinated=1` 作为正样本，输出中的
`hallucination_probability` 表示模型判为幻觉的概率。
如果要把 prompt-aware token cosine 拼进 probe 输入，可以使用：

```bash
python -m dgst.cli train-probe \
  --dataset-file probe_data/amber_discriminative_yesno_probe6_full/probe_dataset.jsonl \
  --run-name amber_discriminative_yesno_probe6_prompt_train \
  --feature-set probe6_prompt \
  --paper-config
```

`probe6_prompt` = 原 `probe6` 每层 6 个 DGST 特征 + 每层
`prompt_last_cosine` / `prompt_mean_cosine` 两个 prompt-aware 特征。

评测 probe：

```bash
python -m dgst.cli eval-probe \
  --probe-run coco_dgst_probe_500_v1 \
  --dataset-file probe_data/coco_dgst_probe_500/probe_dataset.jsonl \
  --input-format probe_dataset
```
