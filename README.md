# semantic_jscc_lab

`semantic_jscc_lab` 是用于完成 CIFAR-10 上 AE / Deep JSCC 课程实验的 Python 项目。当前版本提供数据加载、AE baseline、可复现实验工具、重建指标、绘图函数和命令行入口；后续可以在此基础上继续添加带噪 Deep JSCC 训练和测试脚本。

## 数据格式

数据加载器支持以下输入：

- `.npz` 文件，包含 `data` 和 `labels` 两个数组。
- CIFAR-10 pickle batch 字典，例如 `data_batch_1`、`test_batch`。
- 目录路径，会自动扫描 CIFAR batch 文件。

原始 `data` 必须是 `shape=(N, 3072)`、`dtype=uint8` 的数组；`labels` 是长度为 `N` 的 0-9 标签。每行前 1024 个值是 R，接着 1024 个值是 G，最后 1024 个值是 B，每个通道 row-major。加载后统一转换为 `(N, 3, 32, 32)` 的 `float32`，范围 `[0, 1]`。

默认数据路径写在 `configs/default.yaml`：

```yaml
data:
  data_path: null
```

为了避免把本机路径写死在代码中，正式实验请通过 `--data_path` 显式传入数据路径。

## 环境

使用本机 `mypytorch` conda 环境：

```bash
conda activate mypytorch
```

项目不会从外网下载数据。

## 目录

```text
configs/default.yaml      # 实验默认配置
jscc_lab/data.py          # CIFAR array / pickle / directory loader
jscc_lab/channel.py       # power normalization、SNR/noise std、AWGN
jscc_lab/models.py        # ConvEncoder、ConvDecoder、AutoEncoder
jscc_lab/utils.py         # seed、device、日志、JSON、checkpoint 工具
jscc_lab/metrics.py       # MSE、PSNR、重建评估
jscc_lab/plotting.py      # 训练曲线、图像网格、PSNR、率失真曲线
jscc_lab/cli.py           # 后续脚本复用的通用 CLI 参数
scripts/check_data.py     # 最小数据加载检查命令
scripts/train_ae.py       # 作业任务 (1) AE baseline 训练脚本
scripts/visualize_ae_latent.py       # 作业任务 (2) 原图/重建图和 latent heatmap
scripts/gaussian_latent_sampling.py  # 作业任务 (3) latent 高斯统计与采样生成
scripts/perturb_latent.py            # 作业任务 (4) latent 高斯扰动重建
scripts/train_jscc.py                # 作业任务 (5) 抗噪 Deep JSCC 训练
scripts/eval_jscc_train_snr_models.py # 作业任务 (5) SNR-PSNR 批量评估
scripts/eval_noise_sweep.py          # 作业任务 (6) 固定 SNR=7 模型测试噪声扫描
scripts/eval_rate_distortion.py      # 作业任务 (7) prefix Kp 率失真评估
scripts/run_all_non_innovation.py    # 一键生成所有非创新 baseline 输出
scripts/check_outputs.py             # 检查报告所需关键输出是否存在
tests/smoke_test.py       # 自动生成 tiny_dataset.npz 的冒烟测试
```

## 一键运行全部 Baseline

本项目没有实现创新策略，只完成作业必需的 AE / Deep JSCC baseline、SNR 扫描和 prefix Kp 率失真分析。

完整运行：

```bash
python scripts/run_all_non_innovation.py \
  --data_path /path/to/cifar_data.npz \
  --out_dir outputs/full_run \
  --ae_epochs 100 \
  --jscc_epochs 100 \
  --batch_size 128 \
  --seed 42 \
  --device cuda
```

quick 模式会使用极小样本和 1 epoch 快速跑通全流程：

```bash
python scripts/run_all_non_innovation.py \
  --data_path /path/to/cifar_data.npz \
  --out_dir outputs/full_run \
  --quick \
  --device cuda
```

运行结束后会写入：

```text
outputs/full_run/results_manifest.json
```

也可以单独验收关键输出：

```bash
python scripts/check_outputs.py --out_dir outputs/full_run --write_manifest
```

一键流程的主要输出清单：

```text
outputs/full_run/ae/model_summary.txt
outputs/full_run/ae/latent_shape.txt
outputs/full_run/ae/recon_10.png
outputs/full_run/ae/latent_heatmap_img_*.png
outputs/full_run/gaussian/gaussian_stats.npz
outputs/full_run/gaussian/gaussian_stats_summary.txt
outputs/full_run/gaussian/generated_from_gaussian_10.png
outputs/full_run/perturb/perturb_recon_10.png
outputs/full_run/jscc/task5_snr_psnr.csv
outputs/full_run/jscc/task5_snr_psnr_curve.png
outputs/full_run/task6/task6_noise_sweep.csv
outputs/full_run/task6/task6_psnr_vs_snr.png
outputs/full_run/task7/task7_rate_distortion.csv
outputs/full_run/task7/task7_rate_distortion_curve.png
```

## 任务 (1)：训练 AE Baseline

模型结构满足作业要求：输入为 `(N, 3, 32, 32)`，encoder 的 PyTorch latent tensor 为 `(N, 16, 8, 8)`，对应报告中的 latent code 形状 `8 x 8 x 16`；decoder 输出 `(N, 3, 32, 32)`，最后使用 `Sigmoid` 限制像素范围为 `[0, 1]`。

正式训练示例：

```bash
python scripts/train_ae.py --data_path /path/to/cifar_data.npz --out_dir outputs/ae --epochs 100 --batch_size 128 --device cuda
```

快速测试：

```bash
python scripts/train_ae.py --data_path tests/tiny_dataset.npz --out_dir outputs/debug_ae --epochs 1 --max_train_samples 128 --max_eval_samples 64 --device cuda
```

训练脚本会保存：

```text
outputs/ae/best_ae.pt
outputs/ae/last_ae.pt
outputs/ae/model_summary.txt
outputs/ae/latent_shape.txt
outputs/ae/ae_history.csv
outputs/ae/ae_train_loss.png
```

## 任务 (2)：原图、重建图与 Latent 可视化

该脚本加载 `scripts/train_ae.py` 训练出的 `best_ae.pt`，按同一套 split 从 test split 中固定 seed 抽样 10 张图，保存原图/重建图和每张图的 16 通道 latent heatmap。

```bash
python scripts/visualize_ae_latent.py \
  --data_path /path/to/cifar_data.npz \
  --checkpoint outputs/ae/best_ae.pt \
  --out_dir outputs/ae/task2_latent \
  --seed 42 \
  --device cuda
```

输出包括：

```text
outputs/ae/task2_latent/recon_10.png
outputs/ae/task2_latent/latent_heatmap_img_*.png
outputs/ae/task2_latent/selected_indices.txt
```

## 任务 (3)：Latent 高斯统计与采样生成

该脚本固定 seed 从 test split 抽样 256 张图，计算每个空间位置 `(h,w)` 上 16 维 latent 向量的均值、方差和协方差，并从 8x8 个 16D Gaussian 中独立采样 10 组 latent 送入 decoder。

```bash
python scripts/gaussian_latent_sampling.py \
  --data_path /path/to/cifar_data.npz \
  --checkpoint outputs/ae/best_ae.pt \
  --out_dir outputs/ae/task3_gaussian \
  --seed 42 \
  --device cuda
```

输出包括：

```text
outputs/ae/task3_gaussian/gaussian_stats.npz
outputs/ae/task3_gaussian/gaussian_stats_summary.txt
outputs/ae/task3_gaussian/mean_var_overview.png
outputs/ae/task3_gaussian/generated_from_gaussian_10.png
```

## 任务 (4)：Latent 高斯扰动

可以复用任务 (2) 的 `selected_indices.txt`，保证三列对比图中的样本与任务 (2) 一致：

```bash
python scripts/perturb_latent.py \
  --data_path /path/to/cifar_data.npz \
  --checkpoint outputs/ae/best_ae.pt \
  --out_dir outputs/ae/task4_perturb \
  --selected_indices outputs/ae/task2_latent/selected_indices.txt \
  --noise_std 0.1 \
  --seed 42 \
  --device cuda
```

也可以用 SNR 指定扰动强度，`--snr_db` 会覆盖 `--noise_std`：

```bash
python scripts/perturb_latent.py \
  --data_path /path/to/cifar_data.npz \
  --checkpoint outputs/ae/best_ae.pt \
  --out_dir outputs/ae/task4_perturb_snr7 \
  --selected_indices outputs/ae/task2_latent/selected_indices.txt \
  --snr_db 7 \
  --seed 42 \
  --device cuda
```

输出包括：

```text
outputs/ae/task4_perturb/perturb_recon_10.png
outputs/ae/task4_perturb/perturb_selected_indices.txt
outputs/ae/task4_perturb/perturb_settings.txt
```

## 任务 (5)：抗噪 Deep JSCC

Deep JSCC 使用与 AE 相同的 encoder/decoder 尺寸，latent code 仍为 `(N, 16, 8, 8)`，对应报告中的 `8 x 8 x 16`。训练流程为：

```text
x -> encoder -> power_normalize(P=1) -> AWGN(train_snr_db) -> decoder -> MSELoss
```

代码中明确使用 `noise_var = 10 ** (-snr_db / 10)`，`noise_std = sqrt(noise_var)`；`torch.randn_like` 乘的是标准差 `noise_std`，不是方差。

五个 SNR 模型分别训练：

```bash
python scripts/train_jscc.py --data_path /path/to/cifar_data.npz --train_snr_db 1 --out_dir outputs/jscc/snr_1 --epochs 100 --batch_size 128 --device cuda

python scripts/train_jscc.py --data_path /path/to/cifar_data.npz --train_snr_db 4 --out_dir outputs/jscc/snr_4 --epochs 100 --batch_size 128 --device cuda

python scripts/train_jscc.py --data_path /path/to/cifar_data.npz --train_snr_db 7 --out_dir outputs/jscc/snr_7 --epochs 100 --batch_size 128 --device cuda

python scripts/train_jscc.py --data_path /path/to/cifar_data.npz --train_snr_db 13 --out_dir outputs/jscc/snr_13 --epochs 100 --batch_size 128 --device cuda

python scripts/train_jscc.py --data_path /path/to/cifar_data.npz --train_snr_db 19 --out_dir outputs/jscc/snr_19 --epochs 100 --batch_size 128 --device cuda
```

每个目录会保存：

```text
outputs/jscc/snr_7/best_jscc_snr7.pt
outputs/jscc/snr_7/last_jscc_snr7.pt
outputs/jscc/snr_7/history.csv
outputs/jscc/snr_7/loss_curve.png
outputs/jscc/snr_7/model_summary.txt
```

批量评估五个模型，并绘制训练 SNR 与平均 PSNR 曲线：

```bash
python scripts/eval_jscc_train_snr_models.py --data_path /path/to/cifar_data.npz --ckpt_dir outputs/jscc --out_dir outputs/jscc --device cuda
```

输出：

```text
outputs/jscc/task5_snr_psnr.csv
outputs/jscc/task5_snr_psnr_curve.png
```

## 任务 (6)：固定 SNR=7 模型的测试噪声扫描

该脚本只加载训练 SNR=7 dB 的 Deep JSCC 模型，固定 seed 从 test split 中随机抽取同一批图片，所有测试 SNR 共用这批图片。计时中 encoder 时间包含 `encoder + power_normalize`，decoder 时间只包含 `decoder forward`，AWGN 加噪不计入二者；CUDA 计时会在开始和结束前后同步。

```bash
python scripts/eval_noise_sweep.py --data_path /path/to/cifar_data.npz --ckpt outputs/jscc/snr_7/best_jscc_snr7.pt --out_dir outputs/task6 --num_images 500 --device cuda
```

快速调试可以减少样本数：

```bash
python scripts/eval_noise_sweep.py --data_path tests/tiny_dataset.npz --ckpt outputs/debug_jscc/snr_7/best_jscc_snr7.pt --out_dir outputs/debug_task6 --num_images 32 --batch_size 16 --device cuda
```

输出：

```text
outputs/task6/task6_noise_sweep.csv
outputs/task6/task6_psnr_vs_snr.png
outputs/task6/task6_summary.txt
outputs/task6/task6_selected_indices.txt
```

## 任务 (7)：Prefix Kp 率失真曲线

该脚本只实现题目指定 baseline：先得到 power-normalized latent code，PyTorch shape 为 `(N, 16, 8, 8)`，按题目定义转为 `(N, 8, 8, 16)` 后展平为长度 1024，只保留前 `Kp` 个元素，其余置 0，再 reshape 回 `(N, 8, 8, 16)` 并 permute 回 `(N, 16, 8, 8)`。所有 Kp 共用同一批固定 seed 抽取的测试图片，并在 `--test_snr_db 7` 下加 AWGN 后送入 decoder。

```bash
python scripts/eval_rate_distortion.py --data_path /path/to/cifar_data.npz --ckpt outputs/jscc/snr_7/best_jscc_snr7.pt --out_dir outputs/task7 --num_images 500 --device cuda
```

快速调试可以减少样本数：

```bash
python scripts/eval_rate_distortion.py --data_path tests/tiny_dataset.npz --ckpt outputs/debug_jscc/snr_7/best_jscc_snr7.pt --out_dir outputs/debug_task7 --num_images 32 --batch_size 16 --device cuda
```

输出：

```text
outputs/task7/task7_rate_distortion.csv
outputs/task7/task7_rate_distortion_curve.png
outputs/task7/task7_summary.txt
outputs/task7/task7_selected_indices.txt
```

## 快速检查

在项目根目录运行：

```bash
python tests/smoke_test.py
```

也可以检查真实 CIFAR-10 目录：

```bash
python scripts/check_data.py \
  --data_path /path/to/cifar_data.npz \
  --out_dir runs/check_data \
  --seed 42 \
  --device auto
```

## 后续脚本入口约定

后续训练和测试脚本建议复用 `jscc_lab.cli.build_common_parser()`，统一支持：

```text
--seed
--device
--data_path
--out_dir
```

配置中已经包含作业要求的 SNR 列表 `[1, 4, 7, 13, 19]`、latent shape `8 x 8 x 16`，以及 Kp 列表 `[128, 256, 384, 512, 640, 768, 896, 1024]`。
