# Your ViT is Secretly an Image Segmentation Model  
[![Papers with Code: SOTA on BRAVO (OOD)](https://paperswithcode.co/api/v1/papers/2503.19108/leaderboard-badge.svg?eval=640&live=1)](https://paperswithcode.co/benchmark/bravo-ood?task=image-segmentation&eval=640)
[![Papers with Code: SOTA on COCO 2017 Panoptic Segmentation](https://paperswithcode.co/api/v1/papers/2503.19108/leaderboard-badge.svg?eval=6260&live=1)](https://paperswithcode.co/benchmark/coco-2017-panoptic-segmentation?task=image-segmentation&eval=6260)


**CVPR 2025 ✨ Highlight** · [📄 Paper](https://arxiv.org/abs/2503.19108)

**[Tommie Kerssies](https://tommiekerssies.com)<sup>1</sup>, [Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ)<sup>2,*</sup>, [Alexander Hermans](https://scholar.google.de/citations?user=V0iMeYsAAAAJ)<sup>3</sup>, [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ)<sup>1</sup>, [Giuseppe Averta](https://www.giuseppeaverta.me/)<sup>2</sup>, [Bastian Leibe](https://scholar.google.com/citations?user=ZcULDB0AAAAJ)<sup>3</sup>, [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ)<sup>1</sup>, [Daan de Geus](https://ddegeus.github.io)<sup>1,3</sup>**

¹ Eindhoven University of Technology  
² Polytechnic of Turin  
³ RWTH Aachen University  
\* Work done while visiting RWTH Aachen University

## Overview

We present the **Encoder-only Mask Transformer (EoMT)**, a minimalist image segmentation model that repurposes a plain Vision Transformer (ViT) to jointly encode image patches and segmentation queries as tokens. No adapters. No decoders. Just the ViT.

Leveraging large-scale pre-trained ViTs, EoMT achieves accuracy similar to state-of-the-art methods that rely on complex, task-specific components. At the same time, it is significantly faster thanks to its simplicity, for example up to 4× faster with ViT-L.  

Turns out, *your ViT is secretly an image segmentation model*. EoMT shows that architectural complexity isn't necessary. For segmentation, a plain Transformer is all you need.

## 🚀 NEW: PMT 

Presenting our latest model, [PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders](https://arxiv.org/abs/2603.25398).

PMT reconciles EoMT minimal philosophy with the need of preserving the features of frozen Foundation Models, by mimicking the last layers of EoMT and VidEoMT with a simple and fast decoder.

Take a [look](https://github.com/tue-mps/pmt)!

## 🚀 NEW: VidEoMT 

🔥 We're pleased to present our latest CVPR 2026 paper, [VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](https://arxiv.org/abs/2602.17807).

VidEoMT extends EoMT philosophy to the temporal domain, introducing an encoder-only video segmentation model that is up to 10x faster than competitors.

Go check it [out](https://github.com/tue-mps/videomt)! 


## 🚀 NEW: DINOv3 Support

🔥 We're excited to announce support for **DINOv3** backbones! Our new DINOv3-based EoMT models deliver improved performance across all segmentation tasks:

- **Panoptic Segmentation**: Up to 58.9 PQ on COCO with EoMT-L at 1280×1280
- **Instance Segmentation**: Up to 49.9 mAP on COCO with EoMT-L at 1280×1280  
- **Semantic Segmentation**: Up to 59.5 mIoU on ADE20K with EoMT-L at 512×512

All of this, at the impressive speed of EoMT!

Check out our [DINOv3 Model Zoo](model_zoo/dinov3.md) for all available EoMT configurations and performance benchmarks.

Thanks to the [DINOv3](https://github.com/facebookresearch/dinov3) team for providing these powerful foundation models!

## 🤗 Transformers

EoMT with DINOv2 is also available on [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/model_doc/eomt). See available models [here](https://huggingface.co/models?library=transformers&other=eomt&sort=trending).

## Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n eomt python==3.13.2
conda activate eomt
python3 -m pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation

Download the datasets below depending on which datasets you plan to use.  
You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**COCO**
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
```

**ADE20K**
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xf annotations_instance.tar
zip -r -0 annotations_instance.zip annotations_instance/
rm -rf annotations_instance.tar
rm -rf annotations_instance
```

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.  

## Usage

### Training

To train EoMT from scratch, run:

```bash
python3 main.py fit \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

This command trains the `EoMT-L` model with a 640×640 input size on COCO panoptic segmentation using 4 GPUs. Each GPU processes a batch of 4 images, for a total batch size of 16. Switch to ```dinov3``` in the configuration path to enable the corresponding DINOv3 model.

✅ Make sure the total batch size is `devices × batch_size = 16`  
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

> This configuration takes ~6 hours on 4×NVIDIA H100 GPUs, each using ~26GB VRAM.

To fine-tune a pre-trained EoMT model, add:

```bash
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.  
> `--model.load_ckpt_class_head False` skips loading the classification head when fine-tuning on a dataset with different classes.

> **DINOv3 Models**: When using DINOv3-based configurations, the code expects delta weights relative to DINOv3 weights by default. To disable this behavior and use absolute weights instead, add `--model.delta_weights False`. 

### Evaluating

To evaluate a pre-trained EoMT model, run:

```bash
python3 main.py validate \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

This command evaluates the same `EoMT-L` model using 4 GPUs with a batch size of 4 per GPU.

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

A [notebook](inference.ipynb) is available for quick inference and visualization with auto-downloaded pre-trained models.

> **DINOv3 Models**: When using DINOv3-based configurations, the code expects delta weights relative to DINOv3 weights by default. To disable this behavior and use absolute weights instead, add `--model.delta_weights False`. 

## Inference Optimization & Profiling

This fork (`eomt_opt`) extends the base `validate` command with additional flags for compilation backend selection, warmup handling, and GPU profiling.

### Runtime flags

| Flag | Default | Description |
|------|---------|-------------|
| `--compile_backend` | `inductor` | `inductor` — compile with TorchInductor/Triton. `eager` — plain PyTorch, no compilation. |
| `--compile_mode` | `default` | `default`, `reduce-overhead` (CUDA graphs), or `max-autotune` (kernel search). Ignored when `eager`. |
| `--warmup_batches` | `1` | Batches excluded from inference timing to skip first-batch compilation overhead. Set to `0` to include all batches. |
| `--inductor_cache_dir` | none | Directory to save/load compiled Triton kernels. Eliminates recompilation on subsequent runs. |
| `--cpu` | off | Force CPU execution, bypassing the GPU entirely. Useful as a compute baseline. |
| `--profile_batches` | `0` | Number of batches to profile with `torch.profiler` after warmup. `0` disables profiling. |
| `--profile_output_dir` | `./profile_output` | Directory for Chrome trace files produced by `torch.profiler`. |

### Profiling: torch.profiler (in-process, op-level)

Adds `--profile_batches N` to record CPU and GPU kernel timings for `N` batches after the warmup period. Produces:

- A **top-20 ops table** printed to stdout, sorted by GPU time.
- A **Chrome trace** saved to `--profile_output_dir`, viewable at [https://ui.perfetto.dev](https://ui.perfetto.dev).

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
python3 main.py validate \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 1 --data.batch_size 1 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin \
  --compile_backend eager \
  --warmup_batches 1 \
  --profile_batches 3 \
  --profile_output_dir ./profile_output/eomt_large_640_eager \
  --trainer.logger False
```

The profiler schedule skips `--warmup_batches` batches (to avoid capturing compilation overhead), warms up for one additional batch, then actively records for `--profile_batches` batches.

### Profiling: rocprofv3 (external, hardware counters)

For hardware-level GPU utilisation metrics (occupancy, memory bandwidth, cache efficiency), wrap the command with `rocprofv3`. No code changes are required.

On gfx1151, the PMU hardware limit requires splitting into two passes. `FETCH_SIZE`/`WRITE_SIZE` are also unavailable on this iGPU — use raw `GL2C` counters instead.

```bash
# Pass 1 — kernel trace + occupancy counters
HSA_OVERRIDE_GFX_VERSION=11.5.1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
rocprofv3 --kernel-trace \
  --pmc GPUBusy OccupancyPercent MemUnitBusy WriteUnitStalled \
  -d ./rocprof_pass1 -f csv \
  -- python3 main.py validate \
      -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
      --model.network.masked_attn_enabled False \
      --compile_backend eager --trainer.limit_val_batches 10 \
      --trainer.logger False ...

# Pass 2 — roofline counters (no --kernel-trace; timing comes from pass 1)
HSA_OVERRIDE_GFX_VERSION=11.5.1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
rocprofv3 \
  --pmc VALUInsts SQ_WAVES_sum GL2C_MC_RDREQ_sum GL2C_MC_WRREQ_sum \
  -d ./rocprof_pass2 -f csv \
  -- python3 main.py validate \
      -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
      --model.network.masked_attn_enabled False \
      --compile_backend eager --trainer.limit_val_batches 10 \
      --trainer.logger False ...

# Merge both passes into one summary (replace <hostname> and <pid2> as appropriate):
python3 rocprof_summary.py ./rocprof_pass1/<hostname>/ \
  --extra-counters ./rocprof_pass2/<hostname>/<pid2>_counter_collection.csv
```

Key counters and what they indicate:

| Counter | What it measures | Target |
|---------|-----------------|--------|
| `GPUBusy` | % of time the GPU has work dispatched | > 85% |
| `OccupancyPercent` | % of peak wavefront occupancy across all CUs | > 60% |
| `MeanOccupancyPerCU` | Average active wavefronts per Compute Unit | Higher is better |
| `MemUnitBusy` | % of time the memory subsystem is active | Compare to `GPUBusy` |
| `WriteUnitStalled` | % of time the write unit is stalled | < 10% |
| `FETCH_SIZE` | KB fetched from DRAM (L2 cache misses) | — |
| `WRITE_SIZE` | KB written back to DRAM | — |
| `VALUInsts` | Avg vector ALU instructions per work-item | — |
| `Wavefronts` | Total wavefronts launched | — |
| `L2CacheHit` | % of memory requests served from L2 cache | > 80% ideal |

**Note on gfx1151 (iGPU):** `FETCH_SIZE` and `WRITE_SIZE` are not collectible on this GPU — they are derived metrics that require TCC/EA hardware blocks absent from the unified-memory iGPU design. Use `GL2C_MC_RDREQ_sum` and `GL2C_MC_WRREQ_sum` instead (raw L2→DRAM request counts; each request is assumed to be one 64-byte cache line). The `rocprof_summary.py` script automatically detects which memory counter source is present.

When `GL2C_MC_RDREQ_sum`, `GL2C_MC_WRREQ_sum`, `VALUInsts`, and `Wavefronts` (or `SQ_WAVES_sum`) are all present, `rocprof_summary.py` automatically adds roofline columns (**DRAM GB/s**, **BW util%**, **AI F/B**, **L2Hit%**, **Bound**) to the output table. The script compares each kernel's arithmetic intensity against the hardware ridge point (≈ 870 FLOP/byte for gfx1151) to classify kernels as `MEMORY` or `COMPUTE`-bound.

Low `OccupancyPercent` combined with low `GPUBusy` typically indicates GPU underutilisation — most commonly caused by insufficient batch size. If `MemUnitBusy` is close to `GPUBusy`, the workload is memory-bandwidth bound rather than compute bound. For proper roofline diagnosis, use the arithmetic intensity (AI) column: kernels below 870 FLOP/byte are definitively memory-bound regardless of `MemBusy%`.

> **Note on iGPU (Radeon 8060S / gfx1151):** Use `HSA_OVERRIDE_GFX_VERSION=11.5.1` to target the native gfx1151 kernels in ROCm 7.2.4, and `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` to enable AOTriton flash attention for this GPU. Without the AOTriton flag, `F.scaled_dot_product_attention` falls back to a naive O(n²) implementation that is ~8× slower.

## Model Zoo

We provide pre-trained weights for both DINOv2- and DINOv3-based EoMT models.

- **[DINOv2 Models](model_zoo/dinov2.md)** - Original published results and pre-trained weights.
- **[DINOv3 Models](model_zoo/dinov3.md)** - New DINOv3-based models and pre-trained weights.

## Citation
If you find this work useful in your research, please cite it using the BibTeX entry below:

```BibTeX
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccol\`{o} and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and {de Geus}, Daan},
  title     = {{Your ViT is Secretly an Image Segmentation Model}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

## Acknowledgements

This project builds upon code from the following libraries and repositories:

- [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0 License)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache-2.0 License)  
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (Apache-2.0 License)  
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (Apache-2.0 License)  
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache-2.0 License)
- [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)
