#!/usr/bin/env python3
# ---------------------------------------------------------------
# Video inference test script for EoMT.
# Merges the validate inference path from main.py with the model
# loading approach from inference.ipynb.
#
# Reads a 1280x720 video stream (real device or synthetic),
# center-crops/resizes to 720x720, runs panoptic segmentation,
# and writes coloured output frames to stdout or a file.
# Latency (capture → output) is measured and printed per frame.
# ---------------------------------------------------------------

import argparse
import importlib
import logging
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.amp.autocast_mode import autocast


# ---------------------------------------------------------------------------
# Argument parsing — mirrors the relevant validate flags from main.py
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="EoMT panoptic video inference test"
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g. configs/dinov2/coco/panoptic/eomt_giant_720.yaml)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.bin or .ckpt). "
             "If omitted, attempts to download from Hugging Face Hub.",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index. Use -1 to force CPU.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of frames to process per forward pass.",
    )
    p.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help="Number of frames to process (0 = run until Ctrl-C / stream ends).",
    )
    p.add_argument(
        "--warmup_frames",
        type=int,
        default=2,
        help="Number of initial frames excluded from latency statistics "
             "(covers torch.compile warm-up).",
    )
    p.add_argument(
        "--input_device",
        type=str,
        default=None,
        help="Video capture device path (e.g. /dev/video0). "
             "If omitted, a synthetic random stream is used.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for segmented frames (MJPEG .avi). "
             "If omitted, frames are written as raw RGB bytes to stdout.",
    )
    p.add_argument(
        "--compile_backend",
        type=str,
        default="inductor",
        choices=["eager", "inductor", "migraphx"],
        help="torch.compile backend (matches main.py validate flag).",
    )
    p.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (matches main.py validate flag).",
    )
    p.add_argument(
        "--compiled_model_path",
        type=str,
        default=None,
        help="Path to save/load a compiled MIGraphX engine (.mgx).",
    )
    p.add_argument(
        "--inductor_cache_dir",
        type=str,
        default=None,
        help="Directory for persisting inductor kernel cache.",
    )
    p.add_argument(
        "--compile_disabled",
        action="store_true",
        help="Disable torch.compile entirely (eager execution).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading — mirrors inference.ipynb cell 6 / 8
# ---------------------------------------------------------------------------

def load_model(config, checkpoint_path, device):
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module`.*",
    )

    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_cls = getattr(importlib.import_module(data_module_name), class_name)
    data_kwargs = config["data"].get("init_args", {})

    # We only need dataset metadata (img_size, num_classes, stuff_classes),
    # not an actual dataloader, so we create a lightweight instance.
    dummy_data = data_cls(
        path="/dev/null",          # dataset dir — not used for metadata
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **data_kwargs,
    )

    img_size = dummy_data.img_size
    num_classes = dummy_data.num_classes
    stuff_classes = getattr(dummy_data, "stuff_classes", None)
    if stuff_classes is None:
        stuff_classes = data_kwargs.get("stuff_classes", [])

    # Encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(enc_mod), enc_cls_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    # Network (EoMT)
    network_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(net_mod), net_cls_name)
    network_kwargs = {
        k: v for k, v in network_cfg["init_args"].items() if k != "encoder"
    }
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    # Lightning module
    lit_mod, lit_cls_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_mod), lit_cls_name)
    model_kwargs = {
        k: v
        for k, v in config["model"]["init_args"].items()
        if k != "network"
    }
    model_kwargs["stuff_classes"] = stuff_classes

    model = (
        lit_cls(
            img_size=img_size,
            num_classes=num_classes,
            network=network,
            **model_kwargs,
        )
        .eval()
    )

    def _load_and_apply_ckpt(model, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}

        # Interpolate pos_embed if checkpoint resolution differs from model
        # (mirrors LightningModule._load_ckpt, lightning_module.py:904-917)
        pos_key = "network.encoder.backbone.pos_embed"
        if pos_key in ckpt and hasattr(model.network.encoder.backbone, "pos_embed"):
            ckpt_pos = ckpt[pos_key]
            model_pos = model.network.encoder.backbone.pos_embed
            if ckpt_pos.shape != model_pos.shape:
                src_n, tgt_n = ckpt_pos.shape[1], model_pos.shape[1]
                C = ckpt_pos.shape[2]
                src_h = src_w = int(src_n ** 0.5)
                tgt_h = tgt_w = int(tgt_n ** 0.5)
                pos_2d = ckpt_pos.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2).float()
                pos_2d = torch.nn.functional.interpolate(
                    pos_2d, size=(tgt_h, tgt_w), mode="bicubic", align_corners=False
                ).to(ckpt_pos.dtype)
                ckpt[pos_key] = pos_2d.permute(0, 2, 3, 1).reshape(1, tgt_n, C)
                logging.info(
                    f"Interpolated pos_embed from {src_h}x{src_w} ({src_n} patches) "
                    f"to {tgt_h}x{tgt_w} ({tgt_n} patches)"
                )

        model.load_state_dict(ckpt, strict=False)

    # Load weights
    if checkpoint_path is not None:
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        _load_and_apply_ckpt(model, checkpoint_path)
    else:
        # Try Hugging Face Hub (mirrors inference.ipynb cell 8)
        name = (
            config.get("trainer", {})
            .get("logger", {})
            .get("init_args", {})
            .get("name")
        )
        if name:
            try:
                from huggingface_hub import hf_hub_download

                logging.info(f"Downloading weights from Hugging Face: tue-mps/{name}")
                state_dict_path = hf_hub_download(
                    repo_id=f"tue-mps/{name}", filename="pytorch_model.bin"
                )
                _load_and_apply_ckpt(model, state_dict_path)
            except Exception as exc:
                logging.warning(f"Could not load HF weights: {exc}")
        else:
            logging.warning("No checkpoint path and no logger name in config — using random weights.")

    if device >= 0 and torch.cuda.is_available():
        model = model.to(device)
    else:
        device = "cpu"
        model = model.to(device)

    return model, img_size, num_classes, stuff_classes, device


# ---------------------------------------------------------------------------
# torch.compile — mirrors _compile_model() in main.py
# ---------------------------------------------------------------------------

def compile_model(model, args):
    if args.compile_disabled:
        logging.info("--compile_disabled: skipping torch.compile")
        return model

    if args.device < 0:
        logging.info("CPU mode: skipping torch.compile")
        return model

    backend = args.compile_backend
    if backend == "eager":
        logging.info("compile_backend=eager: no torch.compile")
        return model

    if backend == "migraphx":
        import torch_migraphx  # noqa: F401
        options = {}
        if args.compiled_model_path:
            if os.path.exists(args.compiled_model_path):
                logging.info(f"Loading MIGraphX engine from {args.compiled_model_path}")
                options["load_compiled"] = args.compiled_model_path
            else:
                logging.info(f"Will save MIGraphX engine to {args.compiled_model_path}")
                options["save_compiled"] = args.compiled_model_path
        logging.info("torch.compile(backend='migraphx')")
        return torch.compile(model, backend="migraphx", options=options)

    if args.inductor_cache_dir:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = args.inductor_cache_dir
        torch._inductor.config.fx_graph_cache = True
        logging.info(f"Inductor cache: {args.inductor_cache_dir}")

    logging.info(f"torch.compile(backend='{backend}', mode='{args.compile_mode}')")
    return torch.compile(model, backend=backend, mode=args.compile_mode)


# ---------------------------------------------------------------------------
# Video source
# ---------------------------------------------------------------------------

class SyntheticStream:
    """Yields random uint8 frames at 1280×720."""

    def __init__(self):
        self.frame_idx = 0

    def read(self):
        frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        self.frame_idx += 1
        return True, frame

    def release(self):
        pass


def open_capture(input_device):
    if input_device is None:
        logging.info("No input device specified — using synthetic random stream.")
        return SyntheticStream()

    try:
        import cv2  # optional; only needed for real camera input
    except ImportError:
        logging.error(
            "OpenCV (cv2) is required for real video input. "
            "Install it with: pip install opencv-python  — or use synthetic mode."
        )
        sys.exit(1)

    cap = cv2.VideoCapture(input_device)
    if not cap.isOpened():
        logging.error(f"Cannot open video device: {input_device}")
        sys.exit(1)

    # Request 1280×720 from the device; actual resolution may differ.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"Opened {input_device} — resolution: {actual_w}×{actual_h}")
    return cap


# ---------------------------------------------------------------------------
# Pre-processing: 1280×720 → 720×720 centre-crop then convert to tensor
# ---------------------------------------------------------------------------

def preprocess_frame(frame_bgr_or_rgb: np.ndarray, is_bgr: bool = False) -> torch.Tensor:
    """
    Input : HxWx3 uint8 numpy array (BGR from cv2, or RGB from synthetic).
    Output: 3xHxW uint8 torch.Tensor (RGB).
    Crops the 1280×720 frame to a 720×720 centre square.
    """
    if is_bgr:
        frame_rgb = frame_bgr_or_rgb[:, :, ::-1]  # BGR→RGB without copy overhead
    else:
        frame_rgb = frame_bgr_or_rgb

    h, w = frame_rgb.shape[:2]

    # Centre-crop to a square of min(h, w)
    side = min(h, w)
    top  = (h - side) // 2
    left = (w - side) // 2
    crop = frame_rgb[top: top + side, left: left + side]

    # Resize to model img_size if needed (typically already 720×720)
    if crop.shape[0] != 720 or crop.shape[1] != 720:
        pil = Image.fromarray(crop.copy())
        pil = pil.resize((720, 720), Image.BILINEAR)
        crop = np.array(pil)

    # HWC → CHW, uint8 tensor (model.forward divides by 255 internally)
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)
    return tensor


# ---------------------------------------------------------------------------
# Inference — panoptic path (mirrors inference.ipynb infer_panoptic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_batch(model, imgs_tensor_list, device, dtype=torch.float16):
    """
    imgs_tensor_list : list of 3×H×W uint8 tensors (RGB).
    Returns sem_pred (H×W int64) for the first image in the batch.
    """
    device_type = "cuda" if isinstance(device, int) and device >= 0 else "cpu"
    imgs = [img.to(device) for img in imgs_tensor_list]
    img_sizes = [img.shape[-2:] for img in imgs]

    with autocast(dtype=dtype, device_type=device_type):
        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)

        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], model.img_size, mode="bilinear"
        )
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, img_sizes
        )

        preds = model.to_per_pixel_preds_panoptic(
            mask_logits,
            class_logits_per_layer[-1],
            model.stuff_classes,
            model.mask_thresh,
            model.overlap_thresh,
        )

    # Return per-frame semantic segmentation maps (HxW, class ids)
    results = []
    for pred in preds:
        sem = pred[..., 0].cpu()   # semantic class id
        results.append(sem)
    return results


# ---------------------------------------------------------------------------
# Colourisation — mirrors draw_black_border + plot_panoptic_results in notebook
# ---------------------------------------------------------------------------

def colorize_segmentation(sem: np.ndarray, num_classes: int) -> np.ndarray:
    """Map class-id map (H×W int64) to an RGB image (H×W×3 uint8)."""
    import matplotlib.pyplot as plt

    unique_ids = np.unique(sem)
    valid_ids = unique_ids[unique_ids >= 0]
    n = max(len(valid_ids), 1)

    id_to_color = {}
    for i, cid in enumerate(valid_ids):
        rgb = plt.cm.hsv(i / n)[:3]
        id_to_color[cid] = (np.array(rgb) * 255).astype(np.uint8)

    out = np.zeros((*sem.shape, 3), dtype=np.uint8)
    for cid, color in id_to_color.items():
        out[sem == cid] = color
    # Background / void pixels (value < 0 or == num_classes) stay black.
    return out


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

class StdoutWriter:
    """Writes raw RGB frames as bytes to stdout (e.g. pipe into ffplay)."""

    def write(self, rgb_frame: np.ndarray):
        sys.stdout.buffer.write(rgb_frame.tobytes())
        sys.stdout.buffer.flush()

    def release(self):
        pass


class VideoFileWriter:
    """Writes MJPEG AVI using OpenCV."""

    def __init__(self, path: str, width: int, height: int, fps: float = 25.0):
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open output video file: {path}")
        logging.info(f"Writing output to {path}")

    def write(self, rgb_frame: np.ndarray):
        import cv2
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def release(self):
        self._writer.release()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    args = parse_args()

    # Suppress noisy torch dynamo logs
    os.environ.setdefault("TORCH_LOGS", "-dynamo")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    torch.set_float32_matmul_precision("medium")
    model, img_size, num_classes, stuff_classes, device = load_model(
        config, args.checkpoint, args.device
    )

    # Compile
    model = compile_model(model, args)

    # Determine autocast dtype
    if isinstance(device, int) and device >= 0 and torch.cuda.is_available():
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.bfloat16  # CPU-safe

    # Open input stream
    cap = open_capture(args.input_device)
    is_bgr = args.input_device is not None  # cv2 gives BGR; synthetic gives RGB

    # Open output writer
    out_h, out_w = img_size  # 720×720 after crop+resize
    if args.output is not None:
        writer = VideoFileWriter(args.output, width=out_w, height=out_h)
    else:
        logging.info(
            f"Writing raw RGB frames ({out_w}x{out_h}) to stdout. "
            "Pipe with: python test.py ... | ffplay -f rawvideo -pixel_format rgb24 "
            f"-video_size {out_w}x{out_h} -"
        )
        writer = StdoutWriter()

    # Latency tracking
    latencies_ms = []        # capture-to-output latencies (after warmup)
    frame_count = 0
    warmup_remaining = args.warmup_frames

    logging.info(
        f"Starting inference  |  img_size={img_size}  batch={args.batch_size}  "
        f"warmup={args.warmup_frames}  backend={args.compile_backend}"
    )

    batch_frames = []        # raw numpy frames for current batch
    batch_tensors = []       # pre-processed tensors

    try:
        while True:
            if args.num_frames > 0 and frame_count >= args.num_frames:
                break

            # ---- Capture -------------------------------------------------------
            t_capture = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                logging.warning("Video stream ended or read error.")
                break

            tensor = preprocess_frame(frame, is_bgr=is_bgr)
            batch_frames.append(frame)
            batch_tensors.append(tensor)

            if len(batch_tensors) < args.batch_size:
                continue  # accumulate until we have a full batch

            # ---- Inference -----------------------------------------------------
            if isinstance(device, int) and device >= 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_infer_start = time.perf_counter()

            seg_maps = infer_batch(model, batch_tensors, device, dtype=autocast_dtype)

            if isinstance(device, int) and device >= 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_infer_end = time.perf_counter()

            # ---- Write output --------------------------------------------------
            for seg in seg_maps:
                seg_np = seg.numpy().astype(np.int64)
                rgb_out = colorize_segmentation(seg_np, num_classes)
                writer.write(rgb_out)

            t_output = time.perf_counter()

            # ---- Latency measurement ------------------------------------------
            total_latency_ms = (t_output - t_capture) * 1000
            infer_latency_ms = (t_infer_end - t_infer_start) * 1000

            if warmup_remaining > 0:
                warmup_remaining -= 1
                logging.info(
                    f"[warmup] frame {frame_count+1}/{args.warmup_frames}  "
                    f"infer={infer_latency_ms:.1f} ms  total={total_latency_ms:.1f} ms"
                )
            else:
                latencies_ms.append(total_latency_ms)
                fps_inst = args.batch_size * 1000.0 / total_latency_ms
                logging.info(
                    f"frame {frame_count + 1}  "
                    f"infer={infer_latency_ms:.1f} ms  "
                    f"total={total_latency_ms:.1f} ms  "
                    f"fps={fps_inst:.1f}"
                )

            frame_count += args.batch_size
            batch_frames.clear()
            batch_tensors.clear()

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")

    # ---- Summary ---------------------------------------------------------------
    if latencies_ms:
        arr = np.array(latencies_ms)
        logging.info(
            f"\n{'='*60}\n"
            f"Processed {frame_count} frames  ({len(latencies_ms)} measured)\n"
            f"  Latency (capture → output)\n"
            f"    mean : {arr.mean():.1f} ms\n"
            f"    min  : {arr.min():.1f} ms\n"
            f"    max  : {arr.max():.1f} ms\n"
            f"    p50  : {np.percentile(arr, 50):.1f} ms\n"
            f"    p95  : {np.percentile(arr, 95):.1f} ms\n"
            f"  Throughput: {1000.0 / arr.mean() * args.batch_size:.1f} fps\n"
            f"{'='*60}"
        )
    else:
        logging.info("No frames measured (all consumed by warmup).")

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
