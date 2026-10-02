# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from PyTorch Lightning,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------


import jsonargparse._typehints as _t
from types import MethodType
from gitignore_parser import parse_gitignore
import logging
import torch
import torch._inductor.config
import warnings
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule

# Suppress PyTorch FX warnings for DINOv3 models
import os
os.environ["TORCH_LOGS"] = "-dynamo"


_orig_single = _t.raise_unexpected_value


def _raise_single(*args, exception=None, **kwargs):
    if isinstance(exception, Exception):
        raise exception
    return _orig_single(*args, exception=exception, **kwargs)


_orig_union = _t.raise_union_unexpected_value


def _raise_union(subtypes, val, vals):
    for e in reversed(vals):
        if isinstance(e, Exception):
            raise e
    return _orig_union(subtypes, val, vals)


_t.raise_unexpected_value = _raise_single
_t.raise_union_unexpected_value = _raise_union


def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            # added below to check val based on global steps instead of batches in case of iteration based val check and gradient accumulation
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("torch_migraphx").setLevel(logging.WARNING)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
        )
        warnings.filterwarnings(
            "ignore", message=r"^Grad strides do not match bucket view strides.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*functools.partial will be a method descriptor in future Python versions*",
        )

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--compile_backend",
            type=str,
            default="inductor",
            choices=["eager", "inductor", "migraphx"],
            help=(
                "Execution backend. "
                "'inductor' (default) compiles with the TorchInductor/Triton backend. "
                "'eager' disables compilation and runs in pure PyTorch eager mode. "
                "'migraphx' compiles with AMD MIGraphX (ROCm only); use with "
                "--compiled_model_path to save/load the engine and --exhaustive_tune "
                "to search for the fastest kernel configuration."
            ),
        )
        parser.add_argument(
            "--compiled_model_path",
            type=str,
            default=None,
            help="Path to save/load the compiled MIGraphX engine (.mgx). "
                 "If the file exists it is loaded (skipping recompilation); "
                 "otherwise the compiled engine is saved there after the first run. "
                 "Only used when --compile_backend=migraphx.",
        )
        parser.add_argument(
            "--exhaustive_tune",
            action="store_true",
            help="Enable MIGraphX exhaustive kernel autotuning. Searches all available "
                 "kernel implementations for each op to find the fastest configuration. "
                 "Significantly increases first-run compilation time but improves "
                 "steady-state performance, especially for GEMMs. Results are saved "
                 "to --compiled_model_path and reused on subsequent runs. "
                 "Only used when --compile_backend=migraphx.",
        )
        parser.add_argument(
            "--compile_mode",
            type=str,
            default="default",
            choices=["default", "reduce-overhead", "max-autotune"],
            help=(
                "torch.compile optimization mode (ignored when compile_backend=eager). "
                "'default': standard kernel generation. "
                "'reduce-overhead': uses CUDA graphs to eliminate kernel-launch overhead; "
                "best for fixed input shapes. "
                "'max-autotune': searches for the fastest kernel configuration per op; "
                "slowest first batch, best steady-state throughput."
            ),
        )
        parser.add_argument(
            "--cpu",
            action="store_true",
            help=(
                "Force CPU execution regardless of GPU availability. "
                "Overrides --trainer.accelerator and --trainer.devices. "
                "Useful as a baseline comparison against GPU-accelerated runs."
            ),
        )
        parser.add_argument(
            "--warmup_batches",
            type=int,
            default=1,
            help=(
                "Number of initial batches to exclude from inference timing. "
                "Defaults to 1 to skip the first-batch compilation overhead when "
                "using torch.compile. Set to 0 to include all batches in the average "
                "(matches original behaviour). Has no effect when --compile_backend=eager."
            ),
        )
        parser.add_argument(
            "--inductor_cache_dir",
            type=str,
            default=None,
            help=(
                "Directory for persisting torch inductor compiled kernel cache. "
                "When set, Triton kernels compiled on the first run are saved here "
                "and loaded on subsequent runs, eliminating first-batch compilation "
                "overhead. Ignored when --compile_backend=eager. "
                "Example: --inductor_cache_dir /tmp/inductor_cache/eomt_large_640"
            ),
        )
        parser.add_argument(
            "--autotune_gemm",
            action="store_true",
            help=(
                "Replace rocBLAS GEMM calls with Triton GEMM kernels and autotune "
                "tile configurations (BLOCK_M, BLOCK_N, BLOCK_K) to find smaller "
                "tiles that reduce VGPR pressure and improve wavefront occupancy. "
                "Slower first run (benchmarks tile options); subsequent runs reuse "
                "cached results from --inductor_cache_dir. Ignored when "
                "--compile_backend=eager."
            ),
        )
        parser.add_argument(
            "--profile_batches",
            type=int,
            default=0,
            help=(
                "Number of batches to profile with torch.profiler after warmup. "
                "0 (default) disables profiling. When > 0, records CPU and GPU kernel "
                "timings, prints a top-20 ops table sorted by GPU time, and saves a "
                "Chrome trace to --profile_output_dir for visualisation in "
                "chrome://tracing or Perfetto (https://ui.perfetto.dev)."
            ),
        )
        parser.add_argument(
            "--profile_output_dir",
            type=str,
            default="./profile_output",
            help=(
                "Directory for torch.profiler Chrome trace files. "
                "Created automatically if it does not exist. "
                "Only used when --profile_batches > 0."
            ),
        )

        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )

        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )

        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )

        parser.link_arguments(
            "model.init_args.ckpt_path",
            "model.init_args.network.init_args.encoder.init_args.ckpt_path",
        )

    def before_instantiate_classes(self):
        subcommand = self.config.get("subcommand", "")
        if subcommand and self.config[subcommand].get("cpu", False):
            logging.info("--cpu set: overriding trainer to use CPU accelerator")
            self.config[subcommand]["trainer"]["accelerator"] = "cpu"
            self.config[subcommand]["trainer"]["devices"] = 1

    def _compile_model(self, model, subcommand):
        """Apply torch.compile according to --compile_backend and --compile_mode."""
        cfg = self.config[subcommand]

        if cfg.get("cpu", False):
            logging.info("--cpu set: skipping torch.compile (not beneficial on CPU)")
            return model

        backend = cfg.get("compile_backend", "inductor")

        if backend == "eager":
            logging.info("compile_backend=eager: running in pure PyTorch eager mode (no torch.compile)")
            return model

        if backend == "migraphx":
            import torch_migraphx  # noqa: F401 — registers the backend
            from migraphx_patch import patch_mgx_module
            patch_mgx_module()
            logging.info("MGXModule patched: @torch.compiler.disable on forward + output buffer caching")

            compiled_model_path = cfg.get("compiled_model_path", None)
            exhaustive_tune = cfg.get("exhaustive_tune", False)
            options = {}
            if exhaustive_tune:
                options["exhaustive_tune"] = True
                logging.info("MIGraphX exhaustive tuning enabled — first-run compilation will be slow")
            if compiled_model_path:
                if os.path.exists(compiled_model_path):
                    logging.info(f"Loading compiled MIGraphX engine from {compiled_model_path}")
                    options["load_compiled"] = compiled_model_path
                else:
                    logging.info(f"Will save compiled MIGraphX engine to {compiled_model_path}")
                    options["save_compiled"] = compiled_model_path

            logging.info("torch.compile(backend='migraphx')")
            return torch.compile(model, backend="migraphx", options=options)

        # inductor (default)
        cache_dir = cfg.get("inductor_cache_dir", None)
        if cache_dir:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
            torch._inductor.config.fx_graph_cache = True
            logging.info(f"Inductor kernel cache: {cache_dir}")

        if cfg.get("autotune_gemm", False):
            torch._inductor.config.max_autotune_gemm = True
            torch._inductor.config.max_autotune_gemm_backends = "TRITON"
            logging.info("GEMM autotuning enabled: Triton will search for optimal tile configuration")

        mode = cfg.get("compile_mode", "default")
        logging.info(f"torch.compile(backend='{backend}', mode='{mode}')")
        return torch.compile(model, backend=backend, mode=mode)

    def fit(self, model, **kwargs):
        if self.trainer.logger is None:
            self.trainer.callbacks = [
                cb for cb in self.trainer.callbacks
                if not isinstance(cb, LearningRateMonitor)
            ]
        if self.trainer.logger is not None and hasattr(self.trainer.logger.experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        model = self._compile_model(model, "fit")
        self.trainer.fit(model, **kwargs)

    def validate(self, model, **kwargs):
        compiled = self._compile_model(model, "validate")
        cfg = self.config["validate"]
        backend = cfg.get("compile_backend", "inductor")
        warmup = cfg.get("warmup_batches", 1)
        if backend != "eager" and not cfg.get("cpu", False) and warmup > 0:
            model._infer_warmup_remaining = warmup
            logging.info(f"Excluding first {warmup} batch(es) from inference timing")

        profile_batches = cfg.get("profile_batches", 0)
        if profile_batches > 0:
            import torch.profiler as tprof
            output_dir = cfg.get("profile_output_dir", "./profile_output")
            os.makedirs(output_dir, exist_ok=True)
            activities = [tprof.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(tprof.ProfilerActivity.CUDA)
            prof = tprof.profile(
                activities=activities,
                schedule=tprof.schedule(
                    wait=warmup,       # skip warmup/compile batches
                    warmup=1,          # one profiler warm-up batch
                    active=profile_batches,
                    repeat=1,
                ),
                on_trace_ready=tprof.tensorboard_trace_handler(output_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            model._profiler = prof
            prof.start()
            logging.info(
                f"torch.profiler active: recording {profile_batches} batch(es) "
                f"after {warmup} warmup batch(es). Trace → {output_dir}"
            )

        self.trainer.validate(compiled, **kwargs)

        if profile_batches > 0:
            prof.stop()
            sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
            print("\n" + "="*80)
            print(f"torch.profiler — top 20 ops by {sort_key}:")
            print("="*80)
            print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
            logging.info(f"Chrome trace saved to {output_dir} — open at https://ui.perfetto.dev")


def cli_main():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )


if __name__ == "__main__":
    cli_main()
