# Patch for torch_migraphx.fx.mgx_module.MGXModule:
#
# Problem 1: MGXModule.forward() contains a Python `if` branching on
# `inp_val.device.type == 'cuda'`, which is a control-flow check on a symbolic
# tensor.  When torch.compile/dynamo retraces subgraphs that include a loaded
# MGXModule, it hits this branch and raises:
#   TraceError: symbolically traced variables cannot be used as inputs to control flow
# With suppress_errors=True set in main.py, dynamo silently falls back to
# PyTorch eager for those subgraphs, producing ~23% of GPU time as unoptimised
# at::native kernels and significant CPU dispatch overhead between batches.
# Fix: decorate MGXModule.forward with @torch.compiler.disable so dynamo treats
# it as an opaque callable and never attempts to trace through it.  The module
# is already compiled by MIGraphX; there is nothing for dynamo to optimise
# inside it.
#
# Problem 2: MGXModule._allocate_param_buffers() calls torch.empty_strided() on
# every forward pass, hitting the CUDA allocator repeatedly and adding ~200 ms/img
# of CPU overhead at inference time.
# Fix: skip reallocation when output buffers are already populated (first call
# only), reusing the same tensors on every subsequent batch.
#
# Usage: call patch_mgx_module() once before torch.compile(..., backend="migraphx").

import torch
import torch_migraphx.fx.mgx_module as _mgx_mod


def patch_mgx_module():
    # Fix 1: prevent dynamo from tracing into MGXModule.forward.
    original_forward = _mgx_mod.MGXModule.forward
    _mgx_mod.MGXModule.forward = torch.compiler.disable(original_forward)

    # Fix 2: cache output buffers to avoid per-batch CUDA allocator overhead.
    _original_allocate = _mgx_mod.MGXModule._allocate_param_buffers

    def _allocate_param_buffers_cached(self, names):
        if self.torch_buffers:
            return
        _original_allocate(self, names)

    _mgx_mod.MGXModule._allocate_param_buffers = _allocate_param_buffers_cached
