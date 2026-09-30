# Patch for torch_migraphx.fx.mgx_module.MGXModule:
#
# Problem: MGXModule.forward() contains a Python `if` branching on
# `inp_val.device.type == 'cuda'`, which is a control-flow check on a symbolic
# tensor.  When torch.compile/dynamo retraces subgraphs that include a loaded
# MGXModule, it hits this branch and raises:
#   TraceError: symbolically traced variables cannot be used as inputs to control flow
# With suppress_errors=True set in main.py, dynamo silently falls back to
# PyTorch eager for those subgraphs, producing ~23% of GPU time as unoptimised
# at::native kernels and significant CPU dispatch overhead between batches.
#
# Fix: decorate MGXModule.forward with @torch.compiler.disable so dynamo treats
# it as an opaque callable and never attempts to trace through it.  The module
# is already compiled by MIGraphX; there is nothing for dynamo to optimise
# inside it.
#
# Usage: call patch_mgx_module() once before torch.compile(..., backend="migraphx").

import torch
import torch_migraphx.fx.mgx_module as _mgx_mod


def patch_mgx_module():
    """Wrap MGXModule.forward with @torch.compiler.disable to prevent dynamo
    from tracing into it and hitting the control-flow TraceError."""
    original_forward = _mgx_mod.MGXModule.forward
    _mgx_mod.MGXModule.forward = torch.compiler.disable(original_forward)
