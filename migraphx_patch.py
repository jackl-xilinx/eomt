# Patch for torch_migraphx.fx.mgx_module.MGXModule:
# Pre-allocate output buffers once at first call and reuse them on every
# subsequent call.  The upstream implementation calls torch.empty_strided()
# inside forward() on every batch, hitting the CUDA allocator repeatedly and
# adding ~200 ms/img of CPU overhead at inference time.
#
# Usage: call patch_mgx_module() once before torch.compile(..., backend="migraphx").

import torch
import torch_migraphx.fx.mgx_module as _mgx_mod
from torch_migraphx.fx.utils import mgx_argument_from_ptr, tensors_from_mgx_arguments


class CachedMGXModule(_mgx_mod.MGXModule):
    """MGXModule with pre-allocated, reused output buffers."""

    def _allocate_param_buffers(self, names):
        # On first call, allocate and cache; on subsequent calls, reuse.
        if self.torch_buffers:
            return
        for param_name in names:
            param_shape = self.program.get_parameter_shapes()[param_name]
            if param_shape.type_string() == 'tuple_type':
                raise RuntimeError(
                    'Tuple return types are not currently supported')
            type_str  = param_shape.type_string()
            lens      = param_shape.lens()
            strides   = param_shape.strides()
            torch_dtype = _mgx_mod.torch_dtype_from_mgx(type_str)
            tensor = torch.empty_strided(
                lens, strides,
                dtype=torch_dtype,
                device=torch.cuda.current_device(),
            )
            self.torch_buffers[param_name] = tensor
            self.mgx_buffers[param_name] = mgx_argument_from_ptr(
                tensor.data_ptr(), param_shape)


def patch_mgx_module():
    """Replace MGXModule with CachedMGXModule in the torch_migraphx namespace."""
    _mgx_mod.MGXModule = CachedMGXModule
    # Also patch the reference held by lower_dynamo which already imported it
    import torch_migraphx.fx.lower as _lower
    _lower.MGXModule = CachedMGXModule
    import torch_migraphx.dynamo.lower_dynamo as _dynamo_lower
    _dynamo_lower.MGXModule = CachedMGXModule
