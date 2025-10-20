"""
2025.10.5
2025.10.5
4.56.2
0.23.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 16, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': 0, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': False, 'triton.cooperative_reductions': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': False, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True}
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from peft.tuners.lora.gptq import (torch)


torch_addmm = torch.addmm
torch_add   = torch.add
torch_float16 = torch.float16
# @torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def lora_forward(result, lora_A, lora_B, dropout, x, scaling):
    xA = dropout(x.to(torch_float16)) @ lora_A.weight.to(torch_float16).t()
    # output = result + scaling * xA @ lora_B.weight.t()
    shape = result.shape
    output = torch_addmm(
        result.view(-1, shape[-1]).to(torch_float16),
        xA.view(-1, xA.shape[-1]),
        lora_B.weight.to(torch_float16).t(),
        alpha = scaling,
        beta = 1,
    ).view(shape)

    bias = lora_B.bias
    if bias is not None:
        output = torch_add(
        output,
        bias.to(torch_float16),
        alpha = scaling,
    )
    return output
pass

def unsloth_forward(self, x: torch.Tensor):
    # note: logic differs from default Linear because merging is not supported
    result = self.quant_linear_module(x)

    if self.disable_adapters:
        return result

    lora_A_keys = self.lora_A.keys()

    for active_adapter in self.active_adapters:
        if active_adapter not in lora_A_keys:
            continue
        torch_result_dtype = result.dtype

        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]

        

        if active_adapter not in self.lora_variant:  # vanilla LoRA
            return lora_forward(result, lora_A, lora_B, dropout, x, scaling)
        else:
            result = self.lora_variant[active_adapter].forward(
                self,
                active_adapter=active_adapter,
                x=x,
                result=result,
            )

        result = result.to(torch_result_dtype)
    return result
