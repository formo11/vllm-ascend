#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""GPTQ (Linear) skeleton for Ascend.

Upstream vLLM's GPTQ linear path stores:
- qweight: int32, packed along input K dimension
- qzeros: int32, packed along output N dimension
- scales: fp16/bf16
- g_idx: int32 (group index / act-order permutation)

This file provides a minimal, kernel-guarded implementation that matches the
expected vLLM LinearMethod interface (create_weights / process_weights_after_loading / apply).

NOTE: This is intentionally a *skeleton*: it will raise a clear error unless a
compatible GPTQ GEMM kernel exists in the runtime (e.g., torch.ops._C.gptq_gemm).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

import torch

try:
    # vLLM core interfaces.
    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.parameter import (
        ChannelQuantScaleParameter,
        GroupQuantScaleParameter,
        PackedColumnParameter,
        PackedvLLMParameter,
        RowvLLMParameter,
    )
except Exception as exc:  # pragma: no cover
    # Keep import-time failures explicit for environments without vLLM installed.
    raise ImportError(
        "vLLM is required to import Ascend GPTQ skeleton."
    ) from exc


class ExllamaState(enum.Enum):
    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()


@dataclass(frozen=True)
class AscendGPTQConfig:
    """Minimal GPTQ config fields used by the linear method.

    This mirrors upstream's GPTQConfig fields required for tensor shapes.
    """

    weight_bits: int
    group_size: int
    desc_act: bool = False
    checkpoint_format: str = ""  # "gptq_v2" enables the v2 qzeros semantics upstream.

    @property
    def pack_factor(self) -> Fraction:
        # Upstream uses Fraction(32, weight_bits) so the same code works for 2/3/4/8.
        return Fraction(32, self.weight_bits)

    @property
    def use_v2_format(self) -> bool:
        return self.checkpoint_format == "gptq_v2"


def _has_gptq_ops() -> bool:
    # The upstream CUDA implementation registers these as torch.ops._C.*.
    return (
        hasattr(torch, "ops")
        and hasattr(torch.ops, "_C")
        and hasattr(torch.ops._C, "gptq_gemm")
    )


def _maybe_has_gptq_shuffle() -> bool:
    return (
        hasattr(torch, "ops")
        and hasattr(torch.ops, "_C")
        and hasattr(torch.ops._C, "gptq_shuffle")
    )


class AscendGPTQLinearMethod(LinearMethodBase):
    """GPTQ LinearMethod skeleton.

    - Allocates/loads GPTQ parameters with upstream-compatible shapes.
    - Optionally performs the upstream "shuffle" step if an op exists.
    - Raises a clear error in apply() unless a GPTQ GEMM op is present.
    """

    def __init__(self, quant_config: AscendGPTQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Shape/TP alignment checks.
        if self.quant_config.group_size != -1:
            if input_size_per_partition % self.quant_config.group_size != 0:
                raise ValueError(
                    "The input size is not aligned with the quantized weight shape. "
                    "This can be caused by too large tensor parallel size."
                )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor.numerator != 0:
            raise ValueError(
                "The output size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size."
            )

        group_size = (
            self.quant_config.group_size
            if self.quant_config.group_size != -1
            else input_size
        )

        # Upstream special-cases row-parallel + non-desc_act to partition scales/qzeros.
        exllama_state = ExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim: Optional[int] = None

        if input_size != input_size_per_partition and self.quant_config.group_size != -1:
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0

        qweight = PackedvLLMParameter(
            data=torch.empty(
                int(input_size_per_partition // self.quant_config.pack_factor),
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        # g_idx layout matches upstream: int32, len = input_size_per_partition.
        g_idx = RowvLLMParameter(
            data=torch.tensor(
                [i // group_size for i in range(input_size_per_partition)],
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )

        qzeros_args = {
            "data": torch.empty(
                scale_and_zero_size,
                int(output_size_per_partition // self.quant_config.pack_factor),
                dtype=torch.int32,
            ),
            "weight_loader": weight_loader,
        }
        weight_scale_args = {
            "data": torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }

        if scale_and_zero_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )
        else:
            scales = GroupQuantScaleParameter(
                output_dim=1,
                input_dim=0,
                **weight_scale_args,
            )
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        # Keep the attribute name consistent with upstream.
        layer.exllama_state = exllama_state

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fail fast on Ascend until a GPTQ GEMM kernel exists.
        if not _has_gptq_ops():
            raise NotImplementedError(
                "GPTQ is not supported on Ascend in this build: "
                "missing torch.ops._C.gptq_gemm. "
                "Use AWQ or compressed-tensors on Ascend, or implement an NPU GPTQ GEMM kernel."
            )

        # Wrap into leaf Parameters for torch.compile compatibility.
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.g_idx = torch.nn.Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        # Optional upstream shuffle (only if the op exists in the runtime).
        if getattr(layer, "exllama_state", ExllamaState.UNUSED) == ExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act:
                layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
            else:
                layer.g_idx.data = torch.empty((0,), dtype=torch.int, device=layer.g_idx.device)
            layer.exllama_state = ExllamaState.READY

            if _maybe_has_gptq_shuffle():
                torch.ops._C.gptq_shuffle(layer.qweight, layer.g_idx, self.quant_config.weight_bits)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.qweight.shape[-1],)
        reshaped_x = x.reshape(-1, x.shape[-1])

        output = torch.ops._C.gptq_gemm(
            reshaped_x,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            layer.g_idx,
            getattr(layer, "exllama_state", ExllamaState.UNUSED) == ExllamaState.READY,
            self.quant_config.use_v2_format,
            self.quant_config.weight_bits,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)


@register_quantization_config("gptq")
class AscendGPTQQuantConfig(QuantizationConfig):
    """Registers a GPTQ config for Ascend.

    This intentionally fails fast on NPU until a real GPTQ GEMM kernel exists.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        checkpoint_format: str = "",
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.checkpoint_format = checkpoint_format

        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is supported for GPTQ, "
                f"but got {self.weight_bits} bits."
            )

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware does not support 'get_min_capability' feature."
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        # Same as upstream GPTQ.
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendGPTQQuantConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        checkpoint_format = cls.get_from_keys_or(
            config, ["checkpoint_format"], default=""
        )
        return cls(weight_bits, group_size, desc_act, checkpoint_format)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        del prefix
        if isinstance(layer, LinearBase):
            cfg = AscendGPTQConfig(
                weight_bits=self.weight_bits,
                group_size=self.group_size,
                desc_act=self.desc_act,
                checkpoint_format=self.checkpoint_format,
            )
            return AscendGPTQLinearMethod(cfg)
        return None

