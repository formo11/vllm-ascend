#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from typing import Any, Callable, Dict, Optional

import os

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.experts_selector import select_experts


def unpack_from_int32(
    weight: torch.Tensor,
    shape: torch.Size,
    num_bits: int,
    packed_dim: int = 1,
) -> torch.Tensor:
    """
    Unpacks quantized weights from int32 format back to original bits.

    :param weight: The packed int32 tensor containing quantized weights
    :param shape: Original shape to restore, defaults to None
    :param num_bits: The number of bits used for quantization (<= 8)
    :param packed_dim: Dimension along which weights are packed (0 or 1), defaults to 1
    :return: Unpacked tensor with int8 dtype after applying offset correction
    """
    assert weight.dtype == torch.int32, f"Expecting `weight.dtype` is torch.int32 but got {weight.dtype}."
    assert num_bits <= 8, f"Expecting `num_bits` should not be larger than 8 but got {num_bits}."

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] * pack_factor),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[:, i::pack_factor] = (weight >>
                                                  (num_bits * i)) & mask
        original_row_size = int(shape[1])
        unpacked_weight = unpacked_weight[:, :original_row_size]
    else:
        unpacked_weight = torch.zeros(
            (weight.shape[0] * pack_factor, weight.shape[1]),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[i::pack_factor, :] = (weight >>
                                                  (num_bits * i)) & mask
        original_row_size = int(shape[0])
        unpacked_weight = unpacked_weight[:original_row_size, :]

    # Convert unsigned n-bit values into signed integers.
    #
    # NOTE: Different toolchains store int4 differently:
    # - "offset" (zero-point) encoding: stored u in [0..15] represents q=u-8
    #   (common for symmetric int4 where near-zero values cluster around 8).
    # - "twos" (two's-complement) encoding: u in [0..15] is interpreted as
    #   signed int4 via sign extension (near-zero clusters around 0/15).
    #
    # This checkpoint family (DeepSeek/Qwen w4a16 from llmcompressor) is
    # typically offset-encoded. We support an env override and a cheap auto
    # detector to avoid silent accuracy collapse.
    mode = os.environ.get("VLLM_ASCEND_W4A16_INT4_ENCODING", "auto").lower()
    if mode not in ("auto", "offset", "twos"):
        mode = "auto"

    if mode == "auto" and num_bits == 4:
        sample = unpacked_weight.flatten()
        if sample.numel() > 0:
            sample = sample[:min(sample.numel(), 65536)].to(torch.int64)
            counts = torch.bincount(sample, minlength=16)
            # If the most frequent nibble is 8, it's almost certainly offset.
            if int(torch.argmax(counts).item()) == 8:
                mode = "offset"
            else:
                mode = "twos"

    if mode == "offset":
        zero_point = 1 << (num_bits - 1)
        unpacked_weight = (unpacked_weight - zero_point).to(torch.int8)
    else:
        sign_bit = 1 << (num_bits - 1)
        unpacked_weight = ((unpacked_weight ^ sign_bit) - sign_bit).to(torch.int8)

    return unpacked_weight


def pack_to_int32(weight: torch.Tensor) -> torch.Tensor:
    """
    Packs quantized weights into int32 format for storage.

    :param weight: The 3D tensor to pack, must be int8 or int32 dtype
    :return: Packed tensor with int32 dtype optimized for storage
    """
    assert weight.dim(
    ) == 3, f"Expecting `weight.dim()` is 3 ([e, n, k] or [e, k, n]) but got {weight.dim()}."
    assert weight.dtype in [
        torch.int8, torch.int32
    ], f"Expecting `weight.dtype` is torch.int8 or torch.int32 bug got {weight.dtype}."

    if weight.dtype == torch.int32:
        assert weight.shape[
            -1] % 8 == 0, "the last dim of weight needs to be divided by 8."
        packed_weight = torch_npu.npu_convert_weight_to_int4pack(
            weight.flatten(0, 1))
        packed_weight = packed_weight.view(weight.shape[0], weight.shape[1],
                                           -1)
    else:
        assert weight.shape[
            -1] % 4 == 0, "the last dim of weight needs to be divided by 4."
        packed_weight = weight.view(torch.int32).contiguous()

    return packed_weight


class AscendW4A16LinearMethod:
    """Linear method for Ascend W4A16 (weight-only int4, activation fp16/bf16).

    This matches the common compressed-tensors "pack-quantized" checkpoint layout:
    - `weight_packed`: int32 with shape [out_features, in_features / 8]
    - `weight_scale`: bf16/fp16 with shape [out_features, in_features / group_size]
    - `weight_g_idx`: int32 with shape [in_features]
    - `weight_shape`: int64 with shape [2]
    """

    def __init__(self, group_size: int = 128) -> None:
        self.group_size = group_size
        self.num_bits = 4
        self.pack_factor = 32 // self.num_bits  # 8 int4 values per int32

    def get_weight(self, input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        if input_size % self.pack_factor != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by {self.pack_factor} for int4 packing"
            )
        if self.group_size <= 0 or input_size % self.group_size != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by group_size ({self.group_size})"
            )
        if output_size % self.pack_factor != 0:
            raise ValueError(
                f"output_size ({output_size}) must be divisible by {self.pack_factor} for int4 packing"
            )
        return {
            "weight_packed": torch.empty(
                output_size,
                input_size // self.pack_factor,
                dtype=torch.int32,
            ),
            # Runtime weight format expected by torch_npu.npu_weight_quant_batchmatmul:
            # int32 int4pack with shape [K, N/8].
            "weight_int4pack": torch.empty(
                input_size,
                output_size // self.pack_factor,
                dtype=torch.int32,
            ),
            "weight_g_idx": torch.empty(input_size, dtype=torch.int32),
            "weight_shape": torch.empty(2, dtype=torch.int64),
        }

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(output_size: int,
                             params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    def get_pergroup_param(self,
                           input_size: int,
                           output_size: int,
                           params_dtype: torch.dtype,
                           layer_type: Optional[str] = None) -> Dict[str, Any]:
        return {
            "weight_scale": torch.empty(
                output_size,
                input_size // self.group_size,
                dtype=params_dtype,
            ),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_shape = x.shape
        reshaped_x = x.reshape(-1, original_shape[-1])
        antiquant_scale = getattr(layer, "weight_scale_t", None)
        if antiquant_scale is None:
            antiquant_scale = layer.weight_scale.transpose(0, 1).contiguous()

        # NOTE: torch_npu's aclnnWeightQuantBatchMatmulV2 requires bias dtype
        # to be float32 (DT_FLOAT). Model biases are typically bf16/fp16.
        if bias is not None and bias.dtype is not torch.float32:
            bias = bias.to(dtype=torch.float32)

        output = torch_npu.npu_weight_quant_batchmatmul(
            reshaped_x,
            layer.weight_int4pack,
            antiquant_scale=antiquant_scale.to(reshaped_x.dtype),
            antiquant_group_size=self.group_size,
            bias=bias,
        )
        return output.reshape(*original_shape[:-1], -1)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # compressed-tensors stores `weight_packed` as int32 carrying int4 values,
        # packed along the input(K) axis: shape [N, K/8] for a Linear with weight
        # in the usual [N, K] layout.
        #
        # torch_npu.npu_weight_quant_batchmatmul expects `weight` as:
        # - int8  : shape [K, N]
        # - int4  : int32 from npu_convert_weight_to_int4pack, shape [K, N/8]
        #
        # Therefore we must convert checkpoint layout [N, K/8] -> runtime layout
        # [K, N/8].
        # Do NOT trust `weight_shape` blindly: for some model-parallel linear
        # wrappers the checkpoint's `weight_shape` can be inconsistent with the
        # actual `weight_packed/weight_scale/weight_g_idx` tensor shapes.
        #
        # compressed-tensors packs along the input(K) axis for Linear weights in
        # [N, K] layout: `weight_packed` is [N, K/8].
        packed = layer.weight_packed
        if packed.dim() != 2:
            raise ValueError(
                f"Invalid weight_packed dim for W4A16 linear: {packed.dim()}"
            )
        out_features = int(packed.shape[0])
        in_features = int(packed.shape[1]) * self.pack_factor

        # Unpack checkpoint int4 to int8 values in [-8, 7], shape [N, K].
        unpacked_w_int8 = unpack_from_int32(
            packed,
            torch.Size([out_features, in_features]),
            num_bits=self.num_bits,
            packed_dim=1,
        )
        # Use unpacked tensor as source of truth.
        out_features, in_features = int(unpacked_w_int8.shape[0]), int(
            unpacked_w_int8.shape[1])

        # Make sure `weight_scale` is in [N, K/group] orientation.
        if hasattr(layer, "weight_scale"):
            num_groups = in_features // self.group_size
            sc = layer.weight_scale
            if sc.dim() != 2:
                raise ValueError(
                    f"Invalid weight_scale dim for W4A16 linear: {sc.dim()}"
                )
            if sc.shape[0] == out_features and sc.shape[1] == num_groups:
                pass
            elif sc.shape[1] == out_features and sc.shape[0] == num_groups:
                layer.weight_scale.data = sc.transpose(0, 1).contiguous()
            else:
                raise ValueError(
                    "Unexpected weight_scale shape for W4A16 linear: "
                    f"got {tuple(sc.shape)}, expected ({out_features}, {num_groups}) "
                    f"or ({num_groups}, {out_features})"
                )

        # compressed-tensors may store a non-trivial `weight_g_idx` mapping
        # (aka act-order / non-contiguous group assignment). torch_npu's
        # npu_weight_quant_batchmatmul uses a fixed contiguous grouping defined
        # purely by `antiquant_group_size` and cannot consume `g_idx`.
        #
        # To preserve correctness, we dequantize with `g_idx` and then
        # re-quantize into contiguous groups of size `group_size`.
        if hasattr(layer, "weight_g_idx") and hasattr(layer, "weight_scale"):
            g_idx = layer.weight_g_idx.to(dtype=torch.long)
            if g_idx.numel() == in_features:
                expected_g_idx = (torch.arange(
                    in_features, device=g_idx.device, dtype=torch.long) //
                                  self.group_size)
            else:
                expected_g_idx = None
            if expected_g_idx is not None and not torch.equal(
                    g_idx, expected_g_idx):
                # Dequantize: w_fp32[n, k] = q[n, k] * scale[n, g_idx[k]]
                scale_fp32 = layer.weight_scale.to(dtype=torch.float32)
                idx = g_idx.view(1, -1).expand(out_features, -1)
                scale_per_col = torch.gather(scale_fp32, 1, idx)
                w_fp32 = unpacked_w_int8.to(dtype=torch.float32) * scale_per_col

                # Re-quantize into contiguous groups.
                num_groups = in_features // self.group_size
                new_scale_fp32 = torch.empty(
                    (out_features, num_groups),
                    device=w_fp32.device,
                    dtype=torch.float32,
                )
                q_new = torch.empty(
                    (out_features, in_features),
                    device=w_fp32.device,
                    dtype=torch.int8,
                )
                for group_id in range(num_groups):
                    start = group_id * self.group_size
                    end = start + self.group_size
                    block = w_fp32[:, start:end]
                    max_abs = block.abs().amax(dim=1)
                    # Symmetric int4 uses range [-8, 7]. We scale by 7 to
                    # minimize saturation for typical kernels.
                    s = (max_abs / 7.0).clamp(min=1e-8)
                    new_scale_fp32[:, group_id] = s
                    q_block = torch.round(block / s.unsqueeze(1)).clamp(
                        -8, 7).to(torch.int8)
                    q_new[:, start:end] = q_block

                # Update the layer scale to the contiguous-group version.
                layer.weight_scale.data = new_scale_fp32.to(
                    dtype=layer.weight_scale.dtype)
                layer.weight_scale_t = layer.weight_scale.transpose(0,
                                                                    1).contiguous()
                unpacked_w_int8 = q_new

        # Re-pack into torch_npu int4pack format along N (output) axis.
        # Input to npu_convert_weight_to_int4pack must be int32 with shape [K, N].
        w_k_n_int32 = unpacked_w_int8.transpose(0, 1).contiguous().to(torch.int32)
        weight_int4pack = torch_npu.npu_convert_weight_to_int4pack(w_k_n_int32)
        layer.weight_int4pack.data = weight_int4pack

        # Prepare per-group scale in expected shape: [K/group, N].
        layer.weight_scale_t = layer.weight_scale.transpose(0, 1).contiguous()

        from vllm.model_executor.utils import set_weight_attrs
        # Tag runtime packed weight as [K, N/8].
        for key in ("input_dim", "output_dim"):
            if hasattr(layer.weight_int4pack, key):
                try:
                    delattr(layer.weight_int4pack, key)
                except Exception:
                    pass
        set_weight_attrs(layer.weight_int4pack, {"input_dim": 0, "output_dim": 1})

        layer.weight_packed.data = layer.weight_packed.data.contiguous()
        layer.weight_scale.data = layer.weight_scale.data.contiguous()
        layer.weight_g_idx.data = layer.weight_g_idx.data.contiguous()
        layer.weight_shape.data = layer.weight_shape.data.contiguous()


class AscendW4A16FusedMoEMethod:
    """FusedMoe method for Ascend W4A16.
    """

    def __init__(self) -> None:
        self.transpose_weight = True
        self.num_bits = 4  # dtype = torch.int4
        self.pack_factor = 8  # pack 8 of torch.int4 tensors to torch.int32

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get(
            "group_size", 32)
        ascend_config = get_ascend_config()
        self.dynamic_eplb = ascend_config.dynamic_eplb or ascend_config.expert_map_record_path

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        assert intermediate_size_per_partition % self.pack_factor == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} "
            f"can be divided by `pack_factor` {self.pack_factor}")
        assert hidden_sizes % self.pack_factor == 0, (
            f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `pack_factor` "
            f"{self.pack_factor}")
        param_dict = {}

        param_dict["w13_weight_packed"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.pack_factor,
            dtype=torch.int32)
        param_dict["w2_weight_packed"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // self.pack_factor,
            dtype=torch.int32)

        return param_dict

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        assert intermediate_size_per_partition % self.group_size == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} "
            f"can be divided by `group_size` {self.group_size}")
        assert hidden_sizes % self.group_size == 0, (
            f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `group_size` "
            f"{self.group_size}")
        param_dict = {}

        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.group_size,
            dtype=torch.bfloat16)
        param_dict["w2_weight_scale"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // self.group_size,
            dtype=torch.bfloat16)
        param_dict["w13_weight_shape"] = torch.empty(num_experts,
                                                     2,
                                                     dtype=torch.int32)
        param_dict["w2_weight_shape"] = torch.empty(num_experts,
                                                    2,
                                                    dtype=torch.int32)
        param_dict["w13_weight_offset"] = torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.group_size,
            dtype=torch.bfloat16)
        param_dict["w2_weight_offset"] = torch.zeros(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // self.group_size,
            dtype=torch.bfloat16)

        return param_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
            1] == global_num_experts - global_redundant_expert_num, "Number of global experts mismatch (excluding redundancy)"

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts)

        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        return moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight_packed,
            w2=layer.w2_weight_packed,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_offset=layer.w13_weight_offset,
            w2_offset=layer.w2_weight_offset,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_int4_w4a16=True,
            expert_map=expert_map,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            dynamic_eplb=self.dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask", None))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.transpose_weight:
            w13_shape = layer.w13_weight_packed.data.shape
            w2_shape = layer.w2_weight_packed.data.shape
            unpacked_w13_weight = (unpack_from_int32(
                layer.w13_weight_packed.data.flatten(0, 1),
                torch.Size([
                    w13_shape[0] * w13_shape[1],
                    w13_shape[2] * self.pack_factor
                ]),
                self.num_bits,
            ).view(w13_shape[0], w13_shape[1],
                   -1).transpose(1, 2).contiguous().int())
            unpacked_w2_weight = (unpack_from_int32(
                layer.w2_weight_packed.data.flatten(0, 1),
                torch.Size([
                    w2_shape[0] * w2_shape[1], w2_shape[2] * self.pack_factor
                ]),
                self.num_bits,
            ).view(w2_shape[0], w2_shape[1],
                   -1).transpose(1, 2).contiguous().int())
            layer.w13_weight_packed.data = pack_to_int32(unpacked_w13_weight)
            layer.w2_weight_packed.data = pack_to_int32(unpacked_w2_weight)

            layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(
                1, 2).contiguous()
            layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(
                1, 2).contiguous()

            layer.w13_weight_offset.data = layer.w13_weight_offset.data.transpose(
                1, 2).contiguous()
            layer.w2_weight_offset.data = layer.w2_weight_offset.data.transpose(
                1, 2).contiguous()
