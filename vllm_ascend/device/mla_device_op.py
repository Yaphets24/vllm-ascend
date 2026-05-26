# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import torch
import torch_npu

from vllm_ascend.attention.utils import trans_rope_weight, transdata
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# token count limits within the mlapo operator
MLAPO_MAX_SUPPORTED_TOKENS = 1024

NZ_FMT_LAST_DIM = 16


class MLABaseDeviceAdaptor:
    """Device-specific MLA helpers.

    Each method takes the MLA impl object (`AscendMLAImpl`) so it can read
    quant-related buffers and config fields without `mla_v1.py` having to
    branch on device type.
    """

    # Activation dtype used by the FA-quant path.
    FA_QUANT_ACT_DTYPE: torch.dtype = torch.int8

    # ------------------------------------------------------------------
    # Weight post-processing
    # ------------------------------------------------------------------
    @staticmethod
    def process_weights_for_fused_mlapo(impl, act_dtype: torch.dtype) -> None:
        assert impl.fused_qkv_a_proj is not None
        assert impl.q_a_layernorm is not None
        assert impl.kv_a_layernorm is not None
        kv_a_proj_wt = impl.fused_qkv_a_proj.weight.data[..., impl.q_lora_rank:].contiguous()
        q_a_proj_wt = impl.fused_qkv_a_proj.weight.data[..., : impl.q_lora_rank].contiguous()
        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        kv_a_proj_wt = trans_rope_weight(kv_a_proj_wt, impl.qk_rope_head_dim)
        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        wd_qkv = torch.cat((kv_a_proj_wt, q_a_proj_wt), dim=-1)
        wd_qkv = wd_qkv.t().contiguous()
        wd_qkv = transdata(wd_qkv, block_size=(16, 32)).unsqueeze(0).contiguous()
        impl.wd_qkv = torch_npu.npu_format_cast(wd_qkv, 29)

        kv_a_proj_deq_scl = impl.fused_qkv_a_proj.deq_scale[impl.q_lora_rank:].contiguous()  # type: ignore[union-attr]
        q_a_proj_deq_scl = impl.fused_qkv_a_proj.deq_scale[: impl.q_lora_rank].contiguous()  # type: ignore[union-attr]
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.reshape(impl.kv_lora_rank + impl.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_deq_scl = trans_rope_weight(kv_a_proj_deq_scl, impl.qk_rope_head_dim)
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.view(impl.kv_lora_rank + impl.qk_rope_head_dim).contiguous()
        impl.deq_scale_qkv = torch.cat((kv_a_proj_deq_scl, q_a_proj_deq_scl), dim=-1).contiguous()

        kv_a_proj_qt_bias = impl.fused_qkv_a_proj.quant_bias[impl.q_lora_rank:].contiguous()  # type: ignore[union-attr]
        q_a_proj_qt_bias = impl.fused_qkv_a_proj.quant_bias[: impl.q_lora_rank].contiguous()  # type: ignore[union-attr]
        kv_a_proj_qt_bias = kv_a_proj_qt_bias.reshape(impl.kv_lora_rank + impl.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_qt_bias = trans_rope_weight(kv_a_proj_qt_bias, impl.qk_rope_head_dim)
        kv_a_proj_qt_bias = kv_a_proj_qt_bias.view(impl.kv_lora_rank + impl.qk_rope_head_dim).contiguous()
        impl.quant_bias_qkv = torch.cat((kv_a_proj_qt_bias, q_a_proj_qt_bias), dim=-1).contiguous()

        wu_q = impl.q_proj.weight.data
        wu_q = wu_q.t().reshape(impl.num_heads, impl.qk_nope_head_dim + impl.qk_rope_head_dim, -1)
        wu_q = trans_rope_weight(wu_q, impl.qk_rope_head_dim)
        wu_q = wu_q.reshape(impl.num_heads * (impl.qk_nope_head_dim + impl.qk_rope_head_dim), -1)
        wu_q = transdata(wu_q, block_size=(16, 32)).unsqueeze(0).contiguous()
        impl.wu_q = torch_npu.npu_format_cast(wu_q, 29)

        qb_deq_scl = impl.q_proj.deq_scale.data
        qb_deq_scl = qb_deq_scl.reshape(impl.num_heads, impl.qk_nope_head_dim + impl.qk_rope_head_dim, -1)
        qb_deq_scl = trans_rope_weight(qb_deq_scl, impl.qk_rope_head_dim)
        impl.qb_deq_scl = qb_deq_scl.reshape(impl.num_heads * (impl.qk_nope_head_dim + impl.qk_rope_head_dim))

        qb_qt_bias = impl.q_proj.quant_bias.data
        qb_qt_bias = qb_qt_bias.reshape(impl.num_heads, impl.qk_nope_head_dim + impl.qk_rope_head_dim, -1)
        qb_qt_bias = trans_rope_weight(qb_qt_bias, impl.qk_rope_head_dim)
        impl.qb_qt_bias = qb_qt_bias.reshape(impl.num_heads * (impl.qk_nope_head_dim + impl.qk_rope_head_dim))

        device = impl.q_proj.weight.device
        impl.gamma1 = impl.q_a_layernorm.weight.data  # type: ignore[union-attr]
        impl.beta1 = torch.zeros_like(impl.gamma1) if (_bias := impl.q_a_layernorm.bias) is None else _bias.data  # type: ignore[union-attr]
        impl.gamma2 = impl.kv_a_layernorm.weight.data  # type: ignore[union-attr]
        impl.quant_scale0 = impl.fused_qkv_a_proj.input_scale.data  # type: ignore[union-attr]
        impl.quant_offset0 = impl.fused_qkv_a_proj.input_offset.data  # type: ignore[union-attr]
        impl.quant_scale1 = impl.q_proj.input_scale.data
        impl.quant_offset1 = impl.q_proj.input_offset.data
        impl.ctkv_scale = torch.tensor([1], dtype=act_dtype, device=device)
        impl.q_nope_scale = torch.tensor([1], dtype=act_dtype, device=device)

        # On KV consumers (decode-only) MLAPO uses the transformed weights built above;
        # the original fused_qkv_a_proj/q_proj weights and quant params are no longer
        # referenced, so drop them to save memory.
        if (
            impl.vllm_config.kv_transfer_config is not None
            and impl.vllm_config.kv_transfer_config.is_kv_consumer
            and impl.vllm_config.scheduler_config.max_num_batched_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
        ):
            impl.fused_qkv_a_proj.weight = None  # type: ignore[union-attr]
            impl.fused_qkv_a_proj.deq_scale = None  # type: ignore[union-attr]
            impl.fused_qkv_a_proj.quant_bias = None  # type: ignore[union-attr]
            impl.q_proj.weight = None
            impl.q_proj.deq_scale = None
            impl.q_proj.quant_bias = None
            torch.npu.empty_cache()

    @staticmethod
    def process_weights_for_fa_quant(impl) -> None:
        impl.gamma1 = impl.q_a_layernorm.weight.data  # type: ignore[union-attr]
        impl.gamma2 = impl.kv_a_layernorm.weight.data  # type: ignore[union-attr]
        wu_q = impl.q_proj.weight.data
        impl.wu_q = wu_q
        q_a_proj_fa3 = impl.fused_qkv_a_proj.weight.data[..., : impl.q_lora_rank].contiguous()  # type: ignore[union-attr]
        impl.wd_q = q_a_proj_fa3
        kv_a_proj_fa3 = impl.fused_qkv_a_proj.weight.data[..., impl.q_lora_rank:].contiguous()  # type: ignore[union-attr]
        impl.wd_kv = kv_a_proj_fa3
        impl.dequant_scale_w_uq_qr = impl.q_proj.weight_scale.data.view(1, -1).to(torch.float)
        q_a_proj_deq_scl = impl.fused_qkv_a_proj.weight_scale[: impl.q_lora_rank].contiguous()  # type: ignore[union-attr]
        impl.dequant_scale_w_dq = q_a_proj_deq_scl.view(1, -1).to(torch.float)
        kv_a_proj_deq_scl = impl.fused_qkv_a_proj.weight_scale[impl.q_lora_rank:].contiguous()  # type: ignore[union-attr]
        impl.dequant_scale_w_dkv_kr = kv_a_proj_deq_scl.view(1, -1).to(torch.float)
        layer = impl.vllm_config.compilation_config.static_forward_context[impl.layer_name]
        impl.quant_kscale = layer.quant_kscale
        impl.fak_descale_float = layer.fak_descale_float

    # ------------------------------------------------------------------
    # Chunked-context helpers
    # ------------------------------------------------------------------
    @staticmethod
    def make_chunked_kv_buffers(
        impl,
        toks: int,
        num_heads: int,
        latent_kv_dim: int,
        rope_dim: int,
        cache_kv_c: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_c_normed = torch.empty(
            toks, num_heads, latent_kv_dim, dtype=q_nope.dtype, device=q_nope.device
        )
        k_pe = torch.empty(toks, num_heads, rope_dim, dtype=q_nope.dtype, device=q_nope.device)
        return kv_c_normed, k_pe

    @staticmethod
    def maybe_dequant_chunked_kv_c(impl, kv_c_normed: torch.Tensor) -> torch.Tensor:
        return kv_c_normed

    # ------------------------------------------------------------------
    # KV-cache write-back (rmsnorm + rope)
    # ------------------------------------------------------------------
    @staticmethod
    def get_kv_rmsnorm_rope_scale(impl) -> torch.Tensor | None:
        return None

    # ------------------------------------------------------------------
    # Decode-path layout helpers
    # ------------------------------------------------------------------
    @staticmethod
    def reshape_decode_kv_for_attn(
        impl,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        enable_kv_nz: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if impl.fa_quant_layer:
            # FA-quant on the base device packs kv_lora_rank as `NZ_FMT_LAST_DIM * 2`.
            k_nope = k_nope.view(
                -1,
                impl.num_kv_heads,
                impl.kv_lora_rank // (NZ_FMT_LAST_DIM * 2),
                block_size,
                NZ_FMT_LAST_DIM * 2,
            )
            k_pe = k_pe.view(
                -1,
                impl.num_kv_heads,
                impl.qk_rope_head_dim // NZ_FMT_LAST_DIM,
                block_size,
                NZ_FMT_LAST_DIM,
            )
            return k_nope, k_pe
        return MLABaseDeviceAdaptor._reshape_decode_kv_default(
            impl, k_nope, k_pe, block_size, enable_kv_nz
        )

    @staticmethod
    def _reshape_decode_kv_default(
        impl,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        enable_kv_nz: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if enable_kv_nz:
            k_nope = k_nope.view(
                -1, impl.num_kv_heads, impl.kv_lora_rank // NZ_FMT_LAST_DIM, block_size, NZ_FMT_LAST_DIM
            )
            k_pe = k_pe.view(
                -1, impl.num_kv_heads, impl.qk_rope_head_dim // NZ_FMT_LAST_DIM, block_size, NZ_FMT_LAST_DIM
            )
        else:
            k_nope = k_nope.view(-1, impl.num_kv_heads, block_size, impl.kv_lora_rank)
            k_pe = k_pe.view(-1, impl.num_kv_heads, block_size, impl.qk_rope_head_dim)
        return k_nope, k_pe

    @staticmethod
    def configure_decode_fa_quant_layout(
        impl,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        dequant_scale_q_nope: torch.Tensor,
        num_tokens: int,
    ) -> tuple:
        """Reshape q_nope/q_pe/dequant_scale for the FA-quant decode path.

        Returns a tuple of:
            (q_nope, q_pe, dequant_scale_q_nope,
             input_layout, attn_mask, sparse_mode, actual_seq_lengths,
             attn_output_shape)
        """
        q_nope = q_nope.view(num_tokens, 1, impl.num_heads, -1).contiguous()
        q_pe = q_pe.view(num_tokens, 1, impl.num_heads, -1).contiguous()
        dequant_scale_q_nope = dequant_scale_q_nope.view(num_tokens, 1, impl.num_heads)
        input_layout = "BSND_NBSD"
        attn_mask = None
        sparse_mode = 0
        actual_seq_lengths = None
        attn_output_shape = (impl.num_heads, num_tokens, 1, impl.kv_lora_rank)
        return (
            q_nope,
            q_pe,
            dequant_scale_q_nope,
            input_layout,
            attn_mask,
            sparse_mode,
            actual_seq_lengths,
            attn_output_shape,
        )

    # ------------------------------------------------------------------
    # Decode-path Q quantization
    # ------------------------------------------------------------------
    @staticmethod
    def quantize_decode_q(
        impl,
        decode_ql_nope: torch.Tensor,
        decode_q_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return decode_ql_nope, decode_q_pe, None


class MLAFp8DeviceAdaptor(MLABaseDeviceAdaptor):
    """Adaptor for devices that quantize FA activations to float8_e4m3fn
    and run the MXFP8 MLAPO weight layout."""

    FA_QUANT_ACT_DTYPE = torch.float8_e4m3fn

    @staticmethod
    def process_weights_for_fused_mlapo(impl, act_dtype: torch.dtype) -> None:
        assert impl.fused_qkv_a_proj is not None

        weight_dq = impl.fused_qkv_a_proj.weight.data[..., : impl.q_lora_rank].contiguous()
        impl.weight_dq = torch_npu.npu_format_cast(weight_dq, 29)

        weight_uq_qr = impl.q_proj.weight.data.contiguous()
        impl.weight_uq_qr_scale = impl.q_proj.weight_scale.data.transpose(0, 1)
        impl.weight_uq_qr_scale = impl.weight_uq_qr_scale.reshape(
            -1, impl.weight_uq_qr_scale.shape[1] * impl.weight_uq_qr_scale.shape[2]
        )
        impl.weight_uq_qr = torch_npu.npu_format_cast(weight_uq_qr, 29)

        weight_dkv_kr = impl.fused_qkv_a_proj.weight.data[..., impl.q_lora_rank:].contiguous()
        impl.weight_dkv_kr = torch_npu.npu_format_cast(weight_dkv_kr, 29)

        weight_scale = impl.fused_qkv_a_proj.weight_scale
        weight_scale = weight_scale.transpose(0, 1)
        weight_scale = weight_scale.reshape(-1, weight_scale.shape[1] * weight_scale.shape[2])
        impl.weight_dq_scale = weight_scale[: impl.q_lora_rank, ...]
        impl.weight_dkv_kr_scale = weight_scale[impl.q_lora_rank:, ...]
        if impl.fa_quant_layer:
            layer = impl.vllm_config.compilation_config.static_forward_context[impl.layer_name]
            impl.quant_kscale = layer.quant_kscale
            impl.fak_descale_float = layer.fak_descale_float
            impl.fak_descale_reciprocal = layer.fak_descale_reciprocal

    @staticmethod
    def process_weights_for_fa_quant(impl) -> None:
        layer = impl.vllm_config.compilation_config.static_forward_context[impl.layer_name]
        impl.fak_descale_float = layer.fak_descale_float
        impl.quant_kscale = layer.quant_kscale
        impl.fak_descale_reciprocal = layer.fak_descale_reciprocal

    @staticmethod
    def make_chunked_kv_buffers(
        impl,
        toks: int,
        num_heads: int,
        latent_kv_dim: int,
        rope_dim: int,
        cache_kv_c: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if impl.fa_quant_layer:
            kv_c_normed = torch.empty(
                toks, num_heads, latent_kv_dim, dtype=cache_kv_c.dtype, device=cache_kv_c.device
            )
            k_pe = torch.empty(toks, num_heads, rope_dim, dtype=q_pe.dtype, device=q_pe.device)
            return kv_c_normed, k_pe
        return MLABaseDeviceAdaptor.make_chunked_kv_buffers(
            impl, toks, num_heads, latent_kv_dim, rope_dim, cache_kv_c, q_nope, q_pe
        )

    @staticmethod
    def maybe_dequant_chunked_kv_c(impl, kv_c_normed: torch.Tensor) -> torch.Tensor:
        if impl.fa_quant_layer:
            return torch.mul(
                kv_c_normed.to(impl.fak_descale_float.dtype), impl.fak_descale_float
            ).to(torch.bfloat16)
        return kv_c_normed

    @staticmethod
    def get_kv_rmsnorm_rope_scale(impl) -> torch.Tensor | None:
        if impl.fa_quant_layer:
            return impl.fak_descale_reciprocal
        return None

    @staticmethod
    def reshape_decode_kv_for_attn(
        impl,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        enable_kv_nz: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # This adaptor skips the fa_quant NZ-double reshape used by the base device.
        return MLABaseDeviceAdaptor._reshape_decode_kv_default(
            impl, k_nope, k_pe, block_size, enable_kv_nz
        )

    @staticmethod
    def configure_decode_fa_quant_layout(
        impl,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        dequant_scale_q_nope: torch.Tensor,
        num_tokens: int,
    ) -> tuple:
        q_nope = q_nope.view(num_tokens, impl.num_heads, 1, -1).contiguous()
        q_pe = q_pe.view(num_tokens, impl.num_heads, 1, -1)
        dequant_scale_q_nope = dequant_scale_q_nope.view(num_tokens, impl.num_heads, 1)
        attn_mask = None
        input_layout = "BNSD"
        sparse_mode = 0
        actual_seq_lengths = None
        attn_output_shape = (num_tokens, impl.num_heads, 1, impl.kv_lora_rank)
        return (
            q_nope,
            q_pe,
            dequant_scale_q_nope,
            input_layout,
            attn_mask,
            sparse_mode,
            actual_seq_lengths,
            attn_output_shape,
        )

    @staticmethod
    def quantize_decode_q(
        impl,
        decode_ql_nope: torch.Tensor,
        decode_q_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not impl.fa_quant_layer:
            return decode_ql_nope, decode_q_pe, None
        decode_ql_nope, dequant_scale_q_nope = torch_npu.npu_dynamic_quant(
            decode_ql_nope, dst_type=torch.float8_e4m3fn
        )
        decode_q_pe = (
            decode_q_pe / dequant_scale_q_nope.unsqueeze(-1) / impl.fak_descale_float
        ).to(torch.bfloat16)
        return decode_ql_nope, decode_q_pe, dequant_scale_q_nope


def _get_mla_device_adaptor() -> type[MLABaseDeviceAdaptor]:
    if get_ascend_device_type() == AscendDeviceType.A5:
        return MLAFp8DeviceAdaptor
    return MLABaseDeviceAdaptor


MLADeviceOperator: type[MLABaseDeviceAdaptor] = _get_mla_device_adaptor()
