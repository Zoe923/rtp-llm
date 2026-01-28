import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    FusedMoeFactory,
    GroupTopK,
    LinearFactory,
    MlaAttention,
    RMSNorm,
    RMSResNorm,
    SelectTopk,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class GenericMoeLayer(nn.Module):
    """Generic MoE layer supporting both Qwen3 and internal model."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.inter_size
        self.num_experts = config.eplb_config.phy_exp_num(config.expert_num)
        self.top_k = config.moe_k

        # Get quant_config from model_config
        quant_config = config.quant_config
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, quant_config, hw_kernel_config
        )
        self.select_topk = SelectTopk(
            config, moe_config.fake_balance_expert, parallelism_config.dp_rank
        )

        # 检查双模式：只有在 ep_size > 1 时才有意义（确保会走 DeepEP）
        support_dual_mode_config = getattr(moe_config, "support_dual_mode", False)
        ep_size = parallelism_config.ep_size
        self.support_dual_mode = support_dual_mode_config and ep_size > 1

        logging.info(
            f"[GenericMoeLayer.__init__] support_dual_mode_config={support_dual_mode_config}, "
            f"ep_size={ep_size}, final support_dual_mode={self.support_dual_mode}"
        )

        if self.support_dual_mode:
            # ========== 双模式：创建两个 FusedMoe ==========
            import copy

            logging.info(
                "[GenericMoeLayer] Creating DUAL mode FusedMoe (Normal + LowLatency)"
            )

            # Normal 模式
            moe_config_normal = copy.deepcopy(moe_config)
            moe_config_normal.use_deepep_low_latency = False
            moe_config_normal.support_dual_mode = True

            config_adapter_normal = MoEConfigAdapter(
                model_config=config,
                parallelism_config=parallelism_config,
                moe_config=moe_config_normal,
                max_generate_batch_size=max_generate_batch_size,
                quant_config=quant_config,
                enable_cuda_graph=False,  # Normal 不支持 CUDA Graph
            )
            self.fused_moe_normal = FusedMoeFactory().create_fused_moe(
                config_adapter_normal, weights
            )
            logging.info(
                f"[GenericMoeLayer] Normal FusedMoe created: "
                f"{self.fused_moe_normal.router.__class__.__name__}"
            )

            # LowLatency 模式
            moe_config_lowlatency = copy.deepcopy(moe_config)
            moe_config_lowlatency.use_deepep_low_latency = True
            moe_config_lowlatency.support_dual_mode = True

            config_adapter_lowlatency = MoEConfigAdapter(
                model_config=config,
                parallelism_config=parallelism_config,
                moe_config=moe_config_lowlatency,
                max_generate_batch_size=max_generate_batch_size,
                quant_config=quant_config,
                enable_cuda_graph=enable_cuda_graph,  # LowLatency 支持 CUDA Graph
            )
            self.fused_moe_lowlatency = FusedMoeFactory().create_fused_moe(
                config_adapter_lowlatency, weights
            )
            logging.info(
                f"[GenericMoeLayer] LowLatency FusedMoe created: "
                f"{self.fused_moe_lowlatency.router.__class__.__name__}"
            )

            self.fused_moe = None
        else:
            # ========== 单模式（向后兼容）==========
            config_adapter = MoEConfigAdapter(
                model_config=config,
                parallelism_config=parallelism_config,
                moe_config=moe_config,
                max_generate_batch_size=max_generate_batch_size,
                quant_config=quant_config,
                enable_cuda_graph=enable_cuda_graph,
            )
            self.fused_moe = FusedMoeFactory().create_fused_moe(config_adapter, weights)
            self.fused_moe_normal = None
            self.fused_moe_lowlatency = None

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"
        self.num_local_experts = self.w1.shape[0]
        self.add_shared_expert = config.moe_style == 2
        if self.add_shared_expert:
            self.shared_expert = DenseMLP(
                config.activation_type, parallelism_config, weights, quant_config
            )
        else:
            self.shared_expert = None
        if weights.get(W.shared_expert_gate, None) is not None:
            self.shared_expert_gate = LinearFactory.create_linear_from_weights(
                weights, W.shared_expert_gate, None, None, config
            )
        else:
            self.shared_expert_gate = None

        # for group topk
        self.correction_bias = weights.get(W.e_score_correction_b, None)

    def forward(
        self, hidden_states: torch.Tensor, has_prefill_global: Optional[bool] = None
    ) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        router_logits = self.gate(hidden_states)
        router_logits_fp32 = router_logits.float()

        # 确定 topk_ids_dtype
        if self.support_dual_mode:
            # 双模式：使用 lowlatency 的 dtype
            topk_ids_dtype = self.fused_moe_lowlatency.topk_ids_dtype
        else:
            # 单模式
            topk_ids_dtype = self.fused_moe.topk_ids_dtype

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=topk_ids_dtype,
            device=hidden_states.device,
        )

        if self.correction_bias is not None:
            self.group_topk = GroupTopK()
            self.renormalize = self.config.has_moe_norm
            self.num_expert_group = self.config.moe_n_group

            self.topk_group = self.config.moe_topk_group
            self.n_routed_experts = self.config.expert_num  # config.n_routed_experts
            self.routed_scaling_factor = self.config.routed_scaling_factor
            self.group_topk(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                scores=router_logits_fp32,
                correction_bias=self.correction_bias,
                n_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
                renormalize=self.renormalize,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            # Top-K selection using C++ SelectTopkOp
            self.select_topk(router_logits_fp32, topk_ids, topk_weights)

        # ========== 动态选择 FusedMoe ==========
        if self.support_dual_mode:
            # 双模式：根据 has_prefill_global 选择
            if has_prefill_global is None:
                # 向后兼容：根据 token 数量推断
                has_prefill_global = num_tokens > 1
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(
                        f"[GenericMoeLayer] has_prefill_global not provided, "
                        f"inferred from num_tokens: {num_tokens} -> {has_prefill_global}"
                    )

            if has_prefill_global:
                # 有 prefill：使用 Normal 模式
                logging.info(
                    f"[GenericMoeLayer] Using Normal mode "
                    f"(has_prefill_global=True, num_tokens={num_tokens})"
                )
                experts_output = self.fused_moe_normal(
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    activation="SiGLU",
                )
            else:
                # 全部 decode：使用 LowLatency 模式
                logging.info(
                    f"[GenericMoeLayer] Using LowLatency mode "
                    f"(has_prefill_global=False, num_tokens={num_tokens})"
                )
                experts_output = self.fused_moe_lowlatency(
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    activation="SiGLU",
                )
        else:
            # 单模式
            experts_output = self.fused_moe(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation="SiGLU",
            )

        if self.shared_expert is not None:
            shared_expert_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_expert_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states))
                    * shared_expert_output
                )
            experts_output = experts_output + shared_expert_output
        return experts_output


class DecodeLayerOutput:
    def __init__(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        self.hidden_states = hidden_states
        self.residual = residual


class GenericMoeDecoderLayer(nn.Module):
    """Generic MoE decoder layer supporting Dense/MoE hybrid and shared experts."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Get quant_config from model_config
        quant_config = config.quant_config
        if config.attn_config.use_mla:
            self.self_attn = MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
            )
        else:
            attn_configs = config.getAttentionConfigs(parallelism_config.tp_size)
            self.self_attn = CausalAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
            )

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type, parallelism_config, weights, quant_config
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
            )

        # 使用 RMSResNorm 来 fuse residual add 和 layernorm
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
        has_prefill_global: Optional[bool] = None,
    ) -> DecodeLayerOutput:
        # equivalent to:
        # residual = residual + hidden_states
        # hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )

        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        hidden_states = self.post_attention_layernorm(hidden_states, residual)

        # MLP (Dense or MoE，shared expert 逻辑已经在 GenericMoeLayer 内部处理)
        if hasattr(self.mlp, "support_dual_mode") and self.mlp.support_dual_mode:
            # MoE 层且支持双模式：传递 has_prefill_global
            hidden_states = self.mlp(
                hidden_states, has_prefill_global=has_prefill_global
            )
        else:
            # Dense 层或单模式 MoE
            hidden_states = self.mlp(hidden_states)

        # 返回 mlp_output 和 residual，让下一层的 input_layernorm 来 fuse 最后的 add
        return DecodeLayerOutput(hidden_states, residual)


class GenericMoeModel(GptModelBase):
    """Generic MoE model supporting Qwen3-MoE, internal model, and other MoE architectures."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        # Determine attention_type from model_config.attn_config.use_mla
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

        # ========== DeepEP 双模式支持 ==========
        self.support_dual_mode = getattr(moe_config, "support_dual_mode", False)
        self.dp_size = parallelism_config.dp_size

        logging.info(
            f"[GenericMoeModel.__init__] support_dual_mode={self.support_dual_mode}, "
            f"dp_size={self.dp_size}, tp_size={parallelism_config.tp_size}, "
            f"ep_size={parallelism_config.ep_size}"
        )
        # ========== 双模式支持结束 ==========

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]
            fmha_impl.prepare(inputs.attention_inputs)

        # ========== DeepEP 双模式：获取 has_prefill_global ==========
        has_prefill_global = False
        if self.support_dual_mode:
            # 优先从 C++ 获取（性能最优）
            if hasattr(inputs, "has_prefill_global"):
                has_prefill_global = inputs.has_prefill_global
                logging.debug(
                    f"[GenericMoeModel] Using has_prefill_global from C++: "
                    f"{has_prefill_global}"
                )
            else:
                # 兜底：在 Python 层判断（以防 C++ 未设置）
                logging.warning(
                    "[GenericMoeModel] has_prefill_global not set by C++, "
                    "fallback to Python calculation using is_prefill"
                )

                # 使用 is_prefill 判断（与 C++ 层逻辑一致）
                local_is_prefill = inputs.attention_inputs.is_prefill

                # All-gather（如果多 DP）
                if self.dp_size > 1:
                    from rtp_llm.models_py.distributed import collective_torch
                    from rtp_llm.models_py.distributed.collective_torch import Group

                    local_tensor = torch.tensor(
                        [1 if local_is_prefill else 0], dtype=torch.int32, device="cuda"
                    )
                    gathered = collective_torch.all_gather(local_tensor, Group.DP)
                    has_prefill_global = gathered.sum().item() > 0
                else:
                    has_prefill_global = local_is_prefill

                logging.debug(
                    f"[GenericMoeModel] Python fallback: "
                    f"local_is_prefill={local_is_prefill}, "
                    f"has_prefill_global={has_prefill_global}"
                )
        # ========== has_prefill_global 获取结束 ==========

        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                has_prefill_global=has_prefill_global,
            )
            hidden_states = output.hidden_states
            residual = output.residual

        hidden_states = self.norm(hidden_states, residual)

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "GenericMoeLayer",
    "GenericMoeDecoderLayer",
    "GenericMoeModel",
]
