"""DeepEP wrapper with singleton pattern and simplified configuration.

This module provides a unified interface for DeepEP initialization and management,
combining the functionality of the previous DeepEPInitializer and DeepEPWrapper classes.
"""

import gc
import logging
import os
import platform
import threading
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, Tuple

import torch
from deep_ep import Buffer as DeepEPBuffer
from deep_ep import Config as DeepEPConfig
from torch.distributed import ProcessGroup

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops.compute_ops import DeviceType, get_device

__all__ = [
    "DeepepWrapperConfig",
    "DeepEPWrapper",
    "DeepEPBuffer",
    "DeepEPConfig",
    "DeepEPMode",
    "use_accl_ep",
]


def use_accl_ep() -> bool:
    """Check if ACCL EP should be used based on device type."""
    device_type = get_device().get_device_type()
    return not device_type == DeviceType.ROCm


def allow_mnnvl() -> bool:
    """Check if MNNVL is allowed based on architecture and GPU capability."""
    is_sm_100 = torch.cuda.get_device_capability()[0] in [10]
    return "aarch64" in platform.machine() and is_sm_100


class DeepEPMode(IntEnum):
    """The mode of deep_ep."""

    NORMAL = auto()
    LOW_LATENCY = auto()
    LOW_LATENCY_M2N = auto()
    DUAL = auto()  # 新增：双模式（同时支持 Normal 和 LowLatency）


@dataclass
class DeepepWrapperConfig:
    """Simplified configuration for DeepEP containing only required parameters.

    This class extracts only the necessary parameters from MoEConfigAdapter
    to reduce coupling and make configuration comparison easier.
    """

    # Parallelism parameters
    ep_rank: int
    ep_size: int
    tp_size: int
    local_rank: int
    world_size: int

    # Model parameters
    hidden_size: int
    expert_num: int
    moe_k: int

    # MoE-specific parameters
    deep_ep_num_sm: int
    use_deepep_low_latency: bool
    use_deepep_internode: bool

    # Generation parameters
    max_generate_batch_size: int

    # FFN disaggregate parameters (optional)
    enable_ffn_disaggregate: bool = False
    attention_tp_size: int = 0
    attention_dp_size: int = 0
    ffn_tp_size: int = 0
    ffn_dp_size: int = 0
    ll_num_max_token_per_rank: int = 0

    # 新增：双模式支持
    support_dual_mode: bool = False

    @classmethod
    def from_config_adapter(
        cls, config_adapter: MoEConfigAdapter, ll_num_max_token_per_rank: int = 0
    ) -> "DeepepWrapperConfig":
        """Create DeepepWrapperConfig from MoEConfigAdapter.

        Args:
            config_adapter: The full configuration adapter

        Returns:
            A new DeepepWrapperConfig instance with extracted parameters
        """
        model_config = config_adapter.model_config
        parallelism_config = config_adapter.parallelism_config
        moe_config = config_adapter.moe_config
        ffn_config = parallelism_config.ffn_disaggregate_config

        # 获取 support_dual_mode 配置
        support_dual_mode = getattr(moe_config, "support_dual_mode", False)

        return cls(
            # Parallelism parameters
            ep_rank=parallelism_config.ep_rank,
            ep_size=parallelism_config.ep_size,
            tp_size=parallelism_config.tp_size,
            local_rank=parallelism_config.local_rank,
            world_size=parallelism_config.world_size,
            # Model parameters
            hidden_size=model_config.hidden_size,
            expert_num=model_config.expert_num,
            moe_k=model_config.moe_k,
            # MoE-specific parameters
            deep_ep_num_sm=moe_config.deep_ep_num_sm,
            use_deepep_low_latency=moe_config.use_deepep_low_latency,
            use_deepep_internode=moe_config.use_deepep_internode,
            # Generation parameters
            max_generate_batch_size=config_adapter.max_generate_batch_size,
            # FFN disaggregate parameters
            enable_ffn_disaggregate=(
                ffn_config.enable_ffn_disaggregate if ffn_config else False
            ),
            attention_tp_size=(ffn_config.attention_tp_size if ffn_config else 0),
            attention_dp_size=(ffn_config.attention_dp_size if ffn_config else 0),
            ffn_tp_size=(ffn_config.ffn_tp_size if ffn_config else 0),
            ffn_dp_size=(ffn_config.ffn_dp_size if ffn_config else 0),
            ll_num_max_token_per_rank=ll_num_max_token_per_rank,
            # 新增：双模式支持
            support_dual_mode=support_dual_mode,
        )

    def equal(self, other: "DeepepWrapperConfig") -> bool:
        """Compare if two DeepepWrapperConfig instances are equal.

        Args:
            other: Another DeepepWrapperConfig instance to compare with

        Returns:
            True if all parameters are equal, False otherwise

        Note:
            In dual mode, use_deepep_low_latency differences are ignored because
            dual mode needs both Normal and LowLatency configurations.
        """
        logging.info(
            f"[DeepepWrapperConfig.equal] ENTRY: self.support_dual_mode={self.support_dual_mode}, "
            f"other.support_dual_mode={other.support_dual_mode}"
        )

        # 双模式下：忽略 use_deepep_low_latency 和 ll_num_max_token_per_rank 的差异
        # 因为双模式需要同时创建 Normal (low_latency=False, ll_num=0) 和
        # LowLatency (low_latency=True, ll_num=计算值) 两个 FusedMoe
        if self.support_dual_mode and other.support_dual_mode:
            check_low_latency = True  # 双模式下不检查这个字段的差异
            check_ll_num = True  # 双模式下也不检查这个字段
            logging.info(
                f"[DeepepWrapperConfig.equal] Dual mode branch: ignoring differences - "
                f"use_deepep_low_latency(self={self.use_deepep_low_latency}, other={other.use_deepep_low_latency}), "
                f"ll_num_max_token_per_rank(self={self.ll_num_max_token_per_rank}, other={other.ll_num_max_token_per_rank})"
            )
        else:
            check_low_latency = (
                self.use_deepep_low_latency == other.use_deepep_low_latency
            )
            check_ll_num = (
                self.ll_num_max_token_per_rank == other.ll_num_max_token_per_rank
            )
            logging.info(
                f"[DeepepWrapperConfig.equal] Single mode branch: checking all fields "
                f"(low_latency_match={check_low_latency}, ll_num_match={check_ll_num})"
            )

        result = (
            self.ep_rank == other.ep_rank
            and self.ep_size == other.ep_size
            and self.tp_size == other.tp_size
            and self.local_rank == other.local_rank
            and self.world_size == other.world_size
            and self.hidden_size == other.hidden_size
            and self.expert_num == other.expert_num
            and self.moe_k == other.moe_k
            and self.deep_ep_num_sm == other.deep_ep_num_sm
            and check_low_latency  # 双模式下忽略此字段差异
            and self.use_deepep_internode == other.use_deepep_internode
            and self.max_generate_batch_size == other.max_generate_batch_size
            and self.enable_ffn_disaggregate == other.enable_ffn_disaggregate
            and self.attention_tp_size == other.attention_tp_size
            and self.attention_dp_size == other.attention_dp_size
            and self.ffn_tp_size == other.ffn_tp_size
            and self.ffn_dp_size == other.ffn_dp_size
            and check_ll_num  # 双模式下忽略此字段差异
            and self.support_dual_mode == other.support_dual_mode
        )

        logging.info(f"[DeepepWrapperConfig.equal] Result: {result}")
        return result

    def __str__(self) -> str:
        """Return a string representation of the DeepepWrapperConfig."""
        return f"DeepepWrapperConfig(ep_rank={self.ep_rank}, ep_size={self.ep_size}, tp_size={self.tp_size}, local_rank={self.local_rank}, world_size={self.world_size}, hidden_size={self.hidden_size}, expert_num={self.expert_num}, moe_k={self.moe_k}, deep_ep_num_sm={self.deep_ep_num_sm}, use_deepep_low_latency={self.use_deepep_low_latency}, use_deepep_internode={self.use_deepep_internode}, max_generate_batch_size={self.max_generate_batch_size}, enable_ffn_disaggregate={self.enable_ffn_disaggregate}, attention_tp_size={self.attention_tp_size}, attention_dp_size={self.attention_dp_size}, ffn_tp_size={self.ffn_tp_size}, ffn_dp_size={self.ffn_dp_size}, support_dual_mode={self.support_dual_mode})"


class DeepEPWrapper:
    """Unified DeepEP wrapper with singleton pattern (thread-safe).

    This class combines the functionality of DeepEPInitializer and DeepEPWrapper,
    providing both initialization management and DeepEP functionality.
    """

    _instance: Optional["DeepEPWrapper"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __init__(self, group: ProcessGroup, config: DeepepWrapperConfig) -> None:
        """Initialize DeepEPWrapper with ProcessGroup and DeepepWrapperConfig.

        Note: Use get_instance() instead of calling this directly.

        Args:
            group: ProcessGroup for distributed communication
            config: DeepepWrapperConfig containing all necessary configuration
        """
        self._config = config
        self._use_accl_ep = use_accl_ep()

        # 初始化 buffer(s)（支持双模式）
        self._mode, self._normal_buffer, self._lowlatency_buffer = (
            self._init_deepep_buffers(group)
        )

    @classmethod
    def supported(cls) -> bool:
        """Check if DeepEP is supported on current device.

        Returns:
            True if DeepEP is supported, False otherwise
        """
        try:
            import deep_ep

            return True
        except ImportError:
            return False

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if DeepEP is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return cls._initialized

    @classmethod
    def get_instance(
        cls,
        config: DeepepWrapperConfig,
        group: Optional[ProcessGroup] = None,
    ) -> "DeepEPWrapper":
        """Ensure DeepEP is initialized with given config (thread-safe).

        If already initialized with a different config, raises an error.

        Args:
            config: DeepepWrapperConfig to initialize with
            group: ProcessGroup (if not provided, uses torch.distributed.group.WORLD)

        Raises:
            RuntimeError: If DeepEP is not supported or config mismatch
        """
        with cls._lock:
            if cls._initialized:
                if cls._instance is None:
                    raise RuntimeError("DeepEP state is inconsistent")

                # Debug: 打印 support_dual_mode 值
                origin_config = cls._instance._config
                logging.info(
                    f"[DeepEPWrapper.get_instance] Checking config equality: "
                    f"origin.support_dual_mode={origin_config.support_dual_mode}, "
                    f"new.support_dual_mode={config.support_dual_mode}, "
                    f"origin.use_deepep_low_latency={origin_config.use_deepep_low_latency}, "
                    f"new.use_deepep_low_latency={config.use_deepep_low_latency}"
                )

                if not origin_config.equal(config):
                    raise RuntimeError(
                        "DeepEP already initialized with different config, origin: {}, new: {}".format(
                            origin_config, config
                        )
                    )

                return cls._instance

            if not cls.supported():
                raise RuntimeError("DeepEP is not supported on this device")

            if not torch.distributed.is_initialized():
                raise RuntimeError("Distributed environment is not initialized")

            if group is None:
                group = torch.distributed.group.WORLD

            cls._instance = cls(group, config)  # type: ignore
            cls._initialized = True
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset DeepEP singleton state (for testing only).

        Warning: This should only be used in tests.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._destroy_buffer()
                cls._instance = None
            cls._initialized = False

    @property
    def buffer(self) -> DeepEPBuffer:
        """Get the DeepEP buffer (向后兼容).

        如果是双模式，默认返回 lowlatency buffer。
        推荐使用 get_buffer(use_low_latency) 显式指定。

        Returns:
            The initialized DeepEP buffer
        """
        if self._mode == DeepEPMode.DUAL:
            if self._lowlatency_buffer is None:
                raise RuntimeError("LowLatency buffer is not initialized")
            return self._lowlatency_buffer
        elif self._mode == DeepEPMode.NORMAL:
            if self._normal_buffer is None:
                raise RuntimeError("Normal buffer is not initialized")
            return self._normal_buffer
        else:
            if self._lowlatency_buffer is None:
                raise RuntimeError("LowLatency buffer is not initialized")
            return self._lowlatency_buffer

    def get_buffer(self, use_low_latency: bool) -> DeepEPBuffer:
        """根据模式获取对应的 buffer.

        Args:
            use_low_latency: True 获取 LowLatency buffer，False 获取 Normal buffer

        Returns:
            对应的 DeepEP buffer

        Raises:
            RuntimeError: 如果请求的 buffer 未初始化
        """
        # Dual mode: 根据参数返回对应的buffer
        if self._mode == DeepEPMode.DUAL:
            if use_low_latency:
                if self._lowlatency_buffer is None:
                    raise RuntimeError("LowLatency buffer not initialized in DUAL mode")
                logging.info(
                    f"[DeepEPWrapper.get_buffer] Dual mode: returning LowLatency buffer "
                    f"(id={id(self._lowlatency_buffer)})"
                )
                return self._lowlatency_buffer
            else:
                if self._normal_buffer is None:
                    raise RuntimeError("Normal buffer not initialized in DUAL mode")
                logging.info(
                    f"[DeepEPWrapper.get_buffer] Dual mode: returning Normal buffer "
                    f"(id={id(self._normal_buffer)})"
                )
                return self._normal_buffer

        if use_low_latency:
            if self._lowlatency_buffer is None:
                raise RuntimeError(
                    f"LowLatency buffer not initialized. Current mode: {self._mode}"
                )
            logging.info(
                f"[DeepEPWrapper.get_buffer] Returning LowLatency buffer (id={id(self._lowlatency_buffer)})"
            )
            return self._lowlatency_buffer
        else:
            if self._normal_buffer is None:
                raise RuntimeError(
                    f"Normal buffer not initialized. Current mode: {self._mode}"
                )
            logging.info(
                f"[DeepEPWrapper.get_buffer] Returning Normal buffer (id={id(self._normal_buffer)})"
            )
            return self._normal_buffer

    @property
    def mode(self) -> DeepEPMode:
        """Get the DeepEP mode."""
        return self._mode

    @property
    def config(self) -> DeepepWrapperConfig:
        """Get the DeepEP configuration."""
        return self._config

    @property
    def ep_rank(self) -> int:
        """Get expert parallel rank."""
        return self._config.ep_rank

    @property
    def ep_size(self) -> int:
        """Get expert parallel size."""
        return self._config.ep_size

    @property
    def hidden_size(self) -> int:
        """Get hidden size."""
        return self._config.hidden_size

    @property
    def num_experts(self) -> int:
        """Get number of experts."""
        return self._config.expert_num

    @property
    def num_topk(self) -> int:
        """Get top-k value."""
        return self._config.moe_k

    @property
    def ll_num_max_token_per_rank(self) -> int:
        """Get max tokens per rank for low-latency mode."""
        return self._config.ll_num_max_token_per_rank

    @property
    def num_sms(self) -> int:
        """Get number of SMs."""
        return self._config.deep_ep_num_sm

    @property
    def use_accl_ep(self) -> bool:
        """Check if ACCL EP is used."""
        return self._use_accl_ep

    def _init_deepep_buffers(
        self, group: ProcessGroup
    ) -> Tuple[DeepEPMode, Optional[DeepEPBuffer], Optional[DeepEPBuffer]]:
        """Initialize DeepEP buffer(s) based on configuration.

        Args:
            group: ProcessGroup for distributed communication

        Returns:
            Tuple of (DeepEPMode, normal_buffer, lowlatency_buffer)
        """
        config = self._config
        normal_buffer = None
        lowlatency_buffer = None

        if config.support_dual_mode:
            # ========== 双模式：Python创建2个独立buffer ==========
            # C++在dual mode下跳过buffer创建，由Python管理
            logging.info(
                f"[DeepEPWrapper] rank={config.ep_rank}: Initializing DUAL mode "
                f"(Python creates Normal + LowLatency buffers)"
            )

            # 创建Normal buffer
            normal_buffer = self._init_normal_buffer(group)
            logging.info(
                f"[DeepEPWrapper] rank={config.ep_rank}: Normal buffer initialized"
            )

            # 计算LowLatency buffer的ll_num
            ll_num = (
                config.max_generate_batch_size + config.tp_size - 1
            ) // config.tp_size
            matched_tokens = [64, 128]
            for t in matched_tokens:
                if ll_num <= t:
                    ll_num = t
                    break
            else:
                ll_num = 128

            logging.info(
                f"[DeepEPWrapper] rank={config.ep_rank}: Calculated ll_num={ll_num} "
                f"for LowLatency buffer (max_batch={config.max_generate_batch_size})"
            )

            # 创建LowLatency buffer
            lowlatency_buffer = self._init_low_latency_buffer(group, ll_num)
            logging.info(
                f"[DeepEPWrapper] rank={config.ep_rank}: LowLatency buffer initialized"
            )

            return DeepEPMode.DUAL, normal_buffer, lowlatency_buffer

        elif config.use_deepep_low_latency and config.enable_ffn_disaggregate:
            if self._use_accl_ep:
                lowlatency_buffer = self._init_low_latency_m2n_buffer(group)
                return DeepEPMode.LOW_LATENCY_M2N, None, lowlatency_buffer
            else:
                raise RuntimeError(
                    f"[rank: {config.ep_rank}] init deep_ep buffer failed, "
                    f"current deep_ep provider does not support "
                    f"use_deepep_low_latency and enable_ffn_disaggregate"
                )
        elif config.use_deepep_low_latency and not config.enable_ffn_disaggregate:
            lowlatency_buffer = self._init_low_latency_buffer(group)
            return DeepEPMode.LOW_LATENCY, None, lowlatency_buffer
        elif not config.use_deepep_low_latency and not config.enable_ffn_disaggregate:
            normal_buffer = self._init_normal_buffer(group)
            return DeepEPMode.NORMAL, normal_buffer, None
        else:
            raise RuntimeError(
                f"[rank: {config.ep_rank}] init deep_ep buffer failed, "
                f"unsupported configuration: "
                f"use_deepep_low_latency={config.use_deepep_low_latency}, "
                f"enable_ffn_disaggregate={config.enable_ffn_disaggregate}"
            )

    def _init_normal_buffer(self, group: ProcessGroup) -> DeepEPBuffer:
        """Initialize buffer for normal mode."""
        config = self._config
        num_nvl_bytes = 0
        num_rdma_bytes = 0
        num_qps_per_rank = 1

        # Normal-kernel internode
        if config.use_deepep_internode:
            num_nvl_bytes = int(2e9)
            num_rdma_bytes = int(1e9)
            # Normal IBGDA
            if os.environ.get("ACCL_NORMAL_MODE", "IBRC") == "IBGDA":
                os.environ["ACCL_NORMAL_MODE"] = "IBGDA"
                num_qps_per_rank = max(
                    config.deep_ep_num_sm // 2, int(config.expert_num / config.ep_size)
                )
            # Normal IBRC
            else:
                os.environ["ACCL_NORMAL_MODE"] = "IBRC"
                num_qps_per_rank = config.deep_ep_num_sm // 2
        # Normal-kernel intranode
        else:
            num_nvl_bytes = int(2e9)
            num_qps_per_rank = 1

        init_kwargs = {
            "group": group,
            "num_nvl_bytes": num_nvl_bytes,
            "num_rdma_bytes": num_rdma_bytes,
            "low_latency_mode": False,
            "num_qps_per_rank": num_qps_per_rank,
            "explicitly_destroy": True,  # 双模式下显式管理资源，避免析构冲突
        }

        if self._use_accl_ep:
            # Normal模式：显式设置ACCL_LOW_LATENCY_OPTIMIZE=0
            # 防止被LowLatency buffer创建时的环境变量污染
            os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "0"
            init_kwargs["allow_nvlink_for_low_latency_mode"] = True
            if allow_mnnvl():
                init_kwargs["allow_mnnvl"] = True
                init_kwargs["use_fabric"] = True
            else:
                init_kwargs["allow_mnnvl"] = False

        return DeepEPBuffer(**init_kwargs)  # type: ignore

    def _init_low_latency_buffer(
        self, group: ProcessGroup, ll_num_override: int = 0
    ) -> DeepEPBuffer:
        """Initialize buffer for low-latency mode.

        Args:
            group: ProcessGroup for communication
            ll_num_override: 如果非0，使用此值而不是config中的值（避免修改config）
        """
        config = self._config
        # 使用传入的值，或config中的值
        ll_num = (
            ll_num_override if ll_num_override > 0 else config.ll_num_max_token_per_rank
        )

        num_rdma_bytes = DeepEPBuffer.get_low_latency_rdma_size_hint(
            ll_num,
            config.hidden_size,
            config.ep_size,
            config.expert_num,
        )

        if config.local_rank == 0:
            print(
                f"Allocating buffer size: {num_rdma_bytes / 1e6} MB, "
                f"ll_num_max_token_per_rank: {ll_num}, "
                f"hidden_size: {config.hidden_size}, "
                f"ep_size: {config.ep_size}, "
                f"num_experts: {config.expert_num}",
                flush=True,
            )

        num_qps_per_rank = config.expert_num / config.ep_size

        init_kwargs = {
            "group": group,
            "num_nvl_bytes": 0,
            "num_rdma_bytes": num_rdma_bytes,
            "low_latency_mode": True,
            "num_qps_per_rank": num_qps_per_rank,
            "allow_mnnvl": True,
            "explicitly_destroy": True,  # 双模式下显式管理资源，避免析构冲突
        }

        if self._use_accl_ep:
            os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
            init_kwargs["allow_nvlink_for_low_latency_mode"] = True
            if allow_mnnvl():
                init_kwargs["allow_mnnvl"] = True
            else:
                init_kwargs["allow_mnnvl"] = False

        return DeepEPBuffer(**init_kwargs)  # type: ignore

    def _init_low_latency_m2n_buffer(self, group: ProcessGroup) -> DeepEPBuffer:
        """Initialize buffer for low-latency M2N mode."""
        config = self._config
        num_m = config.attention_dp_size * config.attention_tp_size
        num_n = config.ffn_dp_size * config.ffn_tp_size

        if not hasattr(DeepEPBuffer, "get_low_latency_rdma_size_hint_m2n"):
            raise RuntimeError(
                "current deep_ep provider does not support low-latency m2n"
            )

        num_rdma_bytes = DeepEPBuffer.get_low_latency_rdma_size_hint_m2n(
            config.ll_num_max_token_per_rank,
            config.hidden_size,
            num_m + num_n,
            config.expert_num,
            num_m,
        )

        if config.local_rank == 0:
            print(
                f"Allocating buffer size: {num_rdma_bytes / 1e6} MB, "
                f"ll_num_max_token_per_rank: {config.ll_num_max_token_per_rank}, "
                f"hidden_size: {config.hidden_size}, "
                f"expert_num: {config.expert_num}, "
                f"num_m: {num_m}, "
                f"num_n: {num_n}",
                flush=True,
            )

        num_qps_per_rank = config.expert_num / num_n

        init_kwargs = {
            "group": group,
            "num_nvl_bytes": 0,
            "num_rdma_bytes": num_rdma_bytes,
            "low_latency_mode": True,
            "num_qps_per_rank": num_qps_per_rank,
        }

        if self._use_accl_ep:
            init_kwargs["allow_nvlink_for_low_latency_mode"] = True
            init_kwargs["allow_mnnvl"] = False

        return DeepEPBuffer(**init_kwargs)  # type: ignore

    def _destroy_buffer(self) -> None:
        """Destroy the DeepEP buffer(s) and free resources."""
        if self._normal_buffer is not None:
            del self._normal_buffer
            self._normal_buffer = None
        if self._lowlatency_buffer is not None:
            del self._lowlatency_buffer
            self._lowlatency_buffer = None
        gc.collect()
