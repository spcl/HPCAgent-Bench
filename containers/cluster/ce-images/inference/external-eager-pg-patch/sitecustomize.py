import faulthandler
import inspect
import os
import signal
import sys

import torch
import torch.distributed as dist

_original_init_process_group = dist.init_process_group
_original_new_group = dist.new_group
_new_group_parameters = inspect.signature(_original_new_group).parameters


def local_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", str(torch.cuda.current_device())))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def backend_from_call(args, kwargs, positional_index):
    backend = kwargs.get("backend")
    if backend is None and len(args) > positional_index:
        backend = args[positional_index]
    return backend


# Eager init (device_id set) serializes unbatched P2P against the group's other traffic. 0 drops
# device_id everywhere, returning to lazy init while KEEPING the collective/P2P split below.
# Neither setting decides the `received 1024 instead of 256` bootstrap collision that killed the
# four-node probes: 603524/603714/603718 ran with 1, 604479 with 0, and all four died the same
# way. That one is the async-scheduling broadcast on pp.device_group -- see run_cluster.sh.
EAGER_DEVICE_ID = os.environ.get("VLLM_EAGER_PG_DEVICE_ID", "1") == "1"


def eager_init_process_group(*args, **kwargs):
    backend = backend_from_call(args, kwargs, 0)

    if (
        EAGER_DEVICE_ID
        and "nccl" in str(backend).lower()
        and kwargs.get("device_id") is None
        and torch.cuda.is_available()
    ):
        device = local_device()
        kwargs["device_id"] = device

        print(
            f"[external-eager-pg] init_process_group backend={backend} device_id={device}",
            file=sys.stderr,
            flush=True,
        )

    return _original_init_process_group(*args, **kwargs)


def eager_new_group(*args, **kwargs):
    # new_group signature:
    # ranks, timeout, backend, ...
    backend = backend_from_call(args, kwargs, 2)

    if (
        EAGER_DEVICE_ID
        and "device_id" in _new_group_parameters
        and "nccl" in str(backend).lower()
        and kwargs.get("device_id") is None
        and torch.cuda.is_available()
    ):
        device = local_device()
        kwargs["device_id"] = device

        print(
            "[external-eager-pg] new_group "
            f"backend={backend} device_id={device} "
            f"ranks={kwargs.get('ranks', args[0] if args else None)}",
            file=sys.stderr,
            flush=True,
        )

    return _original_new_group(*args, **kwargs)


# Move the pipeline group's collectives to their own communicator.
#
# vLLM puts inter-stage activation transfer (isend/irecv_tensor_dict, v1/worker/gpu_worker.py)
# and ordinary collectives on ONE device_group, which eager init above makes fatal -- torch says
# so itself: "unbatched P2P ops are treated as independent collective ops, and are thus
# serialized with all other ops on this ProcessGroup". Grep a hanging log for that sentence.
# 600262/600263 both died there: a 4-element broadcast on the pp group (PG 5, ranks [0,4,8,12]
# and siblings) stuck at SeqNum=1 for the full 600 s watchdog; neither decoded a token.
#
# Upstream ships this remedy for exactly one broadcast (make_sibling_device_group in
# v1/worker/gpu/pp_utils.py); this extends it to every collective, leaving device_group carrying
# P2P alone. Membership is identical, so global rank ids still address the same processes.
COLLECTIVE_SIBLING_GROUPS = ("pp",)


def patch_pp_collectives() -> None:
    """Give the pp GroupCoordinator a second communicator and route its collectives onto it.

    Installed from :func:`eager_init_process_group` rather than at import: sitecustomize runs
    before vllm exists, but every GroupCoordinator is built AFTER `init_process_group`, so by then
    the module is importable and no coordinator has been missed.
    """
    from vllm.distributed.parallel_state import GroupCoordinator

    original_group_init = GroupCoordinator.__init__

    def group_init(self, *args, **kwargs):
        original_group_init(self, *args, **kwargs)
        # Sentinel-valued on every coordinator, so the wrappers below ask what it IS rather than
        # whether it exists. `unique_name` is "<group_name>:<n>" (parallel_state._get_unique_name).
        self.collective_group = None
        if self.unique_name.rsplit(":", 1)[0] not in COLLECTIVE_SIBLING_GROUPS or self.world_size <= 1:
            return
        # Collective over the WORLD: it mints one group per rank set, so every rank must reach it
        # in the same order. Coordinator construction is already in lockstep, which is why this
        # sits here and not at the first collective, where only the group's own ranks would call.
        self.collective_group = self.make_sibling_device_group(group_desc="external_pp_collectives")
        print(
            f"[external-eager-pg] {self.unique_name}: collectives split onto a sibling communicator",
            file=sys.stderr,
            flush=True,
        )

    def on_collective_group(method):
        """Run ``method`` with ``device_group`` pointing at the sibling.

        Swapping the attribute rather than threading a group argument through: the collectives all
        read ``self.device_group`` at entry, and `broadcast_tensor_dict` even accepts a ``group``
        parameter and then overwrites it, so the attribute is the only honest seam. Single-threaded
        per rank on this path -- the PP handler uses a side CUDA stream, not a side thread.
        """

        def wrapper(self, *args, **kwargs):
            sibling = self.collective_group
            if sibling is None:
                return method(self, *args, **kwargs)
            main_group = self.device_group
            self.device_group = sibling
            try:
                return method(self, *args, **kwargs)
            finally:
                self.device_group = main_group

        return wrapper

    GroupCoordinator.__init__ = group_init
    GroupCoordinator.broadcast = on_collective_group(GroupCoordinator.broadcast)
    GroupCoordinator.broadcast_object_list = on_collective_group(GroupCoordinator.broadcast_object_list)
    GroupCoordinator.broadcast_tensor_dict = on_collective_group(GroupCoordinator.broadcast_tensor_dict)


SPLIT_PP_COLLECTIVES = os.environ.get("VLLM_PP_COLLECTIVE_SPLIT", "1") == "1"

# Async scheduling's sampled-token broadcast, moved off the P2P communicator.
#
# It is the ONLY collective vLLM's V1 runner puts on pp.device_group, and that group otherwise
# carries just the inter-stage P2P, which torch serves from per-pair 2-rank communicators. Under
# lazy init the first decode therefore bootstraps a 4-rank and a 2-rank communicator at once and
# rccl bootstrap.cc reports "Message truncated : received 1024 bytes instead of 512" (nranks x
# 256). Upstream's V2 runner already avoids this with a sibling group of its own
# (v1/worker/gpu/pp_utils.PPHandler); this gives the V1 path the same treatment.
#
# The sibling is the one patch_pp_collectives already builds inside GroupCoordinator.__init__, so
# nothing is created after startup -- which is the whole point, a communicator minted during
# serving is what collides. Same membership, so the global src ranks still address the same
# processes.
#
# OFF by default: --no-async-scheduling removes the broadcast entirely and is the proven baseline.
# Turn this on to run WITH async scheduling, which spares PP a scheduler round trip per decode
# step (v1/core/sched/scheduler.py sends sampled tokens back when async is off).
PP_TOKEN_BROADCAST_SIBLING = os.environ.get("VLLM_PP_TOKEN_BROADCAST_SIBLING", "0") == "1"


def patch_pp_token_broadcast() -> None:
    """Run the two async-scheduling token-broadcast methods against the pp sibling communicator."""
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    def on_sibling(method):
        # Both methods read `pp.device_group` inline, so the attribute is the only seam -- the
        # same one on_collective_group uses. Single-threaded per rank on this path.
        def wrapper(self, *args, **kwargs):
            pp = get_pp_group()
            # Sentinel-valued on every coordinator by group_init, and None on a group that got no
            # sibling -- a pp world_size of 1, where there is no broadcast to move anyway.
            sibling = pp.collective_group
            if sibling is None:
                return method(self, *args, **kwargs)
            main_group = pp.device_group
            pp.device_group = sibling
            try:
                return method(self, *args, **kwargs)
            finally:
                pp.device_group = main_group

        return wrapper

    GPUModelRunner._pp_broadcast_prev_sampled_token_ids = on_sibling(
        GPUModelRunner._pp_broadcast_prev_sampled_token_ids
    )
    GPUModelRunner._pp_receive_prev_sampled_token_ids_to_input_batch = on_sibling(
        GPUModelRunner._pp_receive_prev_sampled_token_ids_to_input_batch
    )
    print(
        "[external-eager-pg] pp sampled-token broadcast routed onto the sibling communicator",
        file=sys.stderr,
        flush=True,
    )


# The MLA chunked-prefill context path transposes its log-sum-exp on ROCm.
#
# `mask_empty_context` (defined in v1/attention/ops/triton_merge_attn_states.py) unpacks lse as
# [num_heads, num_tokens] and sizes its mask from the second dimension. That holds for
# vllm-flash-attn; ROCm takes the upstream-flash_attn branch of
# `_flash_attn_varlen_diff_headdims`, which returns [num_tokens, num_heads], so the mask comes out
# sized num_heads and the fill raises "expanded size (3591) must match the existing size (16)".
# 603980 died there ~58 min in, taking all 121 agents with it.
#
# Reached only when some prefill in a chunk has run out of context and others have not, so it
# needs concurrent prefills of UNEQUAL length across chunks -- and `chunked_prefill_workspace_size`
# is capped at 64k in code (no CLI knob) then split `// num_prefills_with_context`, so a busy
# server chunks small and hits it constantly.
# No config escape either: ROCm's prefill priority is [ROCM_AITER_FA, FLASH_ATTN] and aiter's
# master switch breaks MLA prefill on gfx942.
#
# Transposes ONLY the view this helper gets; the caller's own reference feeds merge_attn_states,
# which already handles the ROCm layout. A transposed view is free -- the helper passes
# stride(0)/stride(1) to the triton kernel. The swap fires only on the unambiguous transposed
# shape, so any other layout still raises exactly as today.
FIX_MLA_LSE_LAYOUT = os.environ.get("VLLM_FIX_MLA_LSE_LAYOUT", "1") == "1"


def patch_mla_empty_context_mask() -> None:
    """Point mla_attention's `mask_empty_context` name at a layout-tolerant wrapper.

    mla_attention does `from ... import mask_empty_context`, so the name has to be rebound in
    THAT module -- patching the defining module would leave the existing binding untouched.

    MLA is kimi's attention and `mask_empty_context` is a 0.27.1 symbol, so neither the module nor
    the name is guaranteed: on vLLM 0.23 serving gpt-oss this raised AttributeError from inside
    init_process_group and took the engine down before the API came up (620914). A patch whose
    target is absent has nothing to fix -- it says so and leaves the eager-PG half to do its job.
    """
    try:
        from vllm.model_executor.layers.attention import mla_attention
    except ImportError:  # pre-0.27.1 layout, or a build with no MLA at all
        print(
            "[external-eager-pg] mla mask_empty_context: no mla_attention module, skipped", file=sys.stderr, flush=True
        )
        return

    original = vars(mla_attention).get("mask_empty_context")
    if original is None:
        print("[external-eager-pg] mla mask_empty_context: symbol absent, skipped", file=sys.stderr, flush=True)
        return

    def mask_empty_context(lse, output, query_start_loc, context_start_loc):
        # [num_tokens, num_heads] against an output of [num_tokens, num_heads, head_dim]. The
        # square case is genuinely ambiguous, so leave it to the original rather than guess.
        if (
            lse.ndim == 2
            and output.ndim == 3
            and lse.shape[0] == output.shape[0]
            and lse.shape[1] == output.shape[1]
            and output.shape[0] != output.shape[1]
        ):
            lse = lse.transpose(0, 1)
        return original(lse, output, query_start_loc, context_start_loc)

    mla_attention.mask_empty_context = mask_empty_context
    print("[external-eager-pg] mla mask_empty_context: ROCm lse layout tolerated", file=sys.stderr, flush=True)


INSTALLED: set[str] = set()

# vLLM sends PP activations over the device group (RCCL) but the tensor-dict METADATA over
# cpu_group, which it hardcodes to gloo in both construction paths -- there is no backend knob, so
# MPI is not reachable without patching vLLM either. This moves send_object/recv_object onto the
# device group so a PP step touches no CPU transport at all.
#
# OFF by default: it addresses TCP latency, not the four-node deaths. Those were a communicator
# bootstrap collision (--no-async-scheduling in run_cluster.sh); the 1800 s gloo "pair closure"
# reported afterwards was a surviving rank waiting on a peer that had already died.
#
# The size handshake exists because the receiver must size its buffer before receiving; that stays,
# it just travels as a device tensor. `.item()` on the received size forces a sync, which is the
# real cost of this patch -- one per PP handoff, against a TCP round trip saved.
METADATA_ON_DEVICE = os.environ.get("VLLM_PP_METADATA_ON_DEVICE", "0") == "1"


def patch_object_transfer_onto_device() -> None:
    """Route GroupCoordinator.send_object/recv_object through device_group instead of cpu_group."""
    import pickle

    import torch
    from vllm.distributed import parallel_state

    coordinator = parallel_state.GroupCoordinator

    def send_object(self, obj, dst: int) -> None:
        device = getattr(self, "device", None)
        if device is None or self.device_group is None:
            return original_send(self, obj, dst)
        payload = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8).to(device)
        size = torch.tensor([payload.numel()], dtype=torch.long, device=device)
        torch.distributed.send(size, dst=self.ranks[dst], group=self.device_group)
        torch.distributed.send(payload, dst=self.ranks[dst], group=self.device_group)

    def recv_object(self, src: int):
        device = getattr(self, "device", None)
        if device is None or self.device_group is None:
            return original_recv(self, src)
        size = torch.empty(1, dtype=torch.long, device=device)
        rank_size = torch.distributed.recv(size, src=self.ranks[src], group=self.device_group)
        payload = torch.empty(int(size.item()), dtype=torch.uint8, device=device)
        rank_payload = torch.distributed.recv(payload, src=self.ranks[src], group=self.device_group)
        assert rank_payload == rank_size, "size and payload arrived from different senders"
        return pickle.loads(payload.cpu().numpy().tobytes())

    original_send = coordinator.send_object
    original_recv = coordinator.recv_object
    coordinator.send_object = send_object
    coordinator.recv_object = recv_object
    print("[external-eager-pg] send_object/recv_object routed onto the device group", file=sys.stderr, flush=True)


def eager_init_and_split(*args, **kwargs):
    """`init_process_group`, then install the pp collective split on top of the fresh world.

    Once per patch, however often the world is rebuilt: both wrap a name around whatever is
    already bound to it, so a second init would nest GroupCoordinator.__init__ inside its own
    wrapper -- a duplicate sibling communicator per pp group per rank -- and stack another
    mask_empty_context wrapper on every call.
    """
    result = eager_init_process_group(*args, **kwargs)
    if SPLIT_PP_COLLECTIVES and "pp_collectives" not in INSTALLED:
        patch_pp_collectives()
        INSTALLED.add("pp_collectives")
    if FIX_MLA_LSE_LAYOUT and "mla_lse_layout" not in INSTALLED:
        patch_mla_empty_context_mask()
        INSTALLED.add("mla_lse_layout")
    if PP_TOKEN_BROADCAST_SIBLING and "pp_token_broadcast" not in INSTALLED:
        # The sibling it routes onto is built by patch_pp_collectives. Without that there is no
        # second communicator and this would silently leave the broadcast where it was.
        if not SPLIT_PP_COLLECTIVES:
            raise RuntimeError("VLLM_PP_TOKEN_BROADCAST_SIBLING=1 needs VLLM_PP_COLLECTIVE_SPLIT=1")
        patch_pp_token_broadcast()
        INSTALLED.add("pp_token_broadcast")
    if METADATA_ON_DEVICE and "metadata_on_device" not in INSTALLED:
        patch_object_transfer_onto_device()
        INSTALLED.add("metadata_on_device")
    return result


dist.init_process_group = eager_init_and_split
dist.new_group = eager_new_group

# Opt-in stack dumper: py-spy is not in this image, so asking the process itself is the only way
# to see where a worker blocks during the ramp (599301: prefill completes, decode never starts).
# SIGUSR1 dumps every thread's stack to that rank's vllm log.
if os.environ.get("DUMP_STACKS_ON_SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
    print("[external-eager-pg] SIGUSR1 stack dumper armed", file=sys.stderr, flush=True)
