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


# Eager init (device_id set) serializes unbatched P2P against the group's other traffic; 0
# drops device_id and falls back to lazy init while keeping the collective/P2P split below.
# Independent of the async-scheduling broadcast fix -- see PP_TOKEN_BROADCAST_SIBLING.
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


# Move the pipeline group's collectives onto their own communicator: vLLM puts inter-stage P2P
# and the pp group's ordinary collectives on ONE device_group, which eager init above makes
# fatal (torch serializes unbatched P2P against every other op on that ProcessGroup). Extends
# upstream's single-broadcast remedy (make_sibling_device_group) to every collective; membership
# is unchanged so global rank ids still address the same processes.
COLLECTIVE_SIBLING_GROUPS = ("pp",)


def patch_pp_collectives() -> None:
    """Give the pp GroupCoordinator a second communicator and route its collectives onto it.

    Installed from eager_init_process_group rather than at import: vllm is not importable yet
    when sitecustomize runs, but every GroupCoordinator is built after init_process_group.
    """
    from vllm.distributed.parallel_state import GroupCoordinator

    original_group_init = GroupCoordinator.__init__

    def group_init(self, *args, **kwargs):
        original_group_init(self, *args, **kwargs)
        # Set on every coordinator so callers ask what it IS, not whether it exists.
        # unique_name is "<group_name>:<n>" (parallel_state._get_unique_name).
        self.collective_group = None
        if self.unique_name.rsplit(":", 1)[0] not in COLLECTIVE_SIBLING_GROUPS or self.world_size <= 1:
            return
        # Mints one group per rank set, so every rank must call in the same order; built here,
        # while construction is still in lockstep, rather than at the first collective.
        self.collective_group = self.make_sibling_device_group(group_desc="external_pp_collectives")
        print(
            f"[external-eager-pg] {self.unique_name}: collectives split onto a sibling communicator",
            file=sys.stderr,
            flush=True,
        )

    def on_collective_group(method):
        """Run ``method`` with ``device_group`` pointing at the sibling.

        Swaps the attribute rather than threading a group argument through: all the collectives
        read ``self.device_group`` at entry, and `broadcast_tensor_dict` even overwrites its own
        ``group`` parameter. Single-threaded per rank on this path.
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

# Async scheduling's sampled-token broadcast, moved off the P2P communicator: it is the only
# collective vLLM's V1 runner puts on pp.device_group, and that group otherwise carries just the
# inter-stage P2P (per-pair 2-rank communicators). Under lazy init the first decode bootstraps
# both communicators at once, which collides. Reuses the sibling patch_pp_collectives already
# built, so nothing new is minted mid-serving. OFF by default: --no-async-scheduling removes the
# broadcast entirely and is the proven baseline; turn this on to keep async scheduling instead.
PP_TOKEN_BROADCAST_SIBLING = os.environ.get("VLLM_PP_TOKEN_BROADCAST_SIBLING", "0") == "1"


def patch_pp_token_broadcast() -> None:
    """Run the two async-scheduling token-broadcast methods against the pp sibling communicator."""
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    def on_sibling(method):
        # Both methods read `pp.device_group` inline, so swapping the attribute is the same seam
        # on_collective_group uses. Single-threaded per rank on this path.
        def wrapper(self, *args, **kwargs):
            pp = get_pp_group()
            # None when group_init gave this group no sibling (pp world_size 1: no broadcast to move).
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
# `mask_empty_context` expects lse as [num_heads, num_tokens] (vllm-flash-attn layout), but ROCm's
# `_flash_attn_varlen_diff_headdims` returns [num_tokens, num_heads], so the mask comes out the
# wrong size and the fill raises a shape-mismatch error. Reached only when some prefill in a chunk
# has run out of context and others have not (unequal prefill lengths in the same chunk); ROCm has
# no config escape from that path.
#
# Transposes only the view this helper gets -- the caller's own reference already handles the
# ROCm layout -- and fires only on the unambiguous transposed shape, so any other layout still
# raises exactly as today.
FIX_MLA_LSE_LAYOUT = os.environ.get("VLLM_FIX_MLA_LSE_LAYOUT", "1") == "1"


def patch_mla_empty_context_mask() -> None:
    """Point mla_attention's `mask_empty_context` name at a layout-tolerant wrapper.

    mla_attention does `from ... import mask_empty_context`, so the name must be rebound in THAT
    module; patching the defining module would leave the existing binding untouched.

    `mask_empty_context` is a 0.27.1 symbol and not guaranteed to exist on other vLLM versions or
    builds without MLA, so a missing symbol is skipped rather than raised.
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
# device group so a PP step touches no CPU transport at all. OFF by default: it addresses TCP
# latency, not a correctness issue.
#
# The size handshake stays, since the receiver must size its buffer before receiving; it just
# travels as a device tensor. `.item()` on the received size forces a sync, which is the real
# cost of this patch -- one per PP handoff, against a TCP round trip saved.
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

# Opt-in stack dumper: py-spy is not in this image, so SIGUSR1 dumps every thread's stack to
# that rank's vllm log instead, to see where a worker is blocked.
if os.environ.get("DUMP_STACKS_ON_SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
    print("[external-eager-pg] SIGUSR1 stack dumper armed", file=sys.stderr, flush=True)
