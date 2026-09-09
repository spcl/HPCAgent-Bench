"""Give torch.Tensor a class-level format_ue8m0 default, so SGLang's DeepSeek loader stops
crashing on ROCm.

sglang 0.5.19.dev20260908, deepseek_weight_loader.py:641, reached by GLM-5.3 and any other
GlmMoeDsa/DeepSeek fp8 checkpoint:

    if _is_fp8_fnuz:                                          # gfx942 uses e4m3fnuz natively
        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(...)   # returns a NEW tensor
    ...
    if weight_scale.format_ue8m0 and weight_scale.dtype == torch.uint8:   # AttributeError

format_ue8m0 is a Python attribute STAMPED onto a scale tensor by fp8.py. The fnuz normalization
above builds a fresh tensor, which carries no such attribute -- and that branch is AMD-only, which
is why the bug ships. A class attribute is the minimal fix with the right semantics: instance
assignments (fp8.py sets True/False all over) still shadow it, and an unstamped tensor now reads
False, which is exactly what the same tree already does defensively at fp8_utils.py:1541
(`getattr(weight_scale, "format_ue8m0", False)`).

False is also the CORRECT value here, not merely a safe one: the branch it guards leads to
should_deepgemm_weight_requant_ue8m0, and deepgemm is NVIDIA-only, so the whole block is dead
code on this hardware.

This is a stopgap for testing whether the 20260909 nightly is needed. Prefer a fixed upstream pin.
"""

try:
    import torch

    if not hasattr(torch.Tensor, "format_ue8m0"):
        torch.Tensor.format_ue8m0 = False
        print("[ue8m0-patch] torch.Tensor.format_ue8m0 default installed", flush=True)
    else:
        print("[ue8m0-patch] already present, nothing to do", flush=True)
except Exception as exc:  # noqa: BLE001 -- a patch that cannot apply must not break the server
    print(f"[ue8m0-patch] FAILED to apply: {type(exc).__name__}: {exc}", flush=True)
