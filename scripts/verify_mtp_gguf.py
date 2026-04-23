#!/usr/bin/env python3
"""Verify that a converted Qwen3.5/3.6 GGUF preserved its MTP (nextn) head.

Usage: verify_mtp_gguf.py <path/to/qwen.gguf> [--safetensors-dir <hf dir>]

Checks (in order):
  [1] Metadata: arch is qwen35/qwen35moe, block_count was bumped, nextn_predict_layers >= 1.
  [2] Head tensors: blk.N.nextn.{eh_proj,enorm,hnorm,shared_head_norm} at expected shapes.
  [3] Body tensors: the 11 attn/MLP/norm tensors of the MTP decoder block.
  [4] Numerical round-trip (optional): compare a few MTP tensors byte-exactly to the
      source safetensors. Requires `safetensors` and `torch`.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf  # noqa: E402


# suffix on GGUF side  → (source name in safetensors, human description)
MTP_HEAD = {
    "nextn.eh_proj.weight":          "mtp.fc.weight",
    "nextn.enorm.weight":            "mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight":            "mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "mtp.norm.weight",
}

MTP_BODY = {
    "attn_norm.weight":           "mtp.layers.0.input_layernorm.weight",
    "post_attention_norm.weight": "mtp.layers.0.post_attention_layernorm.weight",
    "attn_q.weight":              "mtp.layers.0.self_attn.q_proj.weight",
    "attn_k.weight":              "mtp.layers.0.self_attn.k_proj.weight",
    "attn_v.weight":              "mtp.layers.0.self_attn.v_proj.weight",
    "attn_output.weight":         "mtp.layers.0.self_attn.o_proj.weight",
    "attn_q_norm.weight":         "mtp.layers.0.self_attn.q_norm.weight",
    "attn_k_norm.weight":         "mtp.layers.0.self_attn.k_norm.weight",
    "ffn_gate.weight":            "mtp.layers.0.mlp.gate_proj.weight",
    "ffn_up.weight":              "mtp.layers.0.mlp.up_proj.weight",
    "ffn_down.weight":            "mtp.layers.0.mlp.down_proj.weight",
}


def red(s):    return f"\033[31m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"


def kv_scalar(reader, key):
    f = reader.get_field(key)
    if f is None:
        return None
    if f.types and f.types[0] == gguf.GGUFValueType.STRING:
        return bytes(f.parts[-1]).decode("utf-8")
    return f.contents()


def gguf_tensor_to_torch(t):
    """Convert a gguf.ReaderTensor's raw bytes into a torch tensor matching the
    on-disk dtype. Only handles F32/F16/BF16/F64 (enough for our MTP tensors)."""
    import numpy as np
    import torch
    shape = tuple(int(x) for x in reversed(t.shape) if x != 0)  # gguf stores reversed; match torch
    # gguf.ReaderTensor.shape is already in ggml (reversed) order; for a 2-D weight [out,in] in torch
    # this is shape[::-1]. We'll compare by total element count and reshape to torch's view.
    tt = t.tensor_type
    raw = bytes(t.data)  # contiguous on-disk bytes
    if tt == gguf.GGMLQuantizationType.F32:
        arr = np.frombuffer(raw, dtype=np.float32)
        return torch.from_numpy(arr).reshape(shape[::-1])
    if tt == gguf.GGMLQuantizationType.F16:
        arr = np.frombuffer(raw, dtype=np.float16)
        return torch.from_numpy(arr.copy()).reshape(shape[::-1])
    if tt == gguf.GGMLQuantizationType.BF16:
        # numpy has no bf16; reinterpret u16 and bit-shift into f32
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        f32 = u16.view(np.float32)
        return torch.from_numpy(f32.copy()).reshape(shape[::-1])
    raise NotImplementedError(f"gguf dtype {tt.name} not handled in verifier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--safetensors-dir", help="HF source dir to cross-check MTP bytes")
    ap.add_argument("--sample", type=int, default=4,
                    help="how many MTP tensors to numerically verify against safetensors (default 4)")
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.gguf)

    # [1] metadata
    print("=== [1] metadata ===")
    arch = kv_scalar(reader, "general.architecture")
    print(f"arch                   : {arch}")
    ok_arch = arch in ("qwen35", "qwen35moe")
    if not ok_arch:
        print(red(f"  expected qwen35/qwen35moe, got {arch!r}"))

    block_count  = kv_scalar(reader, f"{arch}.block_count")
    n_mtp        = kv_scalar(reader, f"{arch}.nextn_predict_layers")
    n_embd       = kv_scalar(reader, f"{arch}.embedding_length")
    print(f"block_count            : {block_count}")
    print(f"nextn_predict_layers   : {n_mtp}")
    print(f"embedding_length       : {n_embd}")

    if n_mtp is None or n_mtp < 1:
        print(red("  FAIL: nextn_predict_layers missing or < 1"))
        sys.exit(1)

    n_main = block_count - n_mtp
    mtp_bid = n_main
    print(f"→ first MTP block index: blk.{mtp_bid}")

    tensors_by_name = {t.name: t for t in reader.tensors}

    # [2] head
    print("\n=== [2] nextn head ===")
    ok_head = True
    for suffix, src in MTP_HEAD.items():
        tname = f"blk.{mtp_bid}.{suffix}"
        t = tensors_by_name.get(tname)
        if t is None:
            print(red(f"  MISSING  {tname}  (source: {src})"))
            ok_head = False
            continue
        shape = tuple(int(x) for x in t.shape if x != 0)
        print(f"  {green('OK')}  {tname:45s} shape={shape} dtype={t.tensor_type.name}")

    # [3] body
    print("\n=== [3] MTP decoder body ===")
    ok_body = True
    for suffix, src in MTP_BODY.items():
        tname = f"blk.{mtp_bid}.{suffix}"
        t = tensors_by_name.get(tname)
        if t is None:
            print(red(f"  MISSING  {tname}  (source: {src})"))
            ok_body = False
            continue
        shape = tuple(int(x) for x in t.shape if x != 0)
        print(f"  {green('OK')}  {tname:45s} shape={shape} dtype={t.tensor_type.name}")

    # [4] numerical round-trip
    print("\n=== [4] numerical round-trip ===")
    ok_num = True
    if not args.safetensors_dir:
        print(yellow("  skipped (no --safetensors-dir)"))
    else:
        try:
            from safetensors import safe_open  # noqa
            import torch
        except ImportError as e:
            print(yellow(f"  skipped ({e})"))
        else:
            st_dir = Path(args.safetensors_dir)
            idx = json.loads((st_dir / "model.safetensors.index.json").read_text())["weight_map"]

            pairs = list({**MTP_HEAD, **MTP_BODY}.items())[: args.sample]
            for suffix, src in pairs:
                tname = f"blk.{mtp_bid}.{suffix}"
                gt = tensors_by_name.get(tname)
                if gt is None:
                    print(red(f"  MISSING in gguf: {tname}"))
                    ok_num = False
                    continue
                shard = idx.get(src)
                if shard is None:
                    print(red(f"  MISSING in safetensors: {src}"))
                    ok_num = False
                    continue
                with safe_open(st_dir / shard, framework="pt") as f:
                    t_st = f.get_tensor(src)
                t_gg = gguf_tensor_to_torch(gt)

                # Qwen3.5's norm weights get +1 applied at conversion time to match the
                # codebase's RMSNorm convention. Account for that on norm weights.
                expected_plus_one = suffix.endswith("norm.weight")
                t_st_comp = (t_st.float() + 1.0) if expected_plus_one else t_st.float()
                t_gg_comp = t_gg.float().reshape(t_st_comp.shape)

                diff = (t_st_comp - t_gg_comp).abs().max().item()
                tag = f"{suffix}{'  (+1 norm)' if expected_plus_one else ''}"
                if diff == 0.0:
                    print(f"  {green('EXACT')}  {tag:50s} (Δmax=0)")
                elif diff < 1e-3:
                    print(f"  {green('MATCH')}  {tag:50s} (Δmax={diff:.2e})")
                else:
                    print(red(f"  DIVERGE  {tag:50s} (Δmax={diff:.3g})"))
                    ok_num = False

    # summary
    print("\n=== summary ===")
    parts = [
        ("metadata", n_mtp is not None and n_mtp >= 1 and ok_arch),
        ("head",     ok_head),
        ("body",     ok_body),
        ("numeric",  ok_num),
    ]
    for name, ok in parts:
        print(f"  {green('PASS') if ok else red('FAIL')}  {name}")

    sys.exit(0 if all(ok for _, ok in parts) else 1)


if __name__ == "__main__":
    main()
