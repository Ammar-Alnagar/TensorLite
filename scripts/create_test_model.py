#!/usr/bin/env python3
"""
Creates a minimal valid FP8 Hugging Face model folder (config.json + model.safetensors)
for educational testing of TensorLite without needing multi-gigabyte downloads.
"""

import os
import sys
import json
import struct

def float_to_fp8_e4m3(val: float) -> int:
    # Minimal helper to approximate FP8 E4M3 for test weights
    import numpy as np
    # If numpy has float8_e4m3fn use it, else simple bit pack
    try:
        f8 = np.array([val], dtype=np.float32).astype(np.float8_e4m3fn)
        return int(f8.view(np.uint8)[0])
    except Exception:
        # Fallback: simple byte encoding for demo weights
        val = max(-448.0, min(448.0, val))
        if val == 0: return 0
        sign = 0x80 if val < 0 else 0x00
        val = abs(val)
        exp = int(np.floor(np.log2(val)))
        target_exp = exp + 7
        target_exp = max(0, min(15, target_exp))
        mant = int(((val / (2.0 ** exp)) - 1.0) * 8.0) if exp != -7 else 0
        mant = max(0, min(7, mant))
        return sign | (target_exp << 3) | mant

def create_synthetic_model(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Write config.json (Qwen-style architecture, small test dimensions)
    config = {
        "model_type": "qwen2",
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": True,
        "quantization_config": {
            "quant_method": "fp8"
        }
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 2. Build tensors
    # Embedding: [vocab_size, hidden_size] -> [256, 64] FP32
    # Layers 0..1:
    #   input_layernorm.weight: [64] FP32
    #   q_proj.weight: [64, 64] FP8
    #   k_proj.weight: [32, 64] FP8
    #   v_proj.weight: [32, 64] FP8
    #   o_proj.weight: [64, 64] FP8
    #   post_attention_layernorm.weight: [64] FP32
    #   gate_proj.weight: [128, 64] FP8
    #   up_proj.weight: [128, 64] FP8
    #   down_proj.weight: [64, 128] FP8
    # Final norm: [64] FP32

    tensors_data = {}

    def add_fp32_tensor(name, shape, val=1.0):
        count = 1
        for d in shape: count *= d
        raw = struct.pack(f"<{count}f", *([val] * count))
        tensors_data[name] = {"dtype": "F32", "shape": shape, "bytes": raw}

    def add_fp8_tensor(name, shape, scale=0.01):
        count = 1
        for d in shape: count *= d
        # initialize alternating small values
        byte_vals = bytearray([0x38 if (i % 2 == 0) else 0x30 for i in range(count)])
        tensors_data[name] = {"dtype": "F8_E4M3", "shape": shape, "bytes": bytes(byte_vals)}
        # Add scale tensor
        scale_bytes = struct.pack("<f", scale)
        tensors_data[name + "_scale"] = {"dtype": "F32", "shape": [1], "bytes": scale_bytes}

    add_fp32_tensor("model.embed_tokens.weight", [256, 64], 0.1)

    for l in range(2):
        pfx = f"model.layers.{l}"
        add_fp32_tensor(f"{pfx}.input_layernorm.weight", [64], 1.0)
        add_fp8_tensor(f"{pfx}.self_attn.q_proj.weight", [64, 64])
        add_fp8_tensor(f"{pfx}.self_attn.k_proj.weight", [32, 64])
        add_fp8_tensor(f"{pfx}.self_attn.v_proj.weight", [32, 64])
        add_fp8_tensor(f"{pfx}.self_attn.o_proj.weight", [64, 64])
        add_fp32_tensor(f"{pfx}.post_attention_layernorm.weight", [64], 1.0)
        add_fp8_tensor(f"{pfx}.mlp.gate_proj.weight", [128, 64])
        add_fp8_tensor(f"{pfx}.mlp.up_proj.weight", [128, 64])
        add_fp8_tensor(f"{pfx}.mlp.down_proj.weight", [64, 128])

    add_fp32_tensor("model.norm.weight", [64], 1.0)
    add_fp8_tensor("lm_head.weight", [256, 64])

    # 3. Assemble safetensors binary format
    # Header format: JSON mapping tensor name to {dtype, shape, data_offsets}
    # Followed by 8-byte uint64 header size, then JSON UTF-8 bytes, then aligned raw tensor payloads.

    header = {}
    current_offset = 0
    payload = bytearray()

    for name, item in tensors_data.items():
        data_len = len(item["bytes"])
        header[name] = {
            "dtype": item["dtype"],
            "shape": item["shape"],
            "data_offsets": [current_offset, current_offset + data_len]
        }
        payload.extend(item["bytes"])
        current_offset += data_len

    header_json_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_len = len(header_json_bytes)

    safetensors_file = os.path.join(output_dir, "model.safetensors")
    with open(safetensors_file, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json_bytes)
        f.write(payload)

    print(f"Created synthetic FP8 model at: {output_dir}")
    print(f"  config.json: {os.path.getsize(os.path.join(output_dir, 'config.json'))} bytes")
    print(f"  model.safetensors: {os.path.getsize(safetensors_file)} bytes")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "models/test_qwen_fp8"
    create_synthetic_model(out)
