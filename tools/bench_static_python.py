#!/usr/bin/env python3
"""Pinned Python SAM2 static benchmark over the shared safetensors fixture."""
from pathlib import Path
import argparse, json, os, statistics, sys, time
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/segment-anything-2"))
from sam2.build_sam import build_sam2

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("fixture", type=Path)
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--output", type=Path)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--profile-stages", action="store_true")
parser.add_argument("--profile-encoder", action="store_true", help="emit synchronized Hiera block and FPN timings")
parser.add_argument("--profile-decoder", action="store_true", help="emit synchronized mask-decoder phase timings")
parser.add_argument("--decoder-only", action="store_true", help="time repeated decoder calls on one precomputed image embedding")
parser.add_argument(
    "--profile-ops",
    action="store_true",
    help="emit top PyTorch CPU operators from one additional inference; does not affect timing samples",
)
parser.add_argument(
    "--disable-mkldnn",
    action="store_true",
    help="attribution-only: disable PyTorch oneDNN/MKLDNN before model construction",
)
parser.add_argument("--threads", type=int, help="PyTorch CPU worker cap")
args = parser.parse_args()
if args.decoder_only and args.profile_encoder:
    parser.error("--decoder-only cannot be combined with --profile-encoder")

if args.threads:
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
if args.disable_mkldnn:
    torch.backends.mkldnn.enabled = False

fixture = load_file(str(args.fixture))
if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("--device cuda requested but PyTorch cannot access a CUDA device")
model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(args.checkpoint), device=args.device, apply_postprocessing=False)
image, points, labels = (fixture["image"].to(args.device), fixture["points"].to(args.device), fixture["labels"].to(args.device))

def decode(image_output):
    sparse, dense = model.sam_prompt_encoder((points, labels), None, None)
    masks, _, _, _ = model.sam_mask_decoder(image_output["vision_features"], model.sam_prompt_encoder.get_dense_pe(), sparse, dense, False, False, image_output["backbone_fpn"][:2])
    return masks

def infer():
    return decode(model.forward_image(image))

def synchronize():
    if args.device == "cuda":
        torch.cuda.synchronize()

def profile_image_encoder():
    """Profile the same Hiera-block/FPN boundaries exposed by Rust."""
    trunk = model.image_encoder.trunk
    block_times = [0.0] * len(trunk.blocks)
    block_starts = [0.0] * len(trunk.blocks)
    attention_times = [0.0] * len(trunk.blocks)
    attention_starts = [0.0] * len(trunk.blocks)
    attention_qkv_times = [0.0] * len(trunk.blocks)
    attention_projection_times = [0.0] * len(trunk.blocks)
    mlp_times = [0.0] * len(trunk.blocks)
    mlp_starts = [0.0] * len(trunk.blocks)
    mlp_expansion_times = [0.0] * len(trunk.blocks)
    mlp_projection_times = [0.0] * len(trunk.blocks)
    handles = []
    for index, block in enumerate(trunk.blocks):
        def pre_hook(_module, _inputs, index=index):
            synchronize()
            block_starts[index] = time.perf_counter()
        def post_hook(_module, _inputs, _output, index=index):
            synchronize()
            block_times[index] = (time.perf_counter() - block_starts[index]) * 1000
        handles.append(block.register_forward_pre_hook(pre_hook))
        handles.append(block.register_forward_hook(post_hook))
        def attn_pre(_module, _inputs, index=index):
            synchronize()
            attention_starts[index] = time.perf_counter()
        def attn_post(_module, _inputs, _output, index=index):
            synchronize()
            attention_times[index] = (time.perf_counter() - attention_starts[index]) * 1000
        handles.append(block.attn.register_forward_pre_hook(attn_pre))
        handles.append(block.attn.register_forward_hook(attn_post))
        for target, module in ((attention_qkv_times, block.attn.qkv), (attention_projection_times, block.attn.proj)):
            module_start = [0.0]
            def linear_pre(_module, _inputs, module_start=module_start):
                synchronize()
                module_start[0] = time.perf_counter()
            def linear_post(_module, _inputs, _output, target=target, index=index, module_start=module_start):
                synchronize()
                target[index] = (time.perf_counter() - module_start[0]) * 1000
            handles.append(module.register_forward_pre_hook(linear_pre))
            handles.append(module.register_forward_hook(linear_post))
        def mlp_pre(_module, _inputs, index=index):
            synchronize()
            mlp_starts[index] = time.perf_counter()
        def mlp_post(_module, _inputs, _output, index=index):
            synchronize()
            mlp_times[index] = (time.perf_counter() - mlp_starts[index]) * 1000
        handles.append(block.mlp.register_forward_pre_hook(mlp_pre))
        handles.append(block.mlp.register_forward_hook(mlp_post))
        # SAM2's Hiera MLP has exactly two Linear modules.  Hooking those
        # matches Rust's expansion/projection timing and leaves the exact GELU
        # as MLP-total minus the two module times.
        for layer_index, target in enumerate((mlp_expansion_times, mlp_projection_times)):
            layer_start = [0.0]
            def linear_pre(_module, _inputs, layer_start=layer_start):
                synchronize()
                layer_start[0] = time.perf_counter()
            def linear_post(_module, _inputs, _output, target=target, index=index, layer_start=layer_start):
                synchronize()
                target[index] = (time.perf_counter() - layer_start[0]) * 1000
            handles.append(block.mlp.layers[layer_index].register_forward_pre_hook(linear_pre))
            handles.append(block.mlp.layers[layer_index].register_forward_hook(linear_post))
    neck_start = [0.0]
    neck_ms = [0.0]
    def neck_pre(_module, _inputs):
        synchronize()
        neck_start[0] = time.perf_counter()
    def neck_post(_module, _inputs, _output):
        synchronize()
        neck_ms[0] = (time.perf_counter() - neck_start[0]) * 1000
    handles.append(model.image_encoder.neck.register_forward_pre_hook(neck_pre))
    handles.append(model.image_encoder.neck.register_forward_hook(neck_post))
    try:
        output = model.forward_image(image)
        synchronize()
    finally:
        for handle in handles:
            handle.remove()
    return output, block_times, attention_times, attention_qkv_times, attention_projection_times, mlp_times, mlp_expansion_times, mlp_projection_times, neck_ms[0]

def profile_mask_decoder(image_output):
    """Profile the same decoder boundaries reported by Rust.

    Hooks intentionally surround modules rather than individual tensor ops, so
    this remains a diagnostic tool and leaves the regular benchmark hot path
    unchanged.
    """
    decoder = model.sam_mask_decoder
    transformer = decoder.transformer
    phases = {
        "decoder_transformer_self_attention_ms": [0.0] * len(transformer.layers),
        "decoder_transformer_token_to_image_ms": [0.0] * len(transformer.layers),
        "decoder_transformer_mlp_ms": [0.0] * len(transformer.layers),
        "decoder_transformer_image_to_token_ms": [0.0] * len(transformer.layers),
        "decoder_transformer_final_attention_ms": 0.0,
        # Cross-attentions: token-to-image (2), image-to-token (2), final (1).
        "decoder_cross_qkv_ms": [0.0] * 5,
        "decoder_cross_output_projection_ms": [0.0] * 5,
        "decoder_upconv_0_ms": 0.0,
        "decoder_skip_1_ms": 0.0,
        "decoder_upnorm_gelu_ms": 0.0,
        "decoder_upconv_3_ms": 0.0,
        "decoder_skip_0_ms": 0.0,
        "decoder_hypernetworks_ms": 0.0,
        "decoder_heads_ms": 0.0,
    }
    handles = []

    def hook_module(module, key, index=None):
        start = [0.0]
        def pre(_module, _inputs):
            synchronize()
            start[0] = time.perf_counter()
        def post(_module, _inputs, _output):
            synchronize()
            elapsed = (time.perf_counter() - start[0]) * 1000
            if index is None:
                phases[key] += elapsed
            else:
                phases[key][index] = elapsed
        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))

    def hook_projection(module, key, index):
        start = [0.0]
        def pre(_module, _inputs):
            synchronize()
            start[0] = time.perf_counter()
        def post(_module, _inputs, _output):
            synchronize()
            phases[key][index] += (time.perf_counter() - start[0]) * 1000
        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))

    for index, layer in enumerate(transformer.layers):
        hook_module(layer.self_attn, "decoder_transformer_self_attention_ms", index)
        hook_module(layer.cross_attn_token_to_image, "decoder_transformer_token_to_image_ms", index)
        hook_module(layer.mlp, "decoder_transformer_mlp_ms", index)
        hook_module(layer.cross_attn_image_to_token, "decoder_transformer_image_to_token_ms", index)
    hook_module(transformer.final_attn_token_to_image, "decoder_transformer_final_attention_ms")
    cross_attentions = [
        *(layer.cross_attn_token_to_image for layer in transformer.layers),
        *(layer.cross_attn_image_to_token for layer in transformer.layers),
        transformer.final_attn_token_to_image,
    ]
    for index, attention in enumerate(cross_attentions):
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
            hook_projection(projection, "decoder_cross_qkv_ms", index)
        hook_projection(attention.out_proj, "decoder_cross_output_projection_ms", index)
    hook_module(decoder.output_upscaling[0], "decoder_upconv_0_ms")
    # These hooks should read zero for the standard cached-image decode:
    # SAM2's forward_image applies both skip projections before this call.
    hook_module(decoder.conv_s1, "decoder_skip_1_ms")
    # LayerNorm2d and GELU are separate modules in SAM2's sequential block.
    hook_module(decoder.output_upscaling[1], "decoder_upnorm_gelu_ms")
    hook_module(decoder.output_upscaling[2], "decoder_upnorm_gelu_ms")
    hook_module(decoder.output_upscaling[3], "decoder_upconv_3_ms")
    hook_module(decoder.conv_s0, "decoder_skip_0_ms")
    hook_module(decoder.output_upscaling[4], "decoder_upconv_3_ms")
    for hypernetwork in decoder.output_hypernetworks_mlps:
        hook_module(hypernetwork, "decoder_hypernetworks_ms")
    hook_module(decoder.iou_prediction_head, "decoder_heads_ms")
    hook_module(decoder.pred_obj_score_head, "decoder_heads_ms")
    try:
        start = time.perf_counter()
        sparse, dense = model.sam_prompt_encoder((points, labels), None, None)
        phases["decoder_token_setup_ms"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        masks, _, _, _ = decoder(image_output["vision_features"], model.sam_prompt_encoder.get_dense_pe(), sparse, dense, False, False, image_output["backbone_fpn"][:2])
        synchronize()
        phases["decoder_total_ms"] = (time.perf_counter() - start) * 1000
    finally:
        for handle in handles:
            handle.remove()
    cross_totals = [
        *phases["decoder_transformer_token_to_image_ms"],
        *phases["decoder_transformer_image_to_token_ms"],
        phases["decoder_transformer_final_attention_ms"],
    ]
    phases["decoder_cross_total_ms"] = cross_totals
    phases["decoder_cross_kernel_ms"] = [
        total - qkv - projection
        for total, qkv, projection in zip(
            cross_totals,
            phases["decoder_cross_qkv_ms"],
            phases["decoder_cross_output_projection_ms"],
        )
    ]
    return masks, phases

with torch.inference_mode():
    infer()
    cached_image_output = model.forward_image(image) if args.decoder_only else None
    if args.device == "cuda":
        torch.cuda.synchronize()
    cuda_graph = None
    graph_masks = None
    if (
        args.device == "cuda"
        and not args.decoder_only
        and not args.profile_stages
        and not args.profile_encoder
        and not args.profile_decoder
        and os.environ.get("SAM2_CUDA_GRAPH") == "1"
    ):
        # Match the Rust static-input graph path.  PyTorch requires capture
        # warmup on a side stream so lazy cuBLAS/cuDNN initialization is not
        # recorded in the graph.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                infer()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            graph_masks = infer()
        torch.cuda.synchronize()
    samples = []
    stage_profile = None
    encoder_profile = None
    decoder_profile = None
    for iteration in range(args.iterations):
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if (args.profile_stages or args.profile_encoder or args.profile_decoder) and iteration == 0:
            if args.profile_encoder:
                image_output, block_times, attention_times, attention_qkv_times, attention_projection_times, mlp_times, mlp_expansion_times, mlp_projection_times, fpn_ms = profile_image_encoder()
                encoder_profile = (block_times, attention_times, attention_qkv_times, attention_projection_times, mlp_times, mlp_expansion_times, mlp_projection_times, fpn_ms)
            elif args.decoder_only:
                image_output = cached_image_output
            else:
                image_output = model.forward_image(image)
            if args.device == "cuda":
                torch.cuda.synchronize()
            encoder_ms = (time.perf_counter() - start) * 1000
            decoder_start = time.perf_counter()
            if args.profile_decoder:
                masks, decoder_profile = profile_mask_decoder(image_output)
            else:
                masks = decode(image_output)
            if args.device == "cuda":
                torch.cuda.synchronize()
            stage_profile = (encoder_ms, (time.perf_counter() - decoder_start) * 1000)
        else:
            if cuda_graph is not None:
                cuda_graph.replay()
                masks = graph_masks
            else:
                masks = decode(cached_image_output) if args.decoder_only else infer()
            if args.device == "cuda":
                torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)

operator_profile = None
if args.profile_ops:
    # Profile a separate inference so the benchmark's measured samples remain
    # directly comparable with Rust. This instruments the installed PyTorch
    # build only; it does not patch the reference model or its operators.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        with torch.inference_mode():
            decode(cached_image_output) if args.decoder_only else infer()
    operator_profile = [
        {
            "operator": event.key,
            "self_cpu_ms": event.self_cpu_time_total / 1000.0,
            "cpu_total_ms": event.cpu_time_total / 1000.0,
            "calls": event.count,
        }
        for event in sorted(
            profiler.key_averages(), key=lambda event: event.self_cpu_time_total, reverse=True
        )[:20]
    ]

rss_hwm_kb = 0
for line in Path("/proc/self/status").read_text().splitlines():
    if line.startswith("VmHWM:"):
        rss_hwm_kb = int(line.split()[1])
if args.output:
    masks.detach().cpu().float().numpy().astype("<f4").tofile(args.output)
result = {"implementation": "python-torch", "device": args.device, "scope": "decoder" if args.decoder_only else "full", "iterations": args.iterations, "median_ms": statistics.median(samples), "rss_hwm_kb": rss_hwm_kb, "mask_checksum": float(masks.sum())}
if stage_profile:
    result["encoder_ms"], result["decoder_ms"] = stage_profile
if encoder_profile:
    result["hiera_block_ms"], result["hiera_attention_ms"], result["hiera_attention_qkv_ms"], result["hiera_attention_projection_ms"], result["hiera_mlp_ms"], result["hiera_mlp_expansion_ms"], result["hiera_mlp_projection_ms"], result["fpn_ms"] = encoder_profile
    result["hiera_attention_kernel_ms"] = [
        total - qkv - projection
        for total, qkv, projection in zip(result["hiera_attention_ms"], result["hiera_attention_qkv_ms"], result["hiera_attention_projection_ms"])
    ]
    result["hiera_mlp_activation_ms"] = [
        total - expansion - projection
        for total, expansion, projection in zip(result["hiera_mlp_ms"], result["hiera_mlp_expansion_ms"], result["hiera_mlp_projection_ms"])
    ]
if decoder_profile:
    result.update(decoder_profile)
if operator_profile is not None:
    result["top_cpu_operators"] = operator_profile
print(json.dumps(result))
