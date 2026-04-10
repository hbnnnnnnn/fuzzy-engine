"""
Generate images for ANY benchmark using Nexus-Gen V2.
VRAM: ~24 GB (fp8 quantization recommended on RTX 4090)

Usage:
    python generate_nexusgen.py --benchmark geneval \
        --prompt_file ../geneval/prompts/evaluation_metadata.jsonl \
        --outdir outputs/geneval/nexus_gen --n_samples 4 --fp8_quantization

    python generate_nexusgen.py --benchmark tifa \
        --prompt_file ../tifa/tifa_v1.0/tifa_v1.0_text_inputs.json \
        --outdir outputs/tifa/nexus_gen --n_samples 1 --fp8_quantization
"""

import argparse, os, sys, torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_loaders import load_prompts
from image_saver import ImageSaver


def _first_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    workspace_root = os.path.abspath(os.path.join(base_dir, ".."))

    default_nexusgen_dir = _first_existing_path([
        os.environ.get("NEXUSGEN_DIR"),
        os.path.join(base_dir, "Nexus-Gen"),
        os.path.join(workspace_root, "Nexus-Gen"),
    ])
    if default_nexusgen_dir is None:
        default_nexusgen_dir = os.path.join(base_dir, "Nexus-Gen")

    default_ckpt_path = _first_existing_path([
        os.environ.get("NEXUSGEN_CKPT_PATH"),
        os.path.join(default_nexusgen_dir, "models", "Nexus-GenV2"),
    ])
    if default_ckpt_path is None:
        default_ckpt_path = os.path.join(default_nexusgen_dir, "models", "Nexus-GenV2")

    default_generation_decoder_path = _first_existing_path([
        os.environ.get("NEXUSGEN_DECODER_PATH"),
        os.path.join(default_ckpt_path, "generation_decoder.bin"),
    ])
    if default_generation_decoder_path is None:
        default_generation_decoder_path = os.path.join(default_ckpt_path, "generation_decoder.bin")

    default_flux_path = _first_existing_path([
        os.environ.get("NEXUSGEN_FLUX_PATH"),
        os.path.join(default_nexusgen_dir, "models"),
    ])
    if default_flux_path is None:
        default_flux_path = os.path.join(default_nexusgen_dir, "models")

    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, required=True,
                   choices=["geneval", "tifa", "drawbench",
                            "color", "shape", "texture", "spatial",
                            "non_spatial", "complex", "numeracy"])
    p.add_argument("--prompt_file", type=str, required=True)
    p.add_argument("--outdir", type=str, default="outputs/nexus_gen")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--embedded_guidance", type=float, default=3.5)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--enable_cpu_offload", action="store_true", default=True)
    p.add_argument("--fp8_quantization", action="store_true", default=False)
    p.add_argument("--nexusgen_dir", type=str,
                   default=default_nexusgen_dir)
    p.add_argument("--ckpt_path", type=str,
                   default=default_ckpt_path)
    p.add_argument("--generation_decoder_path", type=str,
                   default=default_generation_decoder_path)
    p.add_argument("--flux_path", type=str,
                   default=default_flux_path)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.nexusgen_dir):
        raise FileNotFoundError(f"Nexus-Gen repo not found: {args.nexusgen_dir}")
    if not os.path.isdir(args.ckpt_path):
        raise FileNotFoundError(f"Nexus-Gen checkpoint dir not found: {args.ckpt_path}")
    if not os.path.isfile(args.generation_decoder_path):
        raise FileNotFoundError(f"generation_decoder.bin not found: {args.generation_decoder_path}")
    if not os.path.isdir(args.flux_path):
        raise FileNotFoundError(f"Flux/model directory not found: {args.flux_path}")

    prompts = load_prompts(args.benchmark, args.prompt_file)
    saver = ImageSaver(args.benchmark, args.outdir)

    sys.path.insert(0, args.nexusgen_dir)
    from transformers import AutoConfig
    from modeling.decoder.generation_decoder import NexusGenGenerationDecoder
    from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor

    EN_TEMPLATE = "Generate an image according to the following description: {}"

    print("Loading Nexus-Gen V2 AR model...")
    model_config = AutoConfig.from_pretrained(args.ckpt_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt_path, config=model_config, trust_remote_code=True,
        torch_dtype="auto", device_map=args.device,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(args.ckpt_path)
    model.eval()

    flux_decoder = None
    generation_image_grid_thw = torch.tensor([[1, 18, 18]]).to(args.device)

    for entry in tqdm(prompts, desc="Nexus-Gen"):
        for s in range(args.n_samples):
            torch.manual_seed(args.seed + s)
            formatted_prompt = EN_TEMPLATE.format(entry["prompt"])
            messages = [{"role": "user", "content": [{"type": "text", "text": formatted_prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=1024, return_dict_in_generate=True,
                    generation_image_grid_thw=generation_image_grid_thw,
                )
            output_image_embeddings = outputs["output_image_embeddings"]

            if args.enable_cpu_offload:
                model.cpu()
                torch.cuda.empty_cache()

            if flux_decoder is None:
                flux_decoder = NexusGenGenerationDecoder(
                    args.generation_decoder_path, args.flux_path,
                    device=args.device, enable_cpu_offload=args.enable_cpu_offload,
                    fp8_quantization=args.fp8_quantization,
                )

            image = flux_decoder.decode_image_embeds(
                output_image_embeddings, height=args.height, width=args.width,
                negative_prompt=args.negative_prompt, cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                embedded_guidance=args.embedded_guidance, seed=args.seed + s,
            )
            saver.save(image, entry, s)

            if args.enable_cpu_offload:
                model.to(args.device)

    saver.finalize()
    print(f"Nexus-Gen generation complete -> {args.outdir}")


if __name__ == "__main__":
    main()
