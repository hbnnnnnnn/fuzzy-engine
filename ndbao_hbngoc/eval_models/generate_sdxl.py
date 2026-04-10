"""
Generate images for ANY benchmark using SDXL.
VRAM: ~7 GB (fp16)

Usage:
    # GenEval
    python generate_sdxl.py --benchmark geneval \
        --prompt_file ../geneval/prompts/evaluation_metadata.jsonl \
        --outdir outputs/geneval/sdxl --n_samples 4

    # TIFA
    python generate_sdxl.py --benchmark tifa \
        --prompt_file ../tifa/tifa_v1.0/tifa_v1.0_text_inputs.json \
        --outdir outputs/tifa/sdxl --n_samples 1

    # DrawBench
    python generate_sdxl.py --benchmark drawbench \
        --prompt_file ../drawbench/drawbench_prompts.json \
        --outdir outputs/drawbench/sdxl --n_samples 1

    # T2I-CompBench (per category)
    python generate_sdxl.py --benchmark color \
        --prompt_file ../T2I-CompBench/examples/dataset/color_val.txt \
        --outdir outputs/t2i_compbench_color/sdxl --n_samples 10
"""

import argparse, os, sys, torch
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_loaders import load_prompts
from image_saver import ImageSaver


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, required=True,
                   choices=["geneval", "tifa", "drawbench",
                            "color", "shape", "texture", "spatial",
                            "non_spatial", "complex", "numeracy"],
                   help="Benchmark name")
    p.add_argument("--prompt_file", type=str, required=True)
    p.add_argument("--outdir", type=str, default="outputs/sdxl")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--scale", type=float, default=9.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--H", type=int, default=1024)
    p.add_argument("--W", type=int, default=1024)
    p.add_argument("--negative_prompt", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    prompts = load_prompts(args.benchmark, args.prompt_file)
    saver = ImageSaver(args.benchmark, args.outdir)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    for entry in tqdm(prompts, desc="SDXL"):
        for s in range(args.n_samples):
            generator = torch.Generator(device="cuda").manual_seed(args.seed + s)
            images = pipe(
                entry["prompt"],
                height=args.H, width=args.W,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                num_images_per_prompt=1,
                negative_prompt=args.negative_prompt,
                generator=generator,
            ).images
            saver.save(images[0], entry, s)

    saver.finalize()
    print(f"SDXL generation complete -> {args.outdir}")


if __name__ == "__main__":
    main()
