"""
Generate images for ANY benchmark using Show-O (1.3B, 512x512).
VRAM: ~6 GB (bf16)

Usage:
    python generate_showo.py --benchmark geneval \
        --prompt_file ../geneval/prompts/evaluation_metadata.jsonl \
        --outdir outputs/geneval/show_o --n_samples 4

    python generate_showo.py --benchmark color \
        --prompt_file ../T2I-CompBench/examples/dataset/color_val.txt \
        --outdir outputs/t2i_compbench_color/show_o --n_samples 10
"""

import argparse, json, os, sys, torch
import numpy as np
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
    default_showo_dir = _first_existing_path([
        os.environ.get("SHOWO_DIR"),
        os.path.join(base_dir, "Show-o"),
        os.path.join(workspace_root, "Show-o"),
    ])
    if default_showo_dir is None:
        default_showo_dir = os.path.join(base_dir, "Show-o")

    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, required=True,
                   choices=["geneval", "tifa", "drawbench",
                            "color", "shape", "texture", "spatial",
                            "non_spatial", "complex", "numeracy"])
    p.add_argument("--prompt_file", type=str, required=True)
    p.add_argument("--outdir", type=str, default="outputs/show_o")
    p.add_argument("--showo_dir", type=str,
                   default=default_showo_dir)
    p.add_argument("--model_name", type=str, default="showlab/show-o-512x512")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--generation_timesteps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(args.showo_dir, "configs", "showo_demo_512x512.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Show-O config not found: {config_path}")

    prompts = load_prompts(args.benchmark, args.prompt_file)
    saver = ImageSaver(args.benchmark, args.outdir)

    # Add Show-o repo to path
    sys.path.insert(0, args.showo_dir)
    from omegaconf import OmegaConf
    from models import Showo, MAGVITv2, get_mask_chedule
    from training.prompting_utils import UniversalPrompting
    from transformers import AutoTokenizer

    config = OmegaConf.load(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                        "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
    )
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to("cuda").eval()
    model = Showo.from_pretrained(args.model_name).to("cuda").eval()
    mask_token_id = model.config.mask_token_id

    # Build noise schedule callable (same logic as inference_t2i.py)
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        schedule_args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **schedule_args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    noise_type = config.training.get("noise_type", "mask")

    num_vq_tokens = config.model.showo.num_vq_tokens
    for entry in tqdm(prompts, desc="Show-O"):
        for s in range(args.n_samples):
            torch.manual_seed(args.seed + s)
            image_tokens = torch.ones((1, num_vq_tokens), dtype=torch.long, device="cuda") * mask_token_id
            input_ids, _ = uni_prompting(([entry["prompt"]], image_tokens), 't2i_gen')
            input_ids = input_ids[:1].to("cuda")
            input_ids_minus_lm = torch.where(
                input_ids >= tokenizer.vocab_size, mask_token_id, input_ids,
            )
            with torch.no_grad():
                generated_ids = model.t2i_generate(
                    input_ids=input_ids_minus_lm,
                    uncond_input_ids=None,
                    attention_mask=torch.ones_like(input_ids_minus_lm),
                    guidance_scale=args.guidance_scale,
                    temperature=1.0,
                    timesteps=args.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=noise_type,
                    seq_len=num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )
            generated_image = vq_model.decode_code(generated_ids)
            generated_image = torch.clamp((generated_image + 1.0) / 2.0, 0.0, 1.0)
            arr = (generated_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            saver.save(Image.fromarray(arr), entry, s)

    saver.finalize()
    print(f"Show-O generation complete -> {args.outdir}")


if __name__ == "__main__":
    main()
