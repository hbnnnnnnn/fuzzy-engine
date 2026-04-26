"""
Generate images for ANY benchmark using Emu3-Gen.
VRAM: ~17 GB (bf16)

Usage:
    python generate_emu3.py --benchmark geneval \
        --prompt_file ../geneval/prompts/evaluation_metadata.jsonl \
        --outdir outputs/geneval/emu3_gen --n_samples 4

    python generate_emu3.py --benchmark tifa \
        --prompt_file ../tifa/tifa_v1.0/tifa_v1.0_text_inputs.json \
        --outdir outputs/tifa/emu3_gen --n_samples 1
"""

import argparse, os, sys, torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_loaders import load_prompts
from image_saver import ImageSaver


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, required=True,
                   choices=["geneval", "tifa", "drawbench",
                            "color", "shape", "texture", "spatial",
                            "non_spatial", "complex", "numeracy"])
    p.add_argument("--prompt_file", type=str, required=True)
    p.add_argument("--outdir", type=str, default="outputs/emu3_gen")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=40960)
    p.add_argument("--ratio", type=str, default="1:1")
    p.add_argument("--model_path", type=str, default="BAAI/Emu3-Gen")
    p.add_argument("--vq_path", type=str, default="BAAI/Emu3-VisionTokenizer")
    return p.parse_args()


def main():
    args = parse_args()
    prompts = load_prompts(args.benchmark, args.prompt_file)
    saver = ImageSaver(args.benchmark, args.outdir)

    from transformers import (
        AutoTokenizer, AutoModel, AutoImageProcessor,
        AutoModelForCausalLM, GenerationConfig,
    )
    from transformers.generation import (
        LogitsProcessorList, PrefixConstrainedLogitsProcessor,
        UnbatchedClassifierFreeGuidanceLogitsProcessor,
    )
    from huggingface_hub import snapshot_download
    local_model_dir = snapshot_download(args.model_path)
    # Load as a package so relative imports inside processing_emu3.py work
    import importlib, importlib.util, types
    pkg = types.ModuleType("emu3_model")
    pkg.__path__ = [local_model_dir]
    pkg.__package__ = "emu3_model"
    sys.modules["emu3_model"] = pkg
    for mod_name in ["utils_emu3", "processing_emu3"]:
        spec = importlib.util.spec_from_file_location(
            f"emu3_model.{mod_name}",
            os.path.join(local_model_dir, f"{mod_name}.py"),
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "emu3_model"
        sys.modules[f"emu3_model.{mod_name}"] = mod
        spec.loader.exec_module(mod)
    Emu3Processor = sys.modules["emu3_model.processing_emu3"].Emu3Processor

    print("Loading Emu3-Gen model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto",
        max_memory={0: "20GiB", "cpu": "48GiB"},
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2", trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(args.vq_path, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(args.vq_path, device_map="auto",
        max_memory={0: "20GiB", "cpu": "48GiB"}, trust_remote_code=True).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    gen_config = GenerationConfig(
        use_cache=True, eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id, max_new_tokens=args.max_new_tokens,
        do_sample=True, top_k=2048,
    )
    POSITIVE_PROMPT = " masterpiece, film grained, best quality."
    NEGATIVE_PROMPT = ("lowres, bad anatomy, bad hands, text, error, missing fingers, "
                       "extra digit, fewer digits, cropped, worst quality, low quality, "
                       "normal quality, jpeg artifacts, signature, watermark, username, blurry.")

    for entry in tqdm(prompts, desc="Emu3-Gen"):
        for s in range(args.n_samples):
            torch.manual_seed(args.seed + s)
            full_prompt = entry["prompt"] + POSITIVE_PROMPT
            kwargs = dict(mode="G", ratio=args.ratio, image_area=model.config.image_area,
                          return_tensors="pt", padding="longest")
            pos_inputs = processor(text=full_prompt, **kwargs)
            neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)
            h, w = pos_inputs.image_size[:, 0], pos_inputs.image_size[:, 1]
            constrained_fn = processor.build_prefix_constrained_fn(h, w)
            logits_processor = LogitsProcessorList([
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    args.cfg_scale, model, unconditional_ids=neg_inputs.input_ids.to("cuda"),
                ),
                PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
            ])
            outputs = model.generate(
                pos_inputs.input_ids.to("cuda"), gen_config,
                logits_processor=logits_processor,
                attention_mask=pos_inputs.attention_mask.to("cuda"),
            )
            mm_list = processor.decode(outputs[0])
            saved = False
            for item in mm_list:
                if isinstance(item, Image.Image):
                    saver.save(item, entry, s)
                    saved = True
                    break
            if not saved:
                saver.save(Image.new("RGB", (512, 512), (0, 0, 0)), entry, s)

    saver.finalize()
    print(f"Emu3-Gen generation complete -> {args.outdir}")


if __name__ == "__main__":
    main()
