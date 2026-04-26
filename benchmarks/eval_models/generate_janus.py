"""
Generate images for ANY benchmark using Janus / Janus-Pro models.

VRAM estimates (bf16):
    deepseek-ai/Janus-1.3B      ~3 GB
    deepseek-ai/Janus-Pro-1B    ~3 GB
    deepseek-ai/Janus-Pro-7B    ~14 GB

Usage:
    python generate_janus.py --benchmark geneval \
        --prompt_file ../geneval/prompts/evaluation_metadata.jsonl \
        --model deepseek-ai/Janus-Pro-7B \
        --outdir outputs/geneval/janus_pro_7b --n_samples 4

    python generate_janus.py --benchmark tifa \
        --prompt_file ../tifa/tifa_v1.0/tifa_v1.0_text_inputs.json \
        --model deepseek-ai/Janus-Pro-7B \
        --outdir outputs/tifa/janus_pro_7b --n_samples 1

    python generate_janus.py --benchmark color \
        --prompt_file ../T2I-CompBench/examples/dataset/color_val.txt \
        --model deepseek-ai/Janus-Pro-7B \
        --outdir outputs/t2i_compbench_color/janus_pro_7b --n_samples 10
"""

import argparse, json, os, sys, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

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
    p.add_argument("--model", type=str, default="deepseek-ai/Janus-Pro-7B",
                   choices=["deepseek-ai/Janus-1.3B",
                            "deepseek-ai/Janus-Pro-1B",
                            "deepseek-ai/Janus-Pro-7B"])
    p.add_argument("--outdir", type=str, default="outputs/janus_pro_7b")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--parallel_size", type=int, default=4)
    p.add_argument("--cfg_weight", type=float, default=5.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=384)
    return p.parse_args()


@torch.inference_mode()
def generate_images(model, processor, prompt,
                    temperature=1.0, parallel_size=4, cfg_weight=5.0,
                    image_token_num_per_image=576, img_size=384, patch_size=16):
    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=processor.sft_format, system_prompt="",
    )
    full_prompt = sft_format + processor.image_start_tag
    input_ids = processor.tokenizer.encode(full_prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    past_key_values = None
    for i in range(image_token_num_per_image):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        logits = model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = model.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return [Image.fromarray(dec[i]) for i in range(parallel_size)]


def main():
    args = parse_args()
    prompts = load_prompts(args.benchmark, args.prompt_file)
    saver = ImageSaver(args.benchmark, args.outdir)

    print(f"Loading model: {args.model}")
    processor = VLChatProcessor.from_pretrained(args.model)
    model = MultiModalityCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
        # flash_attention_2 is set per sub-model in config; outer model uses eager
    ).eval()

    torch.manual_seed(args.seed)

    for entry in tqdm(prompts, desc=args.model.split("/")[-1]):
        sample_count = 0
        while sample_count < args.n_samples:
            batch = min(args.parallel_size, args.n_samples - sample_count)
            images = generate_images(
                model, processor, entry["prompt"],
                temperature=args.temperature, parallel_size=batch,
                cfg_weight=args.cfg_weight, img_size=args.img_size,
            )
            for img in images:
                saver.save(img, entry, sample_count)
                sample_count += 1

    saver.finalize()
    print(f"Janus generation complete -> {args.outdir}")


if __name__ == "__main__":
    main()
