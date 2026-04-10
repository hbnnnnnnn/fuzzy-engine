"""
DrawBench evaluation: CLIPScore between generated images and prompts.

Usage (inside 'tifa' or 't2icomp' conda env — needs CLIP):
    python eval_drawbench.py \
        --image_dir outputs/drawbench/sdxl/images \
        --prompts_file ../drawbench/drawbench_prompts.json \
        --output outputs/drawbench/sdxl/drawbench_results.json
"""

import argparse
import json
import os

import torch
import clip
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory with generated images ({id}.png)")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Path to drawbench_prompts.json")
    parser.add_argument("--output", type=str, default="drawbench_results.json",
                        help="Path to save results")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model, device=device)

    with open(args.prompts_file) as f:
        prompts = json.load(f)

    scores = []
    category_scores = {}
    missing = 0

    for i, entry in enumerate(tqdm(prompts, desc="CLIPScore")):
        img_id = f"drawbench_{i:03d}"
        img_path = os.path.join(args.image_dir, f"{img_id}.png")

        if not os.path.exists(img_path):
            missing += 1
            continue

        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        text = clip.tokenize([entry["prompt"]], truncate=True).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(image)
            txt_feat = model.encode_text(text)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).squeeze().item()

        scores.append({"id": img_id, "prompt": entry["prompt"],
                       "category": entry.get("category", ""),
                       "clip_score": sim})

        cat = entry.get("category", "unknown")
        category_scores.setdefault(cat, []).append(sim)

    avg_score = sum(s["clip_score"] for s in scores) / len(scores) if scores else 0.0

    result = {
        "average_clip_score": avg_score,
        "num_images": len(scores),
        "num_missing": missing,
        "category_averages": {
            cat: sum(vals) / len(vals)
            for cat, vals in sorted(category_scores.items())
        },
        "per_image": scores,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"DrawBench CLIPScore: {avg_score:.4f} ({len(scores)} images, {missing} missing)")
    print(f"\nPer-category averages:")
    for cat, avg in result["category_averages"].items():
        print(f"  {cat:<30s} {avg:.4f}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
