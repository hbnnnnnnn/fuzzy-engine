"""
Unified prompt loader for all T2I benchmarks.

Each benchmark has a different prompt format. This module normalizes them into
a common list of dicts:  [{"id": ..., "prompt": ..., "extra": {...}}, ...]

Supported benchmarks:
    geneval       — JSONL with {tag, include, prompt}
    tifa          — JSON array with {id, caption, coco_val_id}
    drawbench     — JSON array with {prompt, category}
    t2i_compbench — plain .txt file, one prompt per line
"""

import json, os


def load_geneval_prompts(metadata_file):
    """Load GenEval evaluation_metadata.jsonl"""
    prompts = []
    with open(metadata_file) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            prompts.append({
                "id": f"{i:05d}",
                "prompt": d["prompt"],
                "extra": d,
            })
    return prompts


def load_tifa_prompts(text_inputs_file):
    """Load TIFA tifa_v1.0_text_inputs.json"""
    with open(text_inputs_file) as f:
        data = json.load(f)
    return [
        {
            "id": item["id"],
            "prompt": item["caption"],
            "extra": item,
        }
        for item in data
    ]


def load_drawbench_prompts(prompts_file):
    """Load DrawBench drawbench_prompts.json"""
    with open(prompts_file) as f:
        data = json.load(f)
    return [
        {
            "id": f"drawbench_{i:03d}",
            "prompt": item["prompt"],
            "extra": item,
        }
        for i, item in enumerate(data)
    ]


def load_t2i_compbench_prompts(txt_file):
    """Load T2I-CompBench prompt txt file (one prompt per line)"""
    prompts = []
    with open(txt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append({
                    "id": line,   # prompt itself is the ID for compbench
                    "prompt": line,
                    "extra": {"source_file": os.path.basename(txt_file)},
                })
    return prompts


def load_prompts(benchmark, prompt_path):
    """
    Universal loader. Returns list of {"id", "prompt", "extra"}.

    benchmark: one of "geneval", "tifa", "drawbench",
               "color", "shape", "texture", "spatial", "non_spatial",
               "complex", "numeracy" (T2I-CompBench categories)
    prompt_path: path to the prompt file
    """
    if benchmark == "geneval":
        return load_geneval_prompts(prompt_path)
    elif benchmark == "tifa":
        return load_tifa_prompts(prompt_path)
    elif benchmark == "drawbench":
        return load_drawbench_prompts(prompt_path)
    else:
        # T2I-CompBench categories
        return load_t2i_compbench_prompts(prompt_path)
