#!/usr/bin/env python3
"""Collect evaluation scores from all benchmarks × models into a summary table."""
import argparse
import json
import os
import re


MODELS = ["sdxl", "janus_1.3b", "janus_pro_1b", "janus_pro_7b", "showo", "emu3", "nexusgen"]
COMPBENCH_CATS = ["color", "shape", "texture", "spatial", "non_spatial", "complex", "numeracy"]


def safe_read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def safe_read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def get_geneval_score(output_root, model):
    """Parse GenEval summary.txt for the overall score."""
    summary = os.path.join(output_root, "geneval", model, "summary.txt")
    text = safe_read_text(summary)
    if not text:
        return None
    # Look for "Overall score" line
    for line in text.splitlines():
        if "overall" in line.lower() and "score" in line.lower():
            m = re.search(r"([\d.]+)\s*$", line)
            if m:
                return float(m.group(1))
    return None


def get_tifa_score(output_root, model):
    """Parse TIFA results JSON for tifa_average."""
    data = safe_read_json(os.path.join(output_root, "tifa", model, "tifa_results.json"))
    if data and "tifa_average" in data:
        return data["tifa_average"]
    return None


def get_drawbench_score(output_root, model):
    """Parse DrawBench results JSON for average_clip_score."""
    data = safe_read_json(os.path.join(output_root, "drawbench", model, "drawbench_results.json"))
    if data and "average_clip_score" in data:
        return data["average_clip_score"]
    return None


def get_compbench_score(output_root, cat, model):
    """Parse T2I-CompBench score for a given category."""
    base = os.path.join(output_root, "t2i_compbench", cat, model)

    if cat in ("color", "shape", "texture"):
        txt = safe_read_text(os.path.join(base, "annotation_blip", "blip_vqa_score.txt"))
        if txt:
            m = re.search(r"([\d.]+)", txt)
            if m:
                return float(m.group(1))
    elif cat == "spatial":
        txt = safe_read_text(os.path.join(base, "annotation_obj_detection_2d", "avg_score.txt"))
        if txt:
            m = re.search(r"([\d.]+)", txt)
            if m:
                return float(m.group(1))
    elif cat == "non_spatial":
        txt = safe_read_text(os.path.join(base, "annotation_clip", "score_avg.txt"))
        if txt:
            m = re.search(r"([\d.]+)", txt)
            if m:
                return float(m.group(1))
    elif cat == "complex":
        txt = safe_read_text(os.path.join(base, "annotation_3_in_1", "vqa_score.txt"))
        if txt:
            m = re.search(r"([\d.]+)", txt)
            if m:
                return float(m.group(1))
    elif cat == "numeracy":
        txt = safe_read_text(os.path.join(base, "annotation_num", "score.txt"))
        if txt:
            m = re.search(r"avg[:\s]*([\d.]+)", txt)
            if m:
                return float(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    # Build header
    benchmarks = ["GenEval", "TIFA", "DrawBench"] + [f"T2I-{c}" for c in COMPBENCH_CATS]
    header = f"{'Model':<18}" + "".join(f"{b:>14}" for b in benchmarks)
    sep = "-" * len(header)

    rows = []
    for model in MODELS:
        scores = []
        scores.append(get_geneval_score(args.output_root, model))
        scores.append(get_tifa_score(args.output_root, model))
        scores.append(get_drawbench_score(args.output_root, model))
        for cat in COMPBENCH_CATS:
            scores.append(get_compbench_score(args.output_root, cat, model))

        cells = []
        for s in scores:
            if s is not None:
                cells.append(f"{s:>14.4f}")
            else:
                cells.append(f"{'—':>14}")
        rows.append(f"{model:<18}" + "".join(cells))

    # Print and save
    summary = "\n".join([header, sep] + rows + [""])
    print(summary)

    out_file = os.path.join(args.output_root, "summary_all.txt")
    with open(out_file, "w") as f:
        f.write(summary)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
