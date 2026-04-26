"""
TIFA evaluation wrapper.

Runs tifa_score_benchmark on generated images.

Usage (inside 'tifa' conda env):
    python eval_tifa.py \
        --qa_file ../tifa/tifa_v1.0/tifa_v1.0_question_answers.json \
        --id2img  outputs/tifa/sdxl/id2img.json \
        --output  outputs/tifa/sdxl/tifa_results.json \
        --vqa_model mplug-large
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", type=str, required=True,
                        help="Path to tifa_v1.0_question_answers.json")
    parser.add_argument("--id2img", type=str, required=True,
                        help="Path to id2img.json produced by image_saver")
    parser.add_argument("--vqa_model", type=str, default="mplug-large",
                        help="VQA model to use (default: mplug-large)")
    parser.add_argument("--output", type=str, default="tifa_results.json",
                        help="Path to save TIFA score results")
    args = parser.parse_args()

    from tifascore import tifa_score_benchmark

    result = tifa_score_benchmark(args.vqa_model, args.qa_file, args.id2img)

    # Save full results
    with open(args.output, "w") as f:
        # question_details has nested defaultdicts; convert to plain dict
        serializable = {
            "tifa_average": result["tifa_average"],
            "tifa_stdev": result["tifa_stdev"],
            "accuracy_by_type": result["accuracy_by_type"],
            "caption_scores": result["caption_scores"],
        }
        json.dump(serializable, f, indent=2)

    print(f"TIFA average score: {result['tifa_average']:.4f}")
    print(f"TIFA stdev:         {result['tifa_stdev']:.4f}")
    print(f"Results saved to:   {args.output}")

    # Print per-type breakdown
    print("\nAccuracy by question type:")
    for qtype, score in sorted(result["accuracy_by_type"].items()):
        print(f"  {qtype:<20s} {score:.4f}")


if __name__ == "__main__":
    main()
