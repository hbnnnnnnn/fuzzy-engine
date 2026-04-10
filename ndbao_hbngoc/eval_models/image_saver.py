"""
Benchmark-aware image saver.

Different benchmarks require different directory layouts:

GenEval:
    {outdir}/{index:05d}/metadata.jsonl
    {outdir}/{index:05d}/samples/{sample:05d}.png

T2I-CompBench:
    {outdir}/samples/{prompt}_{seed:06d}.png

TIFA:
    {outdir}/images/{caption_id}.png
    {outdir}/id2img.json   ← mapping {caption_id: "images/{id}.png"}

DrawBench:
    {outdir}/images/{index:03d}.png
    {outdir}/id2img.json   ← mapping {drawbench_XXX: "images/XXX.png"}
"""

import json, os
from PIL import Image


class ImageSaver:
    """Saves generated images in the correct format for a given benchmark."""

    def __init__(self, benchmark, outdir):
        self.benchmark = benchmark
        self.outdir = outdir
        self._id2img = {}  # for TIFA / DrawBench
        os.makedirs(outdir, exist_ok=True)

    def save(self, image: Image.Image, prompt_entry: dict, sample_idx: int):
        """
        Save one image for one prompt.

        prompt_entry: {"id": ..., "prompt": ..., "extra": {...}}
        sample_idx: 0-based index of this sample for this prompt
        """
        if self.benchmark == "geneval":
            self._save_geneval(image, prompt_entry, sample_idx)
        elif self.benchmark in ("tifa", "drawbench"):
            self._save_flat(image, prompt_entry, sample_idx)
        else:
            # T2I-CompBench
            self._save_compbench(image, prompt_entry, sample_idx)

    def _save_geneval(self, image, entry, sample_idx):
        idx = entry["id"]
        outpath = os.path.join(self.outdir, idx)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        # Write metadata once (for sample 0)
        meta_file = os.path.join(outpath, "metadata.jsonl")
        if not os.path.exists(meta_file):
            with open(meta_file, "w") as fp:
                json.dump(entry["extra"], fp)

        out_file = os.path.join(sample_path, f"{sample_idx:05d}.png")
        if os.path.exists(out_file):
            return  # already generated, skip
        image.save(out_file)

    def _save_compbench(self, image, entry, sample_idx):
        """T2I-CompBench: {outdir}/samples/{prompt}_{seed:06d}.png"""
        sample_path = os.path.join(self.outdir, "samples")
        os.makedirs(sample_path, exist_ok=True)
        prompt = entry["prompt"]
        out_file = os.path.join(sample_path, f"{prompt}_{sample_idx:06d}.png")
        if os.path.exists(out_file):
            return  # already generated, skip
        image.save(out_file)

    def _save_flat(self, image, entry, sample_idx):
        """TIFA/DrawBench: {outdir}/images/{id}.png + id2img.json"""
        img_dir = os.path.join(self.outdir, "images")
        os.makedirs(img_dir, exist_ok=True)
        caption_id = entry["id"]
        fname = f"{caption_id}.png"
        out_file = os.path.join(img_dir, fname)
        # Always register in mapping (needed to rebuild id2img.json on resume)
        self._id2img[str(caption_id)] = f"images/{fname}"
        if os.path.exists(out_file):
            return  # already generated, skip
        image.save(out_file)

    def finalize(self):
        """Write any auxiliary files (id2img.json for TIFA/DrawBench)."""
        if self.benchmark in ("tifa", "drawbench") and self._id2img:
            mapping_path = os.path.join(self.outdir, "id2img.json")
            with open(mapping_path, "w") as f:
                json.dump(self._id2img, f, indent=2)
            print(f"Saved id2img mapping: {mapping_path} ({len(self._id2img)} entries)")
