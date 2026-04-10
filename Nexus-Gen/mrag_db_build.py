import os

import time

import json

import torch

import faiss

import requests

import numpy as np

from PIL import Image

from io import BytesIO

from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset

from transformers import AutoProcessor, AutoModel



# --- CONFIGURATION ---

OUTPUT_DIR = "mrag-db_orgcap"

MIN_AESTHETIC_SCORE = 6.0  # Lowered slightly to ensure you get data faster for testing

BATCH_SIZE = 16

SAVE_INTERVAL = 500 # Save more often

MAX_SAMPLES = 100000



# Models

MODEL_SIGLIP = "google/siglip-so400m-patch14-384"



# Setup Device

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Running on {device.upper()}")



# --- 1. LOAD MODELS ---



print("📦 Loading SigLIP...")

processor_siglip = AutoProcessor.from_pretrained(MODEL_SIGLIP)

model_siglip = AutoModel.from_pretrained(MODEL_SIGLIP).to(device).eval()



print("✅ Models loaded!")



# --- HELPER FUNCTIONS ---



def get_siglip_features(images=None, texts=None):

    """

    Get embeddings for either images OR text using the same SigLIP model.

    Robustly handles both Tensor and Object return types.

    """

    if texts:

        inputs = processor_siglip(text=texts, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)

        with torch.no_grad():

            outputs = model_siglip.get_text_features(**inputs)

    elif images:

        inputs = processor_siglip(images=images, return_tensors="pt").to(device)

        with torch.no_grad():

            outputs = model_siglip.get_image_features(**inputs)

            

    # --- FIX START: Unpack the object if necessary ---

    if hasattr(outputs, "pooler_output"):

        features = outputs.pooler_output

    else:

        features = outputs

    # --- FIX END ---



    # Normalize features (Critical for SigLIP/CLIP!)

    features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features.cpu().numpy()



def download_image(sample):

    # CRITICAL FIX: Handle multiple possible key names for URL

    url = sample.get("URL") or sample.get("url")

    try:

        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)

        if resp.status_code == 200:

            img = Image.open(BytesIO(resp.content)).convert("RGB")

            if img.size[0] < 100 or img.size[1] < 100: return None

            return {"img": img, "sample": sample}

    except:

        pass

    return None







# --- MAIN LOOP ---



def main():

    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

    

    print("🌊 Connecting to Dataset...")

    ds = load_dataset("laion/laion2B-en-aesthetic", split="train", streaming=True)

    ds = ds.shuffle(seed=42, buffer_size=1000)



    # SigLIP Dimension

    embed_dim = 1152 

    index_img = faiss.IndexFlatIP(embed_dim) 

    index_txt = faiss.IndexFlatIP(embed_dim)

    

    metadata_log = []

    total_processed = 0



    print(f"\n🚀 Pipeline Started. Target: {MAX_SAMPLES}")

    downloader = ThreadPoolExecutor(max_workers=16)

    iterator = iter(ds)

    

    while total_processed < MAX_SAMPLES:

        

        # 1. Fetch Candidates

        candidates = []

        # We try to fill a buffer of (BATCH_SIZE * 2) to account for dead links

        while len(candidates) < BATCH_SIZE * 2: 

            try:

                item = next(iterator)

                

                # --- CRITICAL FIX HERE ---

                # Check 'aesthetic' first (based on your check_cols.py output)

                score = item.get("aesthetic") or item.get("AESTHETIC_SCORE") or 0.0

                score = float(score)

                

                if score >= MIN_AESTHETIC_SCORE:

                    candidates.append(item)

                    

            except StopIteration:

                print("🛑 Dataset stream exhausted.")

                break

            except Exception as e:

                # Skip corrupted rows silently

                continue

        

        if not candidates:

            print("🛑 No more candidates found.")

            break



        # 2. Parallel Download

        futures = list(downloader.map(download_image, candidates))

        valid_data = [f for f in futures if f is not None]



        # 3. Process Batch

        # Process in chunks of BATCH_SIZE

        for i in range(0, len(valid_data), BATCH_SIZE):

            batch = valid_data[i : i + BATCH_SIZE]

            if not batch: continue



            imgs = [b['img'] for b in batch]

            try:

                # A. Use original captions from the dataset
                raw_captions = [b['sample'].get("TEXT") or b['sample'].get("text") or "" for b in batch]

                print(f"   📝 Sample caption: {raw_captions[0][:80]}")

                # B. Embed Images (SigLIP)

                img_embs = get_siglip_features(images=imgs)



                # C. Embed original captions (SigLIP)

                txt_embs = get_siglip_features(texts=raw_captions)



                # D. Add to Index

                index_img.add(img_embs)

                index_txt.add(txt_embs)



                # E. Save Files

                for j, item in enumerate(batch):

                    idx_global = total_processed + j

                    fname = f"{idx_global:06d}.webp"

                    

                    item['img'].save(os.path.join(OUTPUT_DIR, "images", fname), "WEBP")

                    

                    metadata_log.append({

                        "id": idx_global,

                        "filename": fname,

                        "caption": raw_captions[j],

                        "url": item['sample'].get("URL"),

                        "score": float(item['sample'].get("aesthetic") or 0)

                    })



                total_processed += len(batch)

                print(f"✅ Processed {total_processed} | Batch: {len(batch)}")



            except Exception as e:

                print(f"❌ Batch Error: {e}")

                torch.cuda.empty_cache()

                continue



            # F. Checkpoint

            if total_processed % SAVE_INTERVAL == 0:

                print("💾 Saving Checkpoint...")

                faiss.write_index(index_img, os.path.join(OUTPUT_DIR, "image.faiss"))

                faiss.write_index(index_txt, os.path.join(OUTPUT_DIR, "text.faiss"))

                with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:

                    json.dump(metadata_log, f, indent=2)



    # Final Save

    print("💾 Final Save...")

    faiss.write_index(index_img, os.path.join(OUTPUT_DIR, "image.faiss"))

    faiss.write_index(index_txt, os.path.join(OUTPUT_DIR, "text.faiss"))

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:

        json.dump(metadata_log, f, indent=2)

    print("🏁 Done!")



if __name__ == "__main__":

    main()

