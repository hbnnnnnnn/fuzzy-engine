"""
Test suite for retrieve_mrag.py
================================
5 tests covering:
  1. Basic text query  (nature / landscape)
  2. Food / object text query
  3. Sports / event text query
  4. Abstract / concept text query
  5. Image-as-query (query by example using an image already in the DB)

Run with:
    python test_retrieval.py                 # all tests
    python test_retrieval.py -v              # verbose output
    python test_retrieval.py TestRetrieval.test_01 -v   # single test
"""

import os
import sys
import shutil
import unittest
import textwrap
from pathlib import Path
from PIL import Image

# ---- point at the repo root so imports work regardless of cwd --------
REPO_ROOT = Path(__file__).resolve().parent          # Nexus-Gen/
DB_PATH   = str(REPO_ROOT / "mrag-db-orgcap")
OUT_DIR: str | None = None   # set via --out-dir; None = no copying

sys.path.insert(0, str(REPO_ROOT))
from retrieve_mrag import MRAGRetriever

# ---- shared retriever (loaded once for the whole test session) -------
_RETRIEVER: MRAGRetriever | None = None


def get_retriever() -> MRAGRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = MRAGRetriever(db_path=DB_PATH)
    return _RETRIEVER


# ─────────────────────────────────────────────────────────────────────
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _copy_results(tag: str, results: list) -> None:
    """Copy retrieved images into OUT_DIR/<safe_tag>/rank_N_id_XXXXX.<ext>"""
    if OUT_DIR is None:
        return
    # e.g. "TEST-01  nature/landscape" → "TEST-01_nature_landscape"
    safe = tag.strip().replace("/", "_").replace(" ", "_").replace("–", "-")
    dest = Path(OUT_DIR) / safe
    dest.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results, 1):
        src = Path(r["image_path"])
        if src.is_file():
            dst = dest / f"rank{i}_id{r['id']:06d}{src.suffix}"
            shutil.copy2(src, dst)
    print(f"[COPY] {len(results)} image(s)  →  {dest}")


def _print_results(tag: str, query, results: list) -> None:
    sep = "─" * 70
    print(f"\n{BOLD}{sep}{RESET}")
    print(f"{BOLD}{YELLOW}[{tag}]{RESET}  Query: {repr(query)!s:.80}")
    print(sep)
    for i, r in enumerate(results, 1):
        caption = textwrap.shorten(r["caption"] or "(no caption)", width=72)
        print(
            f"  {GREEN}#{i}{RESET}  id={r['id']:>6}  score={r['retrieval_score']:.4f}"
            f"  aesthetic={r['aesthetic_score']:.2f}"
        )
        print(f"       caption: {caption}")
        print(f"       path:    {r['image_path']}")
    print(sep)
    _copy_results(tag, results)
# ─────────────────────────────────────────────────────────────────────


class TestRetrieval(unittest.TestCase):
    """Integration tests for the MRAG hybrid + MMR retriever."""

    # ----------------------------------------------------------------
    # Shared assertions applied to every result set
    # ----------------------------------------------------------------
    def _check_results(self, results: list, expected_k: int = 3) -> None:
        self.assertIsInstance(results, list, "retrieve() must return a list")
        self.assertLessEqual(
            len(results), expected_k,
            f"Got {len(results)} results, expected at most {expected_k}",
        )
        self.assertGreater(len(results), 0, "Result list must not be empty")

        seen_ids = set()
        for r in results:
            # required keys
            for key in ("id", "filename", "image_path", "caption", "retrieval_score"):
                self.assertIn(key, r, f"Missing key '{key}' in result dict")

            # non-negative score
            self.assertGreaterEqual(
                r["retrieval_score"], 0.0,
                f"retrieval_score must be >= 0, got {r['retrieval_score']}",
            )

            # image file must exist
            self.assertTrue(
                os.path.isfile(r["image_path"]),
                f"Image file not found: {r['image_path']}",
            )

            # MMR should not return duplicates
            self.assertNotIn(r["id"], seen_ids, f"Duplicate id {r['id']} in results")
            seen_ids.add(r["id"])

    # ----------------------------------------------------------------
    # Test 1 – Nature / outdoor landscape
    # ----------------------------------------------------------------
    def test_01_nature_landscape(self) -> None:
        """Text query: outdoor nature scene."""
        query   = "wooden bridge over a calm lake surrounded by trees in summer"
        results = get_retriever().retrieve(query, k=3, lambda_mmr=0.9)
        _print_results("TEST-01  nature/landscape", query, results)
        self._check_results(results, expected_k=3)

    # ----------------------------------------------------------------
    # Test 2 – Food / object
    # ----------------------------------------------------------------
    def test_02_food_object(self) -> None:
        """Text query: food / close-up object."""
        query   = "beautiful bouquet of pink peonies and roses"
        results = get_retriever().retrieve(query, k=3, lambda_mmr=0.9)
        _print_results("TEST-02  food/object", query, results)
        self._check_results(results, expected_k=3)
        # At least the top result should have a meaningful score
        self.assertGreater(results[0]["retrieval_score"], 0.0)

    # ----------------------------------------------------------------
    # Test 3 – Sports / event
    # ----------------------------------------------------------------
    def test_03_sports_event(self) -> None:
        """Text query: sports or public event."""
        query   = "soccer player celebrating championship trophy victory"
        results = get_retriever().retrieve(query, k=3, lambda_mmr=0.9)
        _print_results("TEST-03  sports/event", query, results)
        self._check_results(results, expected_k=3)

    # ----------------------------------------------------------------
    # Test 4 – Abstract / conceptual (tests cross-modal generalisation)
    # ----------------------------------------------------------------
    def test_04_abstract_concept(self) -> None:
        """Text query: abstract or conceptual description (harder cross-modal)."""
        query   = "warm candle light glowing in winter darkness holiday atmosphere"
        results = get_retriever().retrieve(query, k=3, lambda_mmr=0.9)
        _print_results("TEST-04  abstract/concept", query, results)
        self._check_results(results, expected_k=3)
        # MMR with λ=0.9 should still select 3 *different* items
        ids = [r["id"] for r in results]
        self.assertEqual(len(ids), len(set(ids)), "MMR returned duplicate ids")

    # ----------------------------------------------------------------
    # Test 5 – Image-as-query  (query by example)
    # ----------------------------------------------------------------
    def test_05_image_query(self) -> None:
        """Image query: use an existing DB image as the query."""
        # Grab the first image stored in the DB (id=0)
        first_meta_path = os.path.join(DB_PATH, "images", "000000.webp")
        if not os.path.isfile(first_meta_path):
            self.skipTest(f"Sample image not found: {first_meta_path}")

        query_image = Image.open(first_meta_path).convert("RGB")
        results = get_retriever().retrieve(
            query_image,
            k=3,
            lambda_mmr=0.9,
            query_type="image",
        )
        _print_results("TEST-05  image-as-query (id=0)", "<PIL.Image 000000.webp>", results)
        self._check_results(results, expected_k=3)
        # The query image itself (id=0) should appear in top results
        returned_ids = [r["id"] for r in results]
        self.assertIn(
            0, returned_ids,
            "Query-by-example with id=0 should retrieve itself in top-3",
        )


# ─────────────────────────────────────────────────────────────────────
# Quick smoke-test that can be run without unittest
# ─────────────────────────────────────────────────────────────────────
def run_smoke_tests() -> None:
    """Run all 5 tests and show a compact pass/fail summary."""
    retriever = get_retriever()
    k = 3

    tests = [
        (
            "text",
            "TEST-01 nature/landscape",
            "wooden bridge over a calm lake surrounded by trees in summer",
        ),
        (
            "text",
            "TEST-02 food/object",
            "beautiful bouquet of pink peonies and roses",
        ),
        (
            "text",
            "TEST-03 sports/event",
            "soccer player celebrating championship trophy victory",
        ),
        (
            "text",
            "TEST-04 abstract/concept",
            "warm candle light glowing in winter darkness holiday atmosphere",
        ),
        (
            "image",
            "TEST-05 image-as-query",
            os.path.join(DB_PATH, "images", "000000.webp"),
        ),
    ]

    passed = 0
    for qtype, tag, q in tests:
        try:
            if qtype == "image":
                if not os.path.isfile(q):
                    print(f"{YELLOW}[SKIP]{RESET} {tag}: image file not found ({q})")
                    continue
                query = Image.open(q).convert("RGB")
            else:
                query = q

            results = retriever.retrieve(query, k=k, lambda_mmr=0.9, query_type=qtype)
            _print_results(tag, q, results)

            assert len(results) > 0,            "empty result list"
            assert len(results) <= k,            f"too many results: {len(results)}"
            assert len({r['id'] for r in results}) == len(results), "duplicate ids"
            for r in results:
                assert os.path.isfile(r["image_path"]), f"missing file: {r['image_path']}"

            print(f"{GREEN}[PASS]{RESET} {tag}")
            passed += 1
        except Exception as exc:
            print(f"{RED}[FAIL]{RESET} {tag}: {exc}")

    total = len(tests)
    colour = GREEN if passed == total else (YELLOW if passed > 0 else RED)
    print(f"\n{colour}{BOLD}Results: {passed}/{total} tests passed.{RESET}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MRAG retriever tests")
    parser.add_argument(
        "--unittest",
        action="store_true",
        help="Run as proper unittest.TestCase (adds assertions)",
    )
    parser.add_argument(
        "--db", default=DB_PATH, help="Path to mrag-db-orgcap directory"
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="If set, copy retrieved images into <out-dir>/<test-name>/rank1_... etc."
    )
    args, remaining = parser.parse_known_args()

    if args.db != DB_PATH:
        DB_PATH = args.db

    if args.out_dir:
        OUT_DIR = args.out_dir

    if args.unittest:
        # Pass remaining args to unittest (e.g. -v, test method name)
        sys.argv = [sys.argv[0]] + remaining
        unittest.main()
    else:
        run_smoke_tests()
