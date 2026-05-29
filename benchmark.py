import json
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from embed_EGA import vectorizer, TMFG


def load_and_embed(vec, json_path, batch_size=32):
    """Load a translation JSON and return embeddings for original and translation."""
    embeds = vec.process_json(json_path, batch_size=batch_size)
    return {
        'original': embeds['original']['embeddings'],
        'translation': embeds['translation']['embeddings'],
        'item_numbers': embeds['item_numbers']
    }


def compare_tmfg(ega, emb_a, emb_b, label_a, label_b):
    """
    Run TMFG on two embedding sets and compute structural metrics.
    No UVA or bootEGA — uses all items.
    """
    # Cosine correlation (embeddings are L2-normalized)
    corr_a = ega.compute_corr(emb_a)
    corr_b = ega.compute_corr(emb_b)

    # TMFG networks
    net_a = ega.run_tmfg(corr_a)
    net_b = ega.run_tmfg(corr_b)

    # Community detection
    mem_a, _ = ega.calculate_membership(corr_a)
    mem_b, _ = ega.calculate_membership(corr_b)

    # Structural metrics
    comparison = ega.compare_networks(
        net_a["adjacency"], net_b["adjacency"],
        mem_a, mem_b
    )

    print(f"  {label_a} vs {label_b}: "
          f"Jaccard={comparison['jaccard']:.4f}, "
          f"WeightCorr={comparison['weight_corr']:.4f}, "
          f"NMI={comparison['nmi']:.4f}")

    return comparison


def run_benchmark(expert_json: str, results_dir: str, fwd_files: list = None,
                  output_path: str = None):
    """
    Run benchmark comparisons:
      1. Expert vs Original (1 comparison)
      2. Expert vs each Forward Translation (N comparisons)
    """
    # Initialize
    vec = vectorizer()
    ega = TMFG()

    # Load expert embeddings
    print(f"\nLoading expert translation: {expert_json}")
    expert = load_and_embed(vec, expert_json)

    # Expert vs Original
    print(f"\n{'=' * 60}")
    print("BENCHMARK: Expert vs Original")
    print(f"{'=' * 60}")
    expert_vs_orig = compare_tmfg(
        ega,
        expert['original'],       # English original
        expert['translation'],    # Expert Chinese translation
        "Original", "Expert"
    )

    # Find forward translation files
    if fwd_files:
        fwd_paths = [os.path.join(results_dir, f) for f in fwd_files]
    else:
        fwd_paths = sorted(glob.glob(os.path.join(results_dir, "fwd_*.json")))

    if not fwd_paths:
        print(f"\nNo forward translation files found in {results_dir}")
        return

    # Expert vs each forward translation
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: Expert vs {len(fwd_paths)} Forward Translations")
    print(f"{'=' * 60}")

    all_results = [{
        "comparison": "expert_vs_original",
        "file": expert_json,
        **expert_vs_orig
    }]

    for fwd_path in fwd_paths:
        fname = os.path.basename(fwd_path)
        print(f"\n--- {fname} ---")

        try:
            fwd = load_and_embed(vec, fwd_path)

            # Expert (Chinese) vs Forward Translation (Chinese)
            # Both are translations, so we compare the translation fields
            comp = compare_tmfg(
                ega,
                expert['translation'],   # Expert Chinese
                fwd['translation'],       # LLM Chinese
                "Expert", fname
            )

            all_results.append({
                "comparison": f"expert_vs_{fname}",
                "file": fname,
                **comp
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    # Summary table
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")

    df = pd.DataFrame(all_results)
    print(df[['comparison', 'jaccard', 'weight_corr', 'nmi']].to_string(index=False))

    # Save
    output_path = output_path or os.path.join(results_dir, "benchmark_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark: Expert vs Forward Translations")
    parser.add_argument("expert_json", help="Path to expert translation JSON")
    parser.add_argument("results_dir", help="Directory containing fwd_*.json files")
    parser.add_argument("--fwd-files", nargs="*", help="Specific forward files to compare (optional)")
    parser.add_argument("--output", help="Output CSV path (default: results_dir/benchmark_summary.csv)")

    args = parser.parse_args()

    run_benchmark(
        expert_json=args.expert_json,
        results_dir=args.results_dir,
        fwd_files=args.fwd_files,
        output_path=args.output
    )