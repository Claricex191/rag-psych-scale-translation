import requests
import json
import itertools
from pathlib import Path
from embed_EGA import *
import sys
import pandas as pd
import numpy as np
import re


# ========================== Configuration ==========================

MODELS = ["gpt-5.2-2025-12-11", "claude-opus-4-5-20251101", "gemini-3-pro-preview", "qwen3-vl-plus-2025-12-19"]
TEMPERATURES = [0.6, 0.7, 0.8, 0.9, 1.0]

FORWARD_SERVER_URL = "http://127.0.0.1:8000"
BACKWARD_SERVER_URL = "http://127.0.0.1:8001"  # Assumes a separate back-translation server

# ========================== RAG Client ==========================

class RAGClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.session_id = None

    def check_familiarity(self, scale_path: str, model: str):
        payload = {
            "scale_path": scale_path,
            "model": model,
        }
        if self.session_id:
            payload["session_id"] = self.session_id

        response = requests.post(f"{self.base_url}/check", json=payload)
        result = response.json()

        if result.get("session_id"):
            self.session_id = result["session_id"]

        return result

    def translate(self, scale_path: str, model: str, temperature: float = 0.7,
                  limit: int = 30, extract_from_top: int = 5):
        if not self.session_id:
            return {"success": False, "error": "No session_id. Run check_familiarity first."}

        payload = {
            "session_id": self.session_id,
            "scale_path": scale_path,
            "model": model,
            "temperature": temperature,
            "limit": limit,
            "extract_from_top": extract_from_top,
        }

        response = requests.post(f"{self.base_url}/translate", json=payload)
        result = response.json()

        if result.get("session_id"):
            self.session_id = result["session_id"]

        return result

    def back_translate(self, scale_path: str, model: str, temperature: float = 0.7,
                       limit: int = 30, extract_from_top: int = 5):
        payload = {
            "scale_path": scale_path,
            "model": model,
            "temperature": temperature,
            "limit": limit,
            "extract_from_top": extract_from_top,
        }
        if self.session_id:
            payload["session_id"] = self.session_id

        response = requests.post(f"{self.base_url}/translate", json=payload)
        return response.json()

    def reset_session(self):
        """Reset session state."""
        self.session_id = None


# ========================== Helpers ==========================

def save_json(data, output_path: str):
    try:
        if isinstance(data, str):
            # Strip markdown code fences if present
            cleaned = data.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```\s*$', '', cleaned)
                cleaned = cleaned.strip()

            if cleaned.startswith('{'):
                data = json.loads(cleaned)
            else:
                data = {"raw": data}

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON to {output_path}: {e}")
        return False


def print_translation_preview(translation_text: str, max_items: int = 3):
    try:
        if isinstance(translation_text, str) and translation_text.strip().startswith('{'):
            json_data = json.loads(translation_text)
        elif isinstance(translation_text, dict):
            json_data = translation_text
        else:
            print(f"\n{translation_text}\n")
            return

        if 'items' in json_data and isinstance(json_data['items'], list):
            items = json_data['items']
            print(f"\n  Translated {len(items)} items. Preview:")
            for item in items[:max_items]:
                num = item.get('number', '?')
                trans = item.get('translation', '')
                print(f"    {num}. {trans}")
            if len(items) > max_items:
                print(f"    ... and {len(items) - max_items} more items")
            print()
        else:
            print(f"\n{json_data}\n")
    except (json.JSONDecodeError, TypeError):
        print(f"\n{translation_text}\n")


def make_output_dir(base_dir: str = "results"):
    """Create output directory if needed."""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return base_dir


# ========================== Network Evaluation ==========================

def process_single_embedding(embedding, item_numbers, label: str, ega, uva, bootega):
    """
    Run UVA + bootEGA on one embedding set and compute its final TMFG network.
    Returns a dict of pre-processed data ready for pairwise comparison and visualization.
    """
    data = embedding.T  # [items × dims] → [dims × items]

    print(f"  Running UVA for {label}...")
    reduced = uva.run_uva_iterative(data, threshold=0.25)
    kept = reduced['kept_items']
    print(f"  UVA: {label} kept {len(kept)}/{data.shape[1]}")

    print(f"  Running bootEGA for {label}...")
    boot = bootega.run_bootega_iterative(data[:, kept], n_boots=100, stability_threshold=0.75)
    final_items = [kept[i] for i in boot['kept_items']]
    final_idx = [item_numbers[i] for i in final_items]
    data_final = data[:, final_items]
    print(f"  Final: {label} {len(final_items)} items")

    corr_final = np.corrcoef(data_final.T)
    net_final = ega.run_tmfg(corr_final)
    mem, _ = ega.calculate_membership(corr_final)

    return {
        "corr_final": corr_final,
        "net_final": net_final,
        "mem": mem,
        "final_idx": final_idx,
        "label": label,
    }


def run_three_way_eval(forward_json_path: str, back_json_path: str, output_dir: str, tag: str):
    """
    Run three network comparisons:
      1. Original vs Forward Translation
      2. Forward Translation vs Back Translation
      3. Original vs Back Translation
    Each version's embedding is processed (UVA + bootEGA) and visualized exactly once.
    """
    vec = vectorizer()
    ega_obj = TMFG()
    uva_obj = UVA()
    bootega_obj = BootEGA()

    # Load embeddings
    fwd_embeds = vec.process_json(forward_json_path, batch_size=32)
    orig_embedding = fwd_embeds['original']['embeddings']
    fwd_embedding = fwd_embeds['translation']['embeddings']
    item_numbers = fwd_embeds['item_numbers']

    back_embeds = vec.process_json(back_json_path, batch_size=32)
    back_embedding = back_embeds['translation']['embeddings']

    prefix = f"{output_dir}/{tag}"

    # Process each version once (UVA + bootEGA + TMFG)
    print(f"\n{'=' * 60}")
    print("Processing embeddings (UVA + bootEGA) — each version once")
    print(f"{'=' * 60}")
    proc_orig = process_single_embedding(orig_embedding, item_numbers, "Original",       ega_obj, uva_obj, bootega_obj)
    proc_fwd  = process_single_embedding(fwd_embedding,  item_numbers, "Forward",        ega_obj, uva_obj, bootega_obj)
    proc_back = process_single_embedding(back_embedding,  item_numbers, "BackTranslation", ega_obj, uva_obj, bootega_obj)

    # Visualize each network once
    print(f"\n{'=' * 60}")
    print("Generating network visualizations — each version once")
    print(f"{'=' * 60}")
    for proc in (proc_orig, proc_fwd, proc_back):
        lbl = proc["label"].lower().replace(' ', '_')
        ega_obj.visualize_network(
            corr_matrix=proc["corr_final"], membership=proc["mem"], item_labels=proc["final_idx"],
            output_file=f"{prefix}_{lbl}_network.png",
            title=f'{proc["label"]} TMFG Network (After bootEGA)',
            node_size=8, label_size=0.8
        )

    # Pairwise comparisons using pre-computed results
    all_comparisons = {}
    pairs = [
        ("orig_vs_fwd",  proc_orig, proc_fwd,  "Original vs Forward Translation"),
        ("fwd_vs_back",  proc_fwd,  proc_back, "Forward Translation vs Back Translation"),
        ("orig_vs_back", proc_orig, proc_back, "Original vs Back Translation"),
    ]

    for i, (key, pa, pb, title) in enumerate(pairs, 1):
        print(f"\n{'#' * 60}")
        print(f"COMPARISON {i}/3: {title}")
        print(f"{'#' * 60}")

        target_mem = np.ones(len(pa["final_idx"]), dtype=int)
        comparison = ega_obj.compare_networks(
            pa["net_final"]["adjacency"],
            pb["net_final"]["adjacency"],
            target_mem,
            pb["mem"]
        )
        all_comparisons[key] = comparison

    summary_df = pd.DataFrame(all_comparisons).T
    summary_path = f"{prefix}_summary.csv"
    summary_df.to_csv(summary_path)

    print(f"\n{'=' * 60}")
    print(f"Summary saved to {summary_path}")
    print(summary_df)
    print(f"{'=' * 60}\n")

    return all_comparisons


# ========================== Two-Stage Translation ==========================

def do_forward_translation(fwd_client: RAGClient, scale_path: str,
                           fwd_model: str, temperature: float, output_path: str):
    """
    Stage 1: familiarity check (with user confirmation gate)
    Stage 2: forward translation with RAG
    """
    fwd_client.reset_session()

    # --- Stage 1: Familiarity Check ---
    print(f"\n--- Stage 1: Familiarity Check ({fwd_model}, temp={temperature}) ---")
    check_result = fwd_client.check_familiarity(scale_path, fwd_model)

    if not check_result.get("success"):
        print(f"  ✗ Familiarity check failed: {check_result.get('error')}")
        return False

    print(f"\n  Familiarity Summary:\n")
    summary = check_result.get("familiarity_summary", "")
    # Indent the summary for readability
    for line in summary.split('\n'):
        print(f"    {line}")
    print()

    # --- User Confirmation Gate ---
    while True:
        choice = input("  Proceed to forward translation? (yes/no/skip): ").strip().lower()
        if choice in ('yes', 'y'):
            break
        elif choice in ('no', 'n', 'skip', 's'):
            print("  Skipping this configuration.\n")
            return False
        else:
            print("  Please enter 'yes', 'no', or 'skip'.")

    # --- Stage 2: Forward Translation ---
    print(f"\n--- Stage 2: Forward Translation ({fwd_model}, temp={temperature}) ---")
    translate_result = fwd_client.translate(scale_path, fwd_model, temperature=temperature)

    if not translate_result.get("success"):
        print(f"  Translation failed: {translate_result.get('error')}")
        return False

    translation = translate_result.get('translation', translate_result.get('summary', ''))
    print_translation_preview(translation)
    save_json(translation, output_path)
    print(f"  Saved forward translation to {output_path}")
    return True


def do_back_translation(back_client: RAGClient, scale_path: str,
                        back_model: str, temperature: float, output_path: str):
    back_client.reset_session()

    print(f"\n--- Back Translation ({back_model}, temp={temperature}) ---")
    result = back_client.back_translate(scale_path, back_model, temperature=temperature)

    if not result.get("success"):
        print(f"  Back translation failed: {result.get('error')}")
        return False

    translation = result.get('translation', result.get('summary', ''))
    print_translation_preview(translation)
    save_json(translation, output_path)
    print(f"  Saved back translation to {output_path}")
    return True


# ========================== Parameter Grid Test ==========================

def run_param_grid(scale_path: str, models: list = None, temperatures: list = None,
                   fwd_server_url: str = None, back_server_url: str = None,
                   output_dir: str = "results"):

    models = models or MODELS
    temperatures = temperatures or TEMPERATURES
    fwd_server_url = fwd_server_url or FORWARD_SERVER_URL
    back_server_url = back_server_url or BACKWARD_SERVER_URL
    output_dir = make_output_dir(output_dir)

    fwd_client = RAGClient(base_url=fwd_server_url)
    back_client = RAGClient(base_url=back_server_url)

    # Build rotation pairs: (a,b), (b,c), (c,d), (d,a)
    model_pairs = [(models[i], models[(i + 1) % len(models)]) for i in range(len(models))]

    # Track all results
    all_results = []
    total_configs = len(temperatures) * len(model_pairs)
    current = 0

    for temp in temperatures:
        for fwd_model, back_model in model_pairs:
            current += 1
            fwd_tag = f"{fwd_model.split('-')[0]}_t{temp}"
            back_tag = f"{back_model.split('-')[0]}_t{temp}"
            combo_tag = f"fwd_{fwd_tag}__back_{back_tag}"
            fwd_output = f"{output_dir}/fwd_{fwd_tag}.json"
            back_output = f"{output_dir}/back_{combo_tag}.json"

            print(f"\n{'#' * 60}")
            print(f"[{current}/{total_configs}] FORWARD: {fwd_model} → BACKWARD: {back_model}, temp={temp}")
            print(f"{'#' * 60}")

            # Forward translation (with familiarity check)
            fwd_ok = do_forward_translation(fwd_client, scale_path, fwd_model, temp, fwd_output)
            if not fwd_ok:
                print(f"  Skipping this configuration.\n")
                continue

            # Backward translation (no familiarity check, uses forward output as input)
            back_ok = do_back_translation(back_client, fwd_output, back_model, temp, back_output)
            if not back_ok:
                print(f"  Skipping network eval for this combination.\n")
                continue

            # Three-way network evaluation
            print(f"\n  Running three-way network evaluation...")
            try:
                comparisons = run_three_way_eval(
                    forward_json_path=fwd_output,
                    back_json_path=back_output,
                    output_dir=output_dir,
                    tag=combo_tag
                )

                all_results.append({
                    "forward_model": fwd_model,
                    "backward_model": back_model,
                    "temperature": temp,
                    "orig_vs_fwd": comparisons.get("orig_vs_fwd"),
                    "fwd_vs_back": comparisons.get("fwd_vs_back"),
                    "orig_vs_back": comparisons.get("orig_vs_back"),
                })
            except Exception as e:
                print(f"  Network eval failed: {e}")
                import traceback
                traceback.print_exc()

    # Final summary
    if all_results:
        print(f"\n{'=' * 60}")
        print("ALL RESULTS SUMMARY")
        print(f"{'=' * 60}")
        summary_rows = []
        for r in all_results:
            row = {
                "fwd_model": r["forward_model"],
                "back_model": r["backward_model"],
                "temperature": r["temperature"],
            }
            for comp_name in ["orig_vs_fwd", "fwd_vs_back", "orig_vs_back"]:
                comp = r.get(comp_name, {})
                if isinstance(comp, dict):
                    for k, v in comp.items():
                        row[f"{comp_name}_{k}"] = v
            summary_rows.append(row)

        master_df = pd.DataFrame(summary_rows)
        master_path = f"{output_dir}/master_summary.csv"
        master_df.to_csv(master_path, index=False)
        print(master_df.to_string())
        print(f"\nSaved to {master_path}")
    else:
        print("\nNo completed configurations to summarize.")

    return all_results


# ========================== Interactive Mode ==========================

def interactive_translate():
    fwd_client = RAGClient(base_url=FORWARD_SERVER_URL)
    back_client = RAGClient(base_url=BACKWARD_SERVER_URL)

    print("=" * 60)
    print("Commands:")
    print("  check <scale_path> <model>           - Stage 1: familiarity check")
    print("  translate <scale_path> <model> [temp] - Stage 2: forward translation")
    print("  back <scale_path> <model> [temp]      - Back translation")
    print("  eval <fwd.json> <back.json>           - Three-way network eval")
    print("  reset                                 - Reset session")
    print("  quit                                  - Exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("rag> ").strip()
            if not user_input:
                continue
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == 'check':
                if len(parts) < 3:
                    print("Usage: check <scale_path> <model>\n")
                    continue
                scale_path, model = parts[1], parts[2]
                result = fwd_client.check_familiarity(scale_path, model)
                if result.get("success"):
                    print(f"\n{result.get('familiarity_summary', '')}\n")
                    print(f"Session ID: {fwd_client.session_id}")
                else:
                    print(f"{result.get('error')}\n")

            elif cmd == 'translate':
                if len(parts) < 3:
                    print("Usage: translate <scale_path> <model> [temperature]\n")
                    continue
                scale_path, model = parts[1], parts[2]
                temp = float(parts[3]) if len(parts) > 3 else 0.7
                output_path = f"fwd_{model.split('-')[0]}_t{temp}.json"
                result = fwd_client.translate(scale_path, model, temperature=temp)
                if result.get("success"):
                    translation = result.get('translation', result.get('summary', ''))
                    print_translation_preview(translation)
                    save_json(translation, output_path)
                    print(f"Saved to {output_path}\n")
                else:
                    print(f"✗ {result.get('error')}\n")

            elif cmd == 'back':
                if len(parts) < 3:
                    print("Usage: back <scale_path> <model> [temperature]\n")
                    continue
                scale_path, model = parts[1], parts[2]
                temp = float(parts[3]) if len(parts) > 3 else 0.7
                output_path = f"back_{model.split('-')[0]}_t{temp}.json"
                result = back_client.back_translate(scale_path, model, temperature=temp)
                if result.get("success"):
                    translation = result.get('translation', result.get('summary', ''))
                    print_translation_preview(translation)
                    save_json(translation, output_path)
                    print(f"Saved to {output_path}\n")
                else:
                    print(f"✗ {result.get('error')}\n")

            elif cmd == 'eval':
                if len(parts) < 3:
                    print("Usage: eval <forward.json> <back.json>\n")
                    continue
                fwd_json, back_json = parts[1], parts[2]
                tag = Path(fwd_json).stem + "__" + Path(back_json).stem
                run_three_way_eval(fwd_json, back_json, "results", tag)

            elif cmd == 'reset':
                fwd_client.reset_session()
                back_client.reset_session()
                print("Sessions reset.\n")

            elif cmd in ('quit', 'exit'):
                print("Goodbye!")
                break

            else:
                print(f"Unknown command: {cmd}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


# ========================== Entry Point ==========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Translation Client")
    parser.add_argument("scale", nargs="?", default="scale.json",
                        help="Path to the scale JSON file (default: scale.json)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive REPL mode")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results (default: results)")

    args = parser.parse_args()

    if args.interactive:
        interactive_translate()
    else:
        results = run_param_grid(args.scale, output_dir=args.output_dir)
        sys.exit(0 if results else 1)