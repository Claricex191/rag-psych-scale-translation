import json
import sys


def convert(scale_path: str, expert_path: str, output_path: str):
    # Load original English items
    with open(scale_path, 'r', encoding='utf-8') as f:
        scale = json.load(f)

    originals = scale['items']

    # Load expert translations (one per line, skip blank lines)
    with open(expert_path, 'r', encoding='utf-8') as f:
        expert_lines = [line.strip() for line in f if line.strip()]

    if len(expert_lines) != len(originals):
        print(f"WARNING: {len(expert_lines)} expert lines vs {len(originals)} scale items")

    items = []
    for i, orig in enumerate(originals):
        items.append({
            "number": orig['number'],
            "original": orig['text'],
            "translation": expert_lines[i] if i < len(expert_lines) else "",
            "log": "Expert human translation"
        })

    output = {"items": items}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Created {output_path} with {len(items)} items")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python make_expert_json.py <scale.json> <expert.txt> <output.json>")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2], sys.argv[3])