import json


def txt_to_scale_json(txt_file_path, json_file_path):

    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Extract metadata from first 3 lines
    scale_name = lines[0]
    short_name = lines[1]
    target_population = lines[2]

    # Extract items from remaining lines
    items = []
    for idx, line in enumerate(lines[3:], start=1):
        # Check if line starts with a number followed by a period
        if line and line[0].isdigit() and '.' in line:
            # Format: "1.Item text"
            number_part, text_part = line.split('.', 1)
            item_number = int(number_part.strip())
            item_text = text_part.strip()
        else:
            # No number prefix, use sequential numbering
            item_number = idx
            item_text = line.strip()

        items.append({
            "number": item_number,
            "text": item_text
        })

    # Create the final structure
    scale_data = {
        "scale_info": {
            "name": scale_name,
            "short_name": short_name,
            "target_population": target_population
        },
        "items": items
    }

    # Write to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(scale_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {txt_file_path} to {json_file_path}")
    print(f"Total items: {len(items)}")

    return scale_data


# Example usage
if __name__ == "__main__":
    # Convert single file
    txt_to_scale_json('scale.txt', 'scale_B.json')
