import json

input_file = "/home/ubuntu/jgy-dataset/valid_life_problems_for_finetune(1).jsonl"
output_file = "/home/ubuntu/jgy-dataset/valid_life_problems_for_finetune_converted.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for idx, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)

        instruction = data.get("instruction", "")
        input_text = data.get("input", "")
        output_text = data.get("output", "")

        # Combine instruction and input into user content
        if instruction and input_text:
            user_content = f"{instruction}\n{input_text}"
        elif instruction:
            user_content = instruction
        else:
            user_content = input_text

        converted = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output_text}
            ],
            "id": f"life_{idx:05d}"
        }

        fout.write(json.dumps(converted, ensure_ascii=False) + "\n")

print(f"Converted {idx} records to {output_file}")
