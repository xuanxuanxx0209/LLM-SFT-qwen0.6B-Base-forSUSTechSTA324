
import json
input_path = "/home/ubuntu/datasets/valid_1000.jsonl"
output_path = "/home/ubuntu/datasets/sft_data_boxed.jsonl"
count = 0
with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        data = json.loads(line.strip())
        solution = data["solution"]
        answer = str(data["answer"])
        # 直接替换最后面的#### 答案
        idx = solution.rfind("#### ")
        if idx != -1:
            solution = solution[:idx] + f"#### \boxed{{{answer}}}"
        
        sft_data = {
            "messages": [
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": solution}
            ]
        }
        f_out.write(json.dumps(sft_data, ensure_ascii=False) + "\n")
        count +=1

print(f"Processed {count} entries")
# 验证
with open(output_path, "r") as f:
    line = f.readline()
    data = json.loads(line)
    print("Example answer line:", data["messages"][1]["content"].split("\n")[-1])

