import json
import os
import sys
from openai import OpenAI

# Get API key from environment or prompt user
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("OPENAI_API_KEY not found in environment variables.")
    api_key = input("Please enter your OpenAI API key: ").strip()
    if not api_key:
        print("Error: API key is required.")
        sys.exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def load_json_file(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_tasks(domain_name, original_tasks, augmented_tasks):
    """Use GPT to analyze task quality"""

    # Sample a few tasks for comparison (to avoid token limits)
    original_sample = original_tasks[:5]
    augmented_sample = augmented_tasks[:5]

    # Also sample some later augmented tasks to check style consistency
    if len(augmented_tasks) > 10:
        later_augmented_sample = augmented_tasks[-5:]
    else:
        later_augmented_sample = augmented_tasks[5:10] if len(augmented_tasks) > 5 else []

    prompt = f"""请分析以下{domain_name}领域的任务数据质量。我需要你检查：

1. **任务设置的合理性**：每个任务的设置是否合理、逻辑是否清晰
2. **冗余性检查**：任务描述是否过于啰嗦，是否有不必要的重复信息
3. **风格一致性**：后面生成的augmented任务是否与原始tasks.json的风格保持一致

原始任务样本（tasks.json的前5个任务）：
```json
{json.dumps(original_sample, indent=2, ensure_ascii=False)}
```

增强任务样本（tasks_augmented.json的前5个任务）：
```json
{json.dumps(augmented_sample, indent=2, ensure_ascii=False)}
```

增强任务后期样本（tasks_augmented.json的后5个任务）：
```json
{json.dumps(later_augmented_sample, indent=2, ensure_ascii=False)}
```

请提供详细的分析报告，包括：
1. 任务设置的合理性评估
2. 是否存在冗余或啰嗦的问题
3. 风格一致性评估（前期vs后期的augmented任务，以及与原始任务的对比）
4. 具体的改进建议

请用中文回答。"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个专业的数据质量分析专家，擅长评估任务数据的质量、合理性和一致性。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling GPT API: {str(e)}"

def main():
    # File paths
    base_path = r"D:\Desktop\work\tau2-bench\data\tau2\domains"

    domains = [
        {
            "name": "airline",
            "original": os.path.join(base_path, "airline", "tasks.json"),
            "augmented": os.path.join(base_path, "airline", "tasks_augmented.json")
        },
        {
            "name": "retail",
            "original": os.path.join(base_path, "retail", "tasks.json"),
            "augmented": os.path.join(base_path, "retail", "tasks_augmented.json")
        }
    ]

    print("=" * 80)
    print("任务质量分析报告")
    print("=" * 80)
    print()

    for domain in domains:
        print(f"\n{'='*80}")
        print(f"分析领域: {domain['name'].upper()}")
        print(f"{'='*80}\n")

        # Load tasks
        print(f"加载文件: {domain['original']}")
        original_tasks = load_json_file(domain['original'])
        print(f"原始任务数量: {len(original_tasks)}")

        print(f"加载文件: {domain['augmented']}")
        augmented_tasks = load_json_file(domain['augmented'])
        print(f"增强任务数量: {len(augmented_tasks)}")
        print()

        # Analyze with GPT
        print("正在调用GPT API进行分析...\n")
        analysis = analyze_tasks(domain['name'], original_tasks, augmented_tasks)

        print(analysis)
        print("\n" + "="*80 + "\n")

        # Save analysis to file
        output_file = os.path.join(base_path, domain['name'], f"quality_analysis_{domain['name']}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"任务质量分析报告 - {domain['name'].upper()}\n")
            f.write("="*80 + "\n\n")
            f.write(f"原始任务数量: {len(original_tasks)}\n")
            f.write(f"增强任务数量: {len(augmented_tasks)}\n\n")
            f.write(analysis)

        print(f"分析报告已保存到: {output_file}\n")

    print("\n" + "="*80)
    print("所有分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
