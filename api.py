from openai import OpenAI
import json
import time

client = OpenAI(
    api_key="xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def judge_one(history, sasrec, bert4rec, llm4rec):
    prompt = build_judge_prompt(history, sasrec, bert4rec, llm4rec)

    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a professional recommender system evaluator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0
    )

    text = resp.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except:
        print("Bad response:", text)
        return {"sasrec": None, "bert4rec": None, "llm4rec": None}
