from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

qwen_api_key = os.getenv("QWEN_API_KEY")


try:
    client = OpenAI(
        # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
        api_key=qwen_api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        model="qwen2.5-7b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate a matrix multiplication kernel in CUDA."},
        ],
        logprobs=True,
        top_logprobs=5,
    )
    print(completion)

    print("-" * 100)
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error message: {e}")
    print(
        "Please refer to the documentation: https://www.alibabacloud.com/help/en/model-studio/developer-reference/error-code"
    )
