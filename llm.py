import json
import base64
from typing import Dict, Optional, Tuple, Any

from openai import OpenAI

api_price = {
    "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
    "gpt-4o": {"input": 0.0000025, "output": 0.00001},
}

gpt_client = OpenAI()


def query(
        user_msg_txt: str,
        user_msg_img: Optional[str] = None,
        system_msg: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
) -> Tuple[Dict[str, Any], int, int, float, float]:
    if user_msg_img is not None:
        user_msg_img = encode_image(user_msg_img)
        user_msg_img = f"data:image/jpeg;base64,{user_msg_img}"

    messages = [{"role": "system", "content": system_msg}] if system_msg else []
    user_content = user_msg_txt if user_msg_img is None else [
        {"type": "text", "text": user_msg_txt},
        {"type": "image_url", "image_url": {"url": user_msg_img, "detail": "high"}}
    ]
    messages.append({"role": "user", "content": user_content})
    response = gpt_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    response_content = response.choices[0].message.content
    response_dict = json.loads(response_content)
    output_usage = response.usage.completion_tokens
    output_cost = output_usage * api_price[model]["output"]
    input_usage = response.usage.prompt_tokens
    input_cost = input_usage * api_price[model]["input"]
    return response_dict, input_usage, output_usage, input_cost, output_cost

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    r, _, _, _, _ = query("hello")
    print(r)
