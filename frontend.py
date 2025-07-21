import json
from models import get_image_caption, detect_objects_in_image, classify_image


def get_function_by_name(name):
    if name == "get_image_caption":
        return get_image_caption
    if name == "detect_objects_in_image":
        return detect_objects_in_image
    if name == "classify_image":
        return classify_image

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_image_caption",
            "description": "Generate a caption for an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file for which to generate a caption.",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_objects_in_image",
            "description": "Detect objects in an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file for object detection.",
                    },
                    "threshold": {
                        "type": "string",
                        "description": "Confidence threshold for detections (e.g., '0.9').",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_image",
            "description": "Classify an image into predefined categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file for classification.",
                    },
                    "top_k": {
                        "type": "string",
                        "description": "Number of top predictions to return (e.g., '5').",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
]
MESSAGES = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"},
    {"role": "user",  "content": "Can you describe this picture(path is ./pic1.jpg) and count how many objects in the picture?"},
]

tools = TOOLS
messages = MESSAGES[:]


from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = "./qwen2.5"

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)

messages.append(response.choices[0].message.model_dump())

if tool_calls := messages[-1].get("tool_calls", None):
    for tool_call in tool_calls:
        call_id: str = tool_call["id"]
        if fn_call := tool_call.get("function"):
            fn_name: str = fn_call["name"]
            fn_args: dict = json.loads(fn_call["arguments"])

            fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

            messages.append({
                "role": "tool",
                "content": fn_res,
                "tool_call_id": call_id,
            })

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)

messages.append(response.choices[0].message.model_dump())

print(messages)
