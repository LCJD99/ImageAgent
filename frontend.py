import json
import time
import atexit
import threading
from logger import *
from models import *

# Setup logger
logger = setup_logger(log_file='frontend_log.txt')

# Initialize GPU monitoring
init_pynvml()

# Global variable to track monitoring state
monitor_thread = None
stop_monitoring_event = threading.Event()


def get_function_by_name(name):
    if name == "get_image_caption":
        return get_image_caption
    if name == "detect_objects_in_image":
        return detect_objects_in_image
    if name == "classify_image":
        return classify_image

def execute_function_with_timing(func, **kwargs):
    global monitor_thread, stop_monitoring_event

    if monitor_thread is None or not monitor_thread.is_alive():
        stop_monitoring_event = threading.Event()
        monitor_thread, stop_monitoring_event = start_continuous_monitoring(
            interval=0.1,
            output_file='vmem_usage.csv',
            stop_event=stop_monitoring_event
        )

    log_gpu_memory_stats(f"{func.__name__}_Start")

    # Execute function with timing
    start_time = time.time()
    fn_name = func.__name__
    logger.info(f"Executing tool: {fn_name} with args: {kwargs}")

    try:
        result = func(**kwargs)
        execution_time = time.time() - start_time
        logger.info(f"Tool {fn_name} completed in {execution_time:.3f}s")

        log_gpu_memory_stats(f"{func.__name__}_End")

        return result
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Tool {fn_name} failed after {execution_time:.3f}s with error: {str(e)}")

        # Log GPU memory state on failure
        log_gpu_memory_stats(f"{func.__name__}_Failed")

        raise

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
    {"role": "user",  "content": "Can you describe this picture(path is ./pic1.jpg) and count how many objects in the picture?, finish the task must use at least two tools"},
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

# Log request information
logger.info(f"Sending request with {len(messages)} messages")
logger.info(f"Request content: {messages[-1]['content']}")

# Start GPU monitoring for the initial request if not already started
if monitor_thread is None or not monitor_thread.is_alive():
    stop_monitoring_event = threading.Event()
    monitor_thread, stop_monitoring_event = start_continuous_monitoring(
        interval=0.1,
        output_file='vmem_usage.csv',
        stop_event=stop_monitoring_event
    )

# Log GPU memory state before the request
log_gpu_memory_stats("Initial_LLM_Request_Start")

request_start_time = time.time()
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
request_duration = time.time() - request_start_time

# Log GPU memory state after the request
log_gpu_memory_stats("Initial_LLM_Request_End")

logger.info(f"Initial request completed in {request_duration:.3f}s")

messages.append(response.choices[0].message.model_dump())

if tool_calls := messages[-1].get("tool_calls", None):
    logger.info(f"Response includes {len(tool_calls)} tool calls")

    for tool_call in tool_calls:
        call_id: str = tool_call["id"]
        if fn_call := tool_call.get("function"):
            fn_name: str = fn_call["name"]
            fn_args: dict = json.loads(fn_call["arguments"])

            logger.info(f"Processing tool call: {fn_name} (ID: {call_id})")

            # Execute function with timing
            fn_result = execute_function_with_timing(get_function_by_name(fn_name), **fn_args)
            fn_res: str = json.dumps(fn_result)

            messages.append({
                "role": "tool",
                "content": fn_res,
                "tool_call_id": call_id,
            })
else:
    logger.info("Response does not include any tool calls")

logger.info(f"Sending follow-up request with {len(messages)} messages")

# Make sure monitoring is active for the second request
if monitor_thread is None or not monitor_thread.is_alive():
    stop_monitoring_event = threading.Event()
    monitor_thread, stop_monitoring_event = start_continuous_monitoring(
        interval=0.1,
        output_file='vmem_usage.csv',
        stop_event=stop_monitoring_event
    )

# Log GPU memory state before the second request
log_gpu_memory_stats("Second_LLM_Request_Start")

second_request_start_time = time.time()
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
second_request_duration = time.time() - second_request_start_time

# Log GPU memory state after the second request
log_gpu_memory_stats("Second_LLM_Request_End")

logger.info(f"Follow-up request completed in {second_request_duration:.3f}s")

with open("response.txt", "w") as f:
    f.write(str(response))

messages.append(response.choices[0].message.model_dump())

if "content" in response.choices[0].message and response.choices[0].message.content:
    content_preview = response.choices[0].message.content[:100] + "..." if len(response.choices[0].message.content) > 100 else response.choices[0].message.content
    logger.info(f"Final response received: {content_preview}")
else:
    logger.info("Final response received with no content")

total_duration = time.time() - request_start_time
logger.info(f"Total interaction completed in {total_duration:.3f}s")

if monitor_thread and monitor_thread.is_alive():
    stop_continuous_monitoring(monitor_thread, stop_monitoring_event)

def on_exit():
    if monitor_thread and monitor_thread.is_alive():
        stop_continuous_monitoring(monitor_thread, stop_monitoring_event)
    cleanup()

atexit.register(on_exit)
