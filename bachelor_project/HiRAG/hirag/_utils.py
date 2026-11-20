import asyncio
import contextvars
import html
import json
import logging
import numbers
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any

import numpy as np
import tiktoken

logger = logging.getLogger("HiRAG")
ENCODER = None

# Task-local name of the HiRAG stage whose LLM call is currently running
# (``entity_extraction``, ``answer_generation``, ...). Set by the wrapper
# installed in ``get_stage_llm_func`` around every resolved stage function;
# ``None`` outside stage calls. Run-logging adapters (e.g. hi_insert_run.py)
# read it to attribute each logged call to its pipeline stage. Contextvars
# are asyncio-task-local, so concurrent chunk processing cannot mix stages.
hirag_current_stage: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hirag_current_stage", default=None
)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def extract_first_complete_json(s: str):
    """Extract the first complete JSON object from the string using a stack to track braces."""
    stack = []
    first_json_start = None

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == "}":
            if stack:
                start = stack.pop()  # noqa: F841
                if not stack:
                    first_json_str = s[first_json_start : i + 1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}..."
                        )
                        return None
                    finally:
                        first_json_start = None
    logger.warning("No complete JSON object found in the input string.")
    return None


def parse_value(value: str):
    """Convert a string value to its appropriate type (int, float, bool, None, or keep as string). Work as a more broad 'eval()'"""
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if "." in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist


def extract_values_from_json(
    json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False
):
    """Extract key values from a non-standard or malformed JSON string, handling nested objects."""
    extracted_values = {}

    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group("key").strip('"')  # Strip quotes from key
        value = match.group("value").strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith("{") and value.endswith("}"):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")

    return extracted_values


def convert_response_to_json(response: str) -> dict:
    """Convert response string to JSON, with error handling and fallback to non-standard JSON extraction."""
    prediction_json = extract_first_complete_json(response)

    if prediction_json is None:
        logger.info("Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)

    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response.")
    else:
        logger.info("JSON data successfully extracted.")

    return prediction_json


def _get_tiktoken_encoder(model_or_encoding_name: str):
    """Return a tiktoken encoder, accepting either a model name (e.g. 'gpt-4o')
    or an encoding name (e.g. 'cl100k_base').

    tiktoken.encoding_for_model() only accepts model names, but HiRAG's default
    tiktoken_model_name is the encoding name 'cl100k_base'. Fall back to
    get_encoding() when the input is an encoding name.
    """
    try:
        return tiktoken.encoding_for_model(model_or_encoding_name)
    except KeyError:
        return tiktoken.get_encoding(model_or_encoding_name)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = _get_tiktoken_encoder(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = _get_tiktoken_encoder(model_name)
    content = ENCODER.decode(tokens)
    return content


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def write_json(json_obj: dict, file_name: str) -> None:
    """Write ``json_obj`` to ``file_name`` atomically.

    Serializes to a temporary file in the same directory, fsyncs it, and then
    swaps it over the target via ``os.replace``. A crash or kill mid-write can
    therefore never leave a torn JSON file behind: the target path always
    points to either the complete previous version or the complete new one.
    """
    tmp_file_name = f"{file_name}.tmp"
    with open(tmp_file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file_name, file_name)


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def get_stage_llm_func(
    global_config: dict, stage: str, fallback_key: str = "best_model_func"
):
    """Resolve the LLM function for a HiRAG stage.

    Args:
        global_config: HiRAG global configuration dict (``asdict(hirag)``).
        stage: Stage name, e.g. ``entity_extraction`` or ``answer_generation``.
        fallback_key: ``global_config`` key of the original model function to
            use when stage routing is disabled or the stage is not configured
            (``best_model_func``, or ``cheap_model_func`` for description
            summarization).

    Returns:
        The stage-specific model function if one was configured and built,
        otherwise the original fallback model function. Either way the
        returned function is wrapped so that ``hirag_current_stage`` names
        the calling stage for the duration of each call (transparent to
        callers that never read the context variable).
    """
    stage_funcs = global_config.get("stage_model_funcs") or {}
    stage_func = stage_funcs.get(stage)
    resolved = stage_func if stage_func is not None else global_config[fallback_key]

    @wraps(resolved)
    async def stage_tracking_func(*args: Any, **kwargs: Any) -> Any:
        """Call the resolved stage function with the stage name in context."""
        token = hirag_current_stage.set(stage)
        try:
            return await resolved(*args, **kwargs)
        finally:
            hirag_current_stage.reset(token)

    return stage_tracking_func


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


# Utils types -----------------------------------------------------------------------
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


# Decorators ------------------------------------------------------------------------
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro
