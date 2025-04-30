"""
title: Omni Router Manifold Pipeline
author: Moeakwak
date: 2025-04-30
version: 0.4.0
license: MIT
description: A pipeline for routing OpenAI models, track user usages, etc.
requirements: tabulate
"""

import json
import re
from typing import Literal, Optional, Union, Generator, Iterator
import os
from pydantic import BaseModel
import pytz
import requests
from sqlalchemy import Engine, create_engine, desc, distinct
from sqlmodel import Field, SQLModel, Session, select
from datetime import datetime, timedelta
from tabulate import tabulate
import yaml
from pathlib import Path
from sqlalchemy import func
from typing import Optional
import tiktoken
import argparse
from enum import Enum, auto
import time
import threading
from http import HTTPStatus
import copy

# Schemas


class Model(BaseModel):
    provider: str
    code: str
    extra_identifier: Optional[str] = Field(default=None, description="Extra identifier for the model. Useful for creating multiple instances of the same model.")
    human_name: Optional[str] = Field(default=None)
    prompt_price: Optional[float] = Field(default=None, description="The prompt price of the model per 1M tokens.")
    completion_price: Optional[float] = Field(default=None, description="The completion price of the model per 1M tokens.")
    per_message_price: Optional[float] = Field(default=None, description="The price of the model per message.")
    disable_cost_display_in_completion: Optional[bool] = Field(
        default=False, description="If true, disable showing the cost in the completion. Non-stream completions always don't show the cost."
    )
    no_system_prompt: Optional[bool] = Field(default=False, description="If true, remove the system prompt.")
    update_usage_via_openrouter_api: Optional[bool] = Field(
        default=False, description="If true, fetch usage from the /generation endpoint, for OpenRouter models only."
    )
    fallback_compute_usage: Optional[bool] = Field(
        default=True, description="If true, estimate usage using tiktoken when no usage is found in the response."
    )
    include_reasoning: Optional[bool] = Field(default=False, description="If true, add include_reasoning=true to the request payload.")
    extra_args: Optional[dict] = Field(default=None, description="Extra arguments to pass to the request payload. Will override the original values.")


class Provider(BaseModel):
    key: str
    format: Optional[Literal["openai"]] = Field(default="openai")
    url: str
    api_key: str
    price_ratio: Optional[float] = Field(
        default=1,
        description="The price ratio of the provider. For example, if the price ratio is 0.5, then the actual cost of the provider is half of the original cost.",
    )


class Config(BaseModel):
    providers: list[Provider]
    models: list[Model]


class OpenAICompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# DB


class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    openwebui_id: str = Field(index=True, unique=True)
    email: Optional[str] = Field(index=True, unique=True, default=None)
    name: Optional[str] = Field(default=None)
    balance: float = Field(default=0, description="The balance of the user in USD. May be negative if the user has an outstanding debt.")
    role: str = Field(default="user")
    created_at: datetime = Field(default_factory=datetime.now, index=True)
    updated_at: datetime = Field(default_factory=datetime.now, index=True)


class UsageLog(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    model: str = Field(index=True)
    provider: str = Field(index=True)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    is_stream: bool = Field(default=True, description="If true, the usage is a stream.")
    is_title_generation: bool = Field(default=False, index=True, description="If true, the usage is a title generation.")
    cost: float = Field(description="Computed cost of the usage.")
    actual_cost: float = Field(description="Actual cost of the usage, after applying the price ratio of the provider.")
    content: Optional[str] = Field(default=None, description="The content of the prompt. Only applicable if RECORD_CONTENT is true.")
    created_at: datetime = Field(default_factory=datetime.now, index=True)


def escape_model_code(code: str):
    return code.replace("/", "__").replace(":", "___")


def mask_message_content(body_or_payload: dict, shorten_log: bool = False):
    result = copy.deepcopy(body_or_payload)
    if "messages" in result:
        for message in result["messages"]:
            if "content" in message:
                if shorten_log:
                    message["content"] = message["content"][:50]
                    if len(message["content"]) > 50:
                        message["content"] += "......"
    result = json.dumps(result, indent=2, ensure_ascii=False)
    return result


def print_log(message: str, model: Optional[Model] = None, user: Optional[User] = None):
    prefix = f"Router Pipeline | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if model:
        prefix += f" [{model.code} {model.provider}]"
    if user:
        prefix += f" ({user.name})"
    print(f"{prefix} | {message}")


def print_and_raise(message: str):
    print(f"***** ERROR | {message}")
    raise Exception(message)


class Pipeline:
    class Valves(BaseModel):
        MODELS_CONFIG_YAML_PATH: str = "/app/pipelines/omni_router.yaml"
        DATABASE_URL: str = "sqlite:////app/pipelines/omni_router.db"
        ENABLE_BILLING: bool = True
        RECORD_CONTENT: int = Field(
            default=30, description="Record the first N characters of the content. Set to 0 to disable recording content, -1 to record all content."
        )
        DEFAULT_USER_BALANCE: float = Field(default=10, description="Default balance of the user in USD.")
        BASE_COST_CURRENCY_UNIT: str = Field(default="$", description="The currency unit of the base cost.")
        ACTUAL_COST_CURRENCY_UNIT: str = Field(
            default="$", description="The currency unit of the actual cost, also the currency unit of the user balance."
        )
        TIMEZONE: str = Field(default="UTC", description="The timezone of the server.")
        AUXILIARY_MODEL_CODE: str = Field(default="gpt-4o-mini", description="The model to use for title generation.")
        REQUEST_TIMEOUT: int = Field(default=300, description="The timeout of the request connection in seconds.")
        DEBUG_MODE: bool = Field(default=True, description="If true, show debug logs.")
        DEBUG_SHORTEN_LOG: bool = Field(default=False, description="If true, shorten the log.")

    class ReasoningState(Enum):
        """Enum for tracking the state of reasoning in stream response"""

        NOT_STARTED = auto()  # Initial state
        IN_PROGRESS = auto()  # Currently processing reasoning content
        COMPLETED = auto()  # Finished reasoning, processing normal content

    def print_debug(self, message: str):
        if self.valves.DEBUG_MODE:
            print(f"***** DEBUG | {message}")

    def __init__(self):
        self.type = "manifold"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        self.id = "omni_router"
        self.name = "💬 "

        self.valves = self.Valves(
            **{
                "MODELS_CONFIG_YAML_PATH": os.getenv("MODELS_CONFIG_YAML_PATH", "/app/pipelines/omni_router.yaml"),
                "ENABLE_BILLING": True if os.getenv("ENABLE_BILLING", "true").lower() == "true" else False,
                "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:////app/pipelines/omni_router.db"),
                "RECORD_CONTENT": int(os.getenv("RECORD_CONTENT", 30)),
                "DEFAULT_USER_BALANCE": float(os.getenv("DEFAULT_USER_BALANCE", 10)),
                "BASE_COST_CURRENCY_UNIT": os.getenv("BASE_COST_CURRENCY_UNIT", "$"),
                "ACTUAL_COST_CURRENCY_UNIT": os.getenv("ACTUAL_COST_CURRENCY_UNIT", "$"),
                "REQUEST_TIMEOUT": int(os.getenv("REQUEST_TIMEOUT", 300)),
                "TIMEZONE": os.getenv("TIMEZONE", "UTC"),
                "AUXILIARY_MODEL_CODE": os.getenv("AUXILIARY_MODEL_CODE", "gpt-4o-mini"),
                "DEBUG_MODE": False if os.getenv("DEBUG_MODE", "false").lower() == "false" else True,
                "DEBUG_SHORTEN_LOG": False if os.getenv("DEBUG_SHORTEN_LOG", "true").lower() == "false" else True,
            }
        )
        self.config = self.load_config()
        self.models: dict[str, Model] = self.load_models()
        self.pipelines = self.get_pipelines()
        self.engine = self.setup_db()

    def setup_db(self) -> Engine:
        engine = create_engine(self.valves.DATABASE_URL)
        SQLModel.metadata.create_all(engine)
        return engine

    # async def on_startup(self):
    #     # This function is called when the server is started.
    #     print(f"on_startup:{__name__}")
    #     pass

    # async def on_shutdown(self):
    #     # This function is called when the server is stopped.
    #     print(f"on_shutdown:{__name__}")
    #     pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        # print(f"on_valves_updated:{__name__}")
        self.config = self.load_config()
        self.models = self.load_models()
        self.pipelines = self.get_pipelines()
        self.engine = self.setup_db()

    def load_config(self):
        try:
            path = Path(self.valves.MODELS_CONFIG_YAML_PATH)
            print_log(f"Loading config from {path.absolute()}")
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")

            with open(path, "r") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            return Config.model_validate(data)
        except Exception as e:
            print_log(f"Error loading config: {e}")
            return Config(providers=[])

    def load_models(self) -> dict[str, Model]:
        models = {}
        provider_keys = {p.key for p in self.config.providers}

        if not self.config.models:
            print_log("No models found in config.yaml")
            return models

        for model in self.config.models:
            if model.provider not in provider_keys:
                print_log(f"Provider {model.provider} not found in config.yaml")
                continue
            has_prompt_price = model.prompt_price is not None
            has_completion_price = model.completion_price is not None
            if has_prompt_price != has_completion_price:
                print_and_raise(f"Model {model.code} must have both prompt price and completion price set, or neither")
            elif not has_prompt_price and model.per_message_price is None:
                print_and_raise(f"Model {model.code} must have either (prompt price + completion price) or per message price set")
            model_id = f"{model.provider}_{escape_model_code(model.code)}"
            if model.extra_identifier:
                model_id += f"_{model.extra_identifier}"
            models[model_id] = model

        return models

    def get_pipelines(self) -> list[dict]:
        pipelines = [{"id": model_id, "name": model.human_name or model.code} for model_id, model in self.models.items()]
        pipelines.insert(0, {"id": "service_bot", "name": "🤖 Service Bot"})
        return pipelines

    def get_model_and_provider_by_id(self, model_id: str) -> tuple[Model, Provider]:
        model = self.models.get(model_id)
        if not model:
            print_and_raise(f"Model {model_id} not found")

        provider = next((p for p in self.config.providers if p.key == model.provider), None)
        if not provider:
            print_and_raise(f"Provider {model.provider} not found in config.yaml")

        return model, provider

    def remove_usage_cost_in_messages(self, messages: list[dict]) -> list[dict]:
        for message in messages:
            if "content" in message:
                if isinstance(message["content"], str):
                    message["content"] = re.sub(r"\n*\*\(📊[^)]*\)\*$", "", message["content"], flags=re.MULTILINE)
                elif isinstance(message["content"], list):
                    for content in message["content"]:
                        if isinstance(content, dict) and "text" in content:
                            content["text"] = re.sub(r"\n*\*\(📊[^)]*\)\*$", "", content["text"], flags=re.MULTILINE)
            else:
                self.print_debug(f"Unknown message type: {message}")
        return messages

    def remove_think_tag_in_messages(self, messages: list[dict]) -> list[dict]:
        """Remove think tags and their content from messages."""
        for message in messages:
            if "content" in message:
                if isinstance(message["content"], str):
                    # 移除 think 标签及其内容
                    message["content"] = re.sub(r"<think>\n+.*?\n+</think>\n+", "", message["content"], flags=re.DOTALL).strip()
                elif isinstance(message["content"], list):
                    # 处理多模态消息
                    for content in message["content"]:
                        if isinstance(content, dict) and content.get("type") == "text":
                            content["text"] = re.sub(r"<think>\n+.*?\n+</think>\n+", "", content["text"], flags=re.DOTALL).strip()
        return messages

    def generate_usage_cost_message(
        self, usage: OpenAICompletionUsage, base_cost: float, actual_cost: float, user: User, is_estimate: bool = False, update_later: bool = False
    ) -> str:
        if base_cost == 0 and actual_cost == 0:
            return "\n\n*(📊 This is a free message)*"
        message = f"\n\n*(📊{'Estimated' if is_estimate else ''} Cost {self.valves.ACTUAL_COST_CURRENCY_UNIT}{actual_cost:.6f} | {self.valves.BASE_COST_CURRENCY_UNIT}{base_cost:.6f}"
        if update_later:
            message += ", actual cost will update later"
        return message + ")*"

    def compute_price(self, model: Model, provider: Provider, usage: Optional[OpenAICompletionUsage] = None) -> tuple[float, float]:
        if usage and model.prompt_price and model.completion_price:
            base_cost = (usage.prompt_tokens * model.prompt_price + usage.completion_tokens * model.completion_price) / 1e6
            actual_cost = base_cost * provider.price_ratio
            return base_cost, actual_cost
        elif model.per_message_price:
            return model.per_message_price, model.per_message_price * provider.price_ratio
        else:
            print_log(f"Warning: Model has no pricing information", model)
            return 0, 0

    def get_user_info(self, user: dict) -> User:
        if not "email" in user or not "id" in user or not "role" in user:
            raise Exception("User info not found")

        with Session(self.engine) as session:
            db_user = session.exec(select(User).where(User.email == user["email"])).first()
            if not db_user:
                db_user = User(
                    name=user.get("name"), email=user["email"], openwebui_id=user["id"], role=user["role"], balance=self.valves.DEFAULT_USER_BALANCE
                )
                session.add(db_user)
                session.commit()
            if db_user.name != user.get("name"):
                db_user.name = user.get("name")
                session.add(db_user)
                session.commit()
            if self.valves.ENABLE_BILLING and db_user.balance <= 0 and db_user.role != "admin":
                raise Exception("Your account has insufficient balance. Please top up your account.")
        return db_user

    def pipe(self, user_message: str, model_id: str, messages: list[dict], body: dict) -> Union[str, Generator, Iterator]:
        user_info = body.pop("user")
        self.print_debug(f"PRINT BODY: {mask_message_content(body, self.valves.DEBUG_SHORTEN_LOG)}")

        try:
            user = self.get_user_info(user_info)
        except Exception as e:
            print_log(f"Error getting user info: {e}")
            return "Error: " + str(e)

        if model_id == "service_bot":
            return self.handle_bot(user_message, user)

        try:
            model, provider = self.get_model_and_provider_by_id(model_id)
            headers = {}
            headers["Authorization"] = f"Bearer {provider.api_key}"
            headers["Content-Type"] = "application/json"

            payload = {
                **body,
                "messages": self.remove_think_tag_in_messages(self.remove_usage_cost_in_messages(messages)),
                "model": model.code,
                "stream_options": {"include_usage": True},
            }

            if model.include_reasoning:
                payload["include_reasoning"] = True

            if model.extra_args:
                payload = {**payload, **model.extra_args}

            display_usage_cost = body.get("display_usage_cost")
            args = {
                "is_title_generation": "title" in user_message and model.code == self.valves.AUXILIARY_MODEL_CODE,
                "is_stream": body.get("stream"),
            }
            if display_usage_cost is not None:
                args["display_usage_cost"] = display_usage_cost

            if model.no_system_prompt:
                payload["messages"] = [message for message in messages if message["role"] != "system"]

            if "user" in payload:
                del payload["user"]
            if "user_info" in payload:
                del payload["user_info"]
            if "chat_id" in payload:
                del payload["chat_id"]
            if "title" in payload:
                del payload["title"]

            self.print_debug(f"PAYLOAD: {mask_message_content(payload, self.valves.DEBUG_SHORTEN_LOG)}")

            try:
                r = requests.post(url=f"{provider.url}/chat/completions", json=payload, headers=headers, stream=True, timeout=self.valves.REQUEST_TIMEOUT)

                # 检查响应状态码
                if r.status_code != 200:
                    error_msg = r.text
                    error_msg = f"Error: {r.status_code} {HTTPStatus(r.status_code).phrase} - {error_msg}"
                    print_log(error_msg, model, user)
                    return error_msg

                if body["stream"]:
                    return self.stream_response(r, model, provider, user, messages, user_message, **args)
                else:
                    return self.non_stream_response(r, model, provider, user, messages, user_message, **args)

            except Exception as e:
                error_msg = f"Error: An unexpected error occurred: {str(e)}"
                print_log(error_msg, model, user)
                return "Error: " + error_msg

        except Exception as e:
            error_msg = f"Error: Unknown error occurred - {str(e)}"
            print_log(error_msg, model, user)
            return "Error: " + error_msg

    def fetch_usage_by_api(self, message_id: str, provider: Provider) -> tuple[OpenAICompletionUsage | None, float, dict]:
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        try:
            generation_response = requests.get(f"{provider.url}/generation", params={"id": message_id}, headers=headers)
            if generation_response.status_code == 200:
                data = generation_response.json().get("data", {})
                prompt_tokens = data.get("native_tokens_prompt", 0)
                completion_tokens = data.get("native_tokens_completion", 0)
                total_tokens = prompt_tokens + completion_tokens
                total_cost = data.get("total_cost", 0)
                usage = OpenAICompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
                return usage, total_cost, data
            else:
                return None, 0, {}
        except Exception as e:
            print_log(f"Error fetching usage by API: {e}")
            return None, 0, {}

    def update_usage_in_background(self, message_id: str, usage_log_id: int, model: Model, provider: Provider, user: User):
        """后台任务：尝试获取实际使用量并更新数据库记录"""
        retry_delays = [1, 3, 5, 10]  # 重试延迟时间（秒）

        for i, delay in enumerate(retry_delays):
            time.sleep(delay)
            usage, total_cost, data = self.fetch_usage_by_api(message_id, provider)

            if usage:
                with Session(self.engine) as session:
                    try:
                        log = session.get(UsageLog, usage_log_id)
                        if log:
                            # 计算新的成本
                            actual_cost = total_cost * provider.price_ratio

                            original_cost = log.cost
                            original_actual_cost = log.actual_cost

                            # 更新记录
                            log.prompt_tokens = usage.prompt_tokens
                            log.completion_tokens = usage.completion_tokens
                            log.total_tokens = usage.total_tokens
                            log.cost = total_cost
                            log.actual_cost = actual_cost

                            # 更新用户余额（需要计算差额）
                            user = session.get(User, log.user_id)
                            if user:
                                cost_difference = actual_cost - log.actual_cost
                                user.balance -= cost_difference
                                user.updated_at = datetime.now()

                            session.commit()
                            provider_name = data.get("provider_name") or ""
                            print_log(
                                f"Successfully updated usage log {usage_log_id} with actual usage at {i+1} retries. Cost: {original_cost} -> {total_cost}, Actual cost: {original_actual_cost} -> {actual_cost}, OpenRouter Provider: {provider_name}",
                                model,
                            )
                            return
                    except Exception as e:
                        print_log(f"Error updating usage log {usage_log_id}: {e}")
                        session.rollback()
            else:
                print_log(f"Failed to fetch actual usage for log {usage_log_id}, user {user.name} at {i+1} retries, keeping estimated values")

        print_log(f"Failed to get actual usage for log {usage_log_id} after all retries, keeping estimated values")

    def compute_usage_by_tiktoken(self, model: Model, messages: list[dict], prompt: str, completion: str) -> OpenAICompletionUsage | None:
        try:
            if model.code.startswith("gpt-4o"):
                encoding = tiktoken.encoding_for_model("gpt-4o")
            elif model.code.startswith("gpt-4"):
                encoding = tiktoken.encoding_for_model("gpt-4")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            # Count tokens in messages
            messages_tokens = 0
            for message in messages:
                # Handle different content formats according to OpenAI's format
                content = message.get("content")
                if isinstance(content, str):
                    messages_tokens += len(encoding.encode(content))
                elif isinstance(content, list):
                    # For multimodal messages, only count text content
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            messages_tokens += len(encoding.encode(item.get("text", "")))
                        elif isinstance(item, str):
                            messages_tokens += len(encoding.encode(item))

            prompt_tokens = len(encoding.encode(prompt)) + messages_tokens
            completion_tokens = len(encoding.encode(completion))
            total_tokens = prompt_tokens + completion_tokens
            print_log(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}", model)
            return OpenAICompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
        except Exception as e:
            print_log(f"Error computing usage by tiktoken: {e}", model)
            return None

    def stream_response(
        self,
        r: requests.Response,
        model: Model,
        provider: Provider,
        user: User,
        messages: list[dict],
        user_message: str,
        is_title_generation: bool = False,
        is_stream: bool = True,
        display_usage_cost: bool = True,
    ):
        def generate():
            content = ""
            reasoning_content = ""
            usage = None
            last_chunk: dict | None = None
            stop_chunk: dict | None = None
            usage_chunk: dict | None = None
            message_id: str | None = None

            # Initialize state machine using Enum
            reasoning_state = self.ReasoningState.NOT_STARTED

            for line in r.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")
                if line.startswith(":"):
                    continue
                line = line.strip("data: ")

                if line == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    print_log(f"Error decoding JSON: {line}", model.code, user.name)
                    continue

                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content_delta = delta.get("content") or ""
                    reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    content += content_delta
                    reasoning_content += reasoning_delta

                    # prevent displaying stop chunk
                    if chunk.get("choices", [{}])[0].get("finish_reason") == "stop":
                        stop_chunk = chunk
                    else:
                        last_chunk = chunk
                        message_id = chunk.get("id") if message_id is None else message_id
                        yield "data: " + json.dumps(chunk) + "\n\n"
                elif "usage" in chunk and chunk["usage"] is not None:
                    try:
                        usage = OpenAICompletionUsage(**chunk["usage"])
                        usage_chunk = chunk
                    except TypeError:
                        pass

            if reasoning_content:
                print_log(f"reasoning_content length: {len(reasoning_content)}", model, user)

            is_estimate = False
            update_later = False
            estimated_usage = None
            if usage is None or model.update_usage_via_openrouter_api:
                estimated_usage = self.compute_usage_by_tiktoken(model, messages, user_message, reasoning_content + content)
                is_estimate = usage is None
                usage = usage or estimated_usage

            base_cost, actual_cost = self.compute_price(model, provider, usage)

            usage_log_id = self.add_usage_log(
                user.id, model, usage, base_cost, actual_cost, content=user_message, is_stream=is_stream, is_title_generation=is_title_generation
            )

            if model.update_usage_via_openrouter_api and message_id:
                update_later = True

                thread = threading.Thread(target=self.update_usage_in_background, args=(message_id, usage_log_id, model, provider, user))
                thread.daemon = True
                thread.start()

            if not is_title_generation and actual_cost > 0.001:
                print_log(
                    f"STREAM USAGE: {usage}, base_cost: {base_cost}, actual_cost: {actual_cost}, is_estimate: {is_estimate}, update_later: {update_later}",
                    model,
                    user,
                )

            if not model.disable_cost_display_in_completion and display_usage_cost and not is_title_generation:
                if last_chunk:
                    new_chunk = last_chunk.copy()
                    new_chunk["choices"][0]["delta"]["content"] = self.generate_usage_cost_message(
                        usage, base_cost, actual_cost, user, is_estimate, update_later
                    )
                else:
                    new_chunk = {
                        "choices": [
                            {"delta": {"content": self.generate_usage_cost_message(usage, base_cost, actual_cost, user, is_estimate, update_later)}}
                        ],
                        "finish_reason": "stop",
                    }
                yield "data: " + json.dumps(new_chunk) + "\n\n"
            if stop_chunk:
                yield "data: " + json.dumps(stop_chunk) + "\n\n"
                stop_chunk = None
            if usage_chunk:
                yield "data: " + json.dumps(usage_chunk) + "\n\n"
                usage_chunk = None
            yield "data: [DONE]"

        return generate()

    def non_stream_response(
        self,
        r: requests.Response,
        model: Model,
        provider: Provider,
        user: User,
        messages: list[dict],
        user_message: str,
        is_title_generation: bool = False,
        display_usage_cost: bool = False,
        is_stream: bool = True,
    ):
        response = r.json()
        usage = OpenAICompletionUsage(**response["usage"]) if "usage" in response else None

        # 处理 reasoning content
        message = response["choices"][0]["message"]
        content = message["content"]
        reasoning = message.get("reasoning_content") or message.get("reasoning")

        if reasoning:
            content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            message["content"] = content

        message_id = response.get("id")

        estimated_usage = None
        update_later = False
        if usage is None or model.update_usage_via_openrouter_api:
            estimated_usage = self.compute_usage_by_tiktoken(model, messages, user_message, content)
            usage = usage or estimated_usage

        base_cost, actual_cost = self.compute_price(model, provider, usage)
        usage_log_id = self.add_usage_log(
            user.id, model, usage, base_cost, actual_cost, content=user_message, is_stream=is_stream, is_title_generation=is_title_generation
        )

        if model.update_usage_via_openrouter_api:
            if message_id:
                thread = threading.Thread(target=self.update_usage_in_background, args=(message_id, usage_log_id, model, provider, user))
                thread.daemon = True
                thread.start()
                update_later = True
            else:
                print_log("Warning: update_usage_via_openrouter_api - No message id found, cannot update usage", model, user)

        is_estimate = estimated_usage is not None
        if not is_title_generation and actual_cost > 0.001:
            print_log(
                f"NON-STREAM USAGE: {usage}, base_cost: {base_cost}, actual_cost: {actual_cost}, is_estimate: {is_estimate}, update_later: {update_later}",
                model,
                user,
            )
        # non stream response should not show cost in completion
        # if model.show_cost_in_completion and display_usage_cost and not is_title_generation:
        #     message["content"] += self.generate_usage_cost_message(usage, base_cost, actual_cost, user, usage is None, model.fetch_usage_by_api)

        return response

    def add_usage_log(
        self,
        user_id: int,
        model: Model,
        usage: Optional[OpenAICompletionUsage],
        base_cost: float,
        actual_cost: float,
        content: str,
        is_stream: bool = True,
        is_title_generation: bool = False,
    ) -> int:
        with Session(self.engine) as session:
            try:
                user = session.exec(select(User).where(User.id == user_id)).first()
                if not user:
                    raise Exception("User not found")

                if self.valves.RECORD_CONTENT > 0:
                    short_content = content[: self.valves.RECORD_CONTENT]
                    if len(content) > len(short_content):
                        short_content += "..."
                    content = short_content
                elif self.valves.RECORD_CONTENT == 0:
                    content = None

                usage_log = UsageLog(
                    user_id=user.id,
                    model=model.code,
                    provider=model.provider,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    cost=base_cost,
                    actual_cost=actual_cost,
                    content=content,
                    is_stream=is_stream,
                    is_title_generation=is_title_generation,
                )
                user.balance -= actual_cost
                user.updated_at = datetime.now()

                session.add(usage_log)
                session.commit()

                return usage_log.id
            except Exception as e:
                print_log(f"An error occurred: {str(e)}", model, user)
                session.rollback()
                raise

    def handle_bot(self, message: str, user: User):
        bot = ServiceBot(self)
        return bot.handle_command(message, user)


def add_user_parsers(subparsers: argparse._SubParsersAction) -> dict[str, argparse.ArgumentParser]:
    # Stats command
    stats_parser: argparse.ArgumentParser = subparsers.add_parser("stats", help="Show usage statistics")
    stats_parser.add_argument("-p", "--period", choices=["d", "w", "m"], default="d", help="Period (d: daily, w: weekly, m: monthly)")
    stats_parser.add_argument("--provider", type=str, default=None, help="Filter by provider")

    # Recent command
    recent_parser: argparse.ArgumentParser = subparsers.add_parser("recent", help="Show recent usage logs")
    recent_parser.add_argument("-c", "--count", type=int, default=20, help="Number of logs to show")
    recent_parser.add_argument("-p", "--page", type=int, default=0, help="Page number")
    recent_parser.add_argument("-a", "--all", action="store_true", help="Show all logs (including title generation)")

    # Models command
    models_parser: argparse.ArgumentParser = subparsers.add_parser("models", help="Show all models and their prices")
    models_parser.add_argument("--full", action="store_true", help="Show full model ids (for api calls)")

    # Add other basic commands
    help_parser: argparse.ArgumentParser = subparsers.add_parser("help", help="Show help")
    help_parser.add_argument("help_command", nargs="?", help="Command to show help for")

    me_parser: argparse.ArgumentParser = subparsers.add_parser("me", help="Show your information and balance")

    return {
        "stats": stats_parser,
        "recent": recent_parser,
        "models": models_parser,
        "help": help_parser,
        "me": me_parser,
    }


def get_user_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    # User parser
    parser = argparse.ArgumentParser(description="User commands")
    subparsers = parser.add_subparsers(dest="command")
    added_parsers_map = add_user_parsers(subparsers)
    return parser, added_parsers_map


def get_admin_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    # Admin parser
    parser = argparse.ArgumentParser(description="Admin commands")
    subparsers = parser.add_subparsers(dest="command")
    added_parsers_map = add_user_parsers(subparsers)

    # Admin-only commands
    topup_parser = subparsers.add_parser("topup", help="Top up user balance")
    topup_parser.add_argument("--user", required=True, help="User ID or email")
    topup_parser.add_argument("--amount", type=float, required=True, help="Amount to add")

    set_balance_parser = subparsers.add_parser("set_balance", help="Set user balance")
    set_balance_parser.add_argument("--user", required=True, help="User ID or email")
    set_balance_parser.add_argument("--amount", type=float, required=True, help="New balance amount")

    gstats_parser = subparsers.add_parser("gstats", help="Show global usage statistics")
    gstats_parser.add_argument("-p", "--period", choices=["d", "w", "m"], help="Period (d: daily, w: weekly, m: monthly)")
    gstats_parser.add_argument("-m", "--model", type=str, default=None, help="Filter by model")
    gstats_parser.add_argument("-u", "--user_id", type=str, default=None, help="Filter by user id")
    gstats_parser.add_argument("--provider", type=str, default=None, help="Filter by provider")

    grecent_parser = subparsers.add_parser("grecent", help="Show global recent usage logs")
    grecent_parser.add_argument("-c", "--count", type=int, default=50, help="Number of logs to show")
    grecent_parser.add_argument("-p", "--page", type=int, default=0, help="Page number")
    grecent_parser.add_argument("-m", "--model", type=str, default=None, help="Filter by model")
    grecent_parser.add_argument("-u", "--user_id", type=str, default=None, help="Filter by user id")
    grecent_parser.add_argument("-a", "--all", action="store_true", help="Show all logs (including title generation)")

    list_users_parser = subparsers.add_parser("users", help="List all users")

    added_parsers_map.update(
        {
            "topup": topup_parser,
            "set_balance": set_balance_parser,
            "users": list_users_parser,
            "gstats": gstats_parser,
            "grecent": grecent_parser,
        }
    )

    return parser, added_parsers_map


def clean_usage(usage_text: str) -> str:
    usage_text = re.sub(r"uvicorn ?", "", usage_text)
    return usage_text


class ServiceBot:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.commands = {
            "models": self.models,
            "me": self.me,
            "topup": self.topup,
            "set_balance": self.set_balance,
            "users": self.users,
            "stats": self.stats,
            "gstats": self.gstats,
            "recent": self.recent,
            "grecent": self.grecent,
        }

    def handle_command(self, message: str, user: User) -> str:
        is_admin = user.role == "admin"
        parser, added_parsers_map = get_admin_parser() if is_admin else get_user_parser()
        help_text = f"```\n{clean_usage(parser.format_help())}\n```\n\nType `help <command>` for more information about a command."

        try:
            args = parser.parse_args(message.strip().split())
        except SystemExit:
            return "Invalid command.\n\n" + help_text
        except Exception as e:
            print_log(f"Error parsing command: {e}")
            return "Invalid command.\n\n" + help_text

        if not args.command:
            return "No command specified.\n\n" + help_text
        if args.command not in self.commands and args.command not in added_parsers_map:
            return "Unknown command.\n\n" + help_text

        if args.command in ["topup", "set_balance", "users", "gstats", "grecent"] and not is_admin:
            return help_text

        if args.command == "help":
            if args.help_command:
                return f"```\n{clean_usage(added_parsers_map[args.help_command].format_help())}\n```"
            else:
                return help_text

        try:
            result = self.commands[args.command](args, user)
        except Exception as e:
            print_log(f"Error executing command: {e}")
            return f"An error occurred while executing the command: {e}\n\n" + f"```\n{clean_usage(added_parsers_map[args.command].format_usage())}\n```"

        return result

    # Update the command methods to use the new args format
    def stats(self, args: argparse.Namespace, user: User) -> str:
        return self._get_model_stats(args.period, user.id, args.provider)

    def gstats(self, args: argparse.Namespace, user: User) -> str:
        assert user.role == "admin"
        if args.user_id:
            return self._get_model_stats(args.period, args.user_id, args.provider)
        elif args.model:
            return self._get_user_stats(args.period, args.model)
        else:
            return self._get_model_stats(args.period, provider=args.provider) + "\n\n" + self._get_user_stats(args.period, args.model)

    def recent(self, args: argparse.Namespace, user: User) -> str:
        return self._get_recent_logs(args.count, args.page, user.id, show_title_generation=args.all, filter_auxiliary_model=not args.all)

    def grecent(self, args: argparse.Namespace, user: User) -> str:
        assert user.role == "admin"
        return self._get_recent_logs(
            args.count, args.page, args.user_id, args.model, show_title_generation=args.all, filter_auxiliary_model=not args.all
        )

    def topup(self, args: argparse.Namespace, user: User) -> str:
        assert user.role == "admin"
        with Session(self.pipeline.engine) as session:
            db_user = session.exec(select(User).where((User.id == args.user) | (User.email == args.user))).first()
            if not db_user:
                return "User not found."
            db_user.balance += args.amount
            db_user.updated_at = datetime.now()
            session.commit()
            return f"Successfully topped up {db_user.email}'s balance by {self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{args.amount:.2f}. New balance: {self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{db_user.balance:.2f}"

    def set_balance(self, args: argparse.Namespace, user: User) -> str:
        assert user.role == "admin"
        with Session(self.pipeline.engine) as session:
            db_user = session.exec(select(User).where((User.id == args.user) | (User.email == args.user))).first()
            if not db_user:
                return "User not found."
            db_user.balance = args.amount
            db_user.updated_at = datetime.now()
            session.commit()
            return f"Successfully set {db_user.email}'s balance to {self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{args.amount:.2f}"

    def models(self, args: argparse.Namespace, user: User) -> str:
        price_ratio_map = {}
        for provider in self.pipeline.config.providers:
            price_ratio_map[provider.key] = provider.price_ratio

        headers = [
            "Provider",
            "Model",
            f"Prompt (per 1M)",
            f"Completion (per 1M)",
            f"Per Message",
            "Ratio",
        ]
        if args.full:
            headers.insert(2, "Slug")
        data = []

        for model_id, model in self.pipeline.models.items():
            prompt_price = model.prompt_price or 0
            completion_price = model.completion_price or 0
            per_message_price = model.per_message_price or 0
            price_ratio = price_ratio_map[model.provider] or 1

            base_cost_currency_unit = self.pipeline.valves.BASE_COST_CURRENCY_UNIT
            actual_cost_currency_unit = self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT
            actual_prompt_price = prompt_price * price_ratio
            actual_completion_price = completion_price * price_ratio
            actual_per_message_price = per_message_price * price_ratio

            prompt_price_text = (
                f"{actual_cost_currency_unit}{actual_prompt_price:.2f} ({base_cost_currency_unit}{prompt_price:.2f})" if prompt_price > 0 else "-"
            )
            completion_price_text = (
                f"{actual_cost_currency_unit}{actual_completion_price:.2f} ({base_cost_currency_unit}{completion_price:.2f})"
                if completion_price > 0
                else "-"
            )
            per_message_price_text = (
                f"{actual_cost_currency_unit}{actual_per_message_price:.2f} ({base_cost_currency_unit}{per_message_price:.2f})"
                if per_message_price > 0
                else "-"
            )

            item = [
                model.provider,
                model.human_name or model.code,
                prompt_price_text,
                completion_price_text,
                per_message_price_text,
                price_ratio_map[model.provider] or "-",
            ]
            if args.full:
                item.insert(2, f"{self.pipeline.id}.{model_id}")

            data.append(item)

        return f"{tabulate(data, headers=headers, tablefmt='pipe', colalign=('left',))}"

    def me(self, args: argparse.Namespace, user: User) -> str:
        return f"""
Your information:
- ID: {user.id}
- Email: {user.email}
- Name: {user.name or 'N/A'}
- Balance: {self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{user.balance:.6f}
- Role: {user.role}
- Created at: {user.created_at}
""".strip()

    def users(self, args: argparse.Namespace, user: User) -> str:
        with Session(self.pipeline.engine) as session:
            users = session.exec(select(User)).all()
            headers = ["ID", "Name", "Email", "Balance", "Role", "Updated At", "Created At"]
            data = [
                [
                    u.id,
                    u.name or "N/A",
                    u.email,
                    f"{self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{u.balance:.3f}",
                    u.role,
                    u.updated_at.astimezone(pytz.timezone(self.pipeline.valves.TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S"),
                    u.created_at.astimezone(pytz.timezone(self.pipeline.valves.TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S"),
                ]
                for u in users
            ]
            return f"{tabulate(data, headers=headers, tablefmt='pipe', colalign=('left',))}"

    def _get_model_stats(self, period: Optional[str] = None, user_id: Optional[int] = None, provider: Optional[str] = None) -> str:
        if period:
            time_delta = {
                "d": timedelta(days=1),
                "w": timedelta(weeks=1),
                "m": timedelta(days=30),
            }.get(period.lower(), None)
        else:
            time_delta = None

        with Session(self.pipeline.engine) as session:
            query = select(
                UsageLog.model,
                UsageLog.provider,
                func.sum(UsageLog.prompt_tokens).label("total_prompt_tokens"),
                func.sum(UsageLog.completion_tokens).label("total_completion_tokens"),
                func.sum(UsageLog.total_tokens).label("total_tokens"),
                func.sum(UsageLog.cost).label("total_cost"),
                func.sum(UsageLog.actual_cost).label("total_actual_cost"),
                func.count().label("count"),
                func.group_concat(distinct(User.name)).label("unique_user_names"),
            ).join(User, UsageLog.user_id == User.id)

            if time_delta:
                start_time = datetime.now() - time_delta
                query = query.where(UsageLog.created_at >= start_time)

            if user_id:
                query = query.where(UsageLog.user_id == user_id)
            
            if provider:
                query = query.where(UsageLog.provider.like(f"%{provider}%"))

            query = query.group_by(UsageLog.model, UsageLog.provider)

            results = session.exec(query).all()

            if not results:
                return "No usage records found for the specified time period."

            headers = ["Model", "Provider", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Base Cost", "Actual Cost", "Usage Count"]
            if not user_id:
                headers.append("Unique Users")
            data = []
            for r in results:
                row = [
                    r.model,
                    r.provider,
                    r.total_prompt_tokens,
                    r.total_completion_tokens,
                    r.total_tokens,
                    f"{self.pipeline.valves.BASE_COST_CURRENCY_UNIT}{r.total_cost:.6f}",
                    f"{self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{r.total_actual_cost:.6f}",
                    r.count,
                ]
                if not user_id:
                    row.append(r.unique_user_names)
                data.append(row)
            # sum row
            sum_row = [
                "Sum",
                "-",
                sum(r.total_prompt_tokens for r in results),
                sum(r.total_completion_tokens for r in results),
                sum(r.total_tokens for r in results),
                f"{self.pipeline.valves.BASE_COST_CURRENCY_UNIT}{sum(r.total_cost for r in results):.6f}",
                f"{self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{sum(r.total_actual_cost for r in results):.6f}",
                sum(r.count for r in results),
            ]
            if not user_id:
                sum_row.append(len(set(name for r in results for name in r.unique_user_names.split(",") if name)))
            data.append(sum_row)

            table = tabulate(data, headers=headers, tablefmt="pipe", colalign=("left",))
            if period:
                period_str = {"d": "Daily", "w": "Weekly", "m": "Monthly"}.get(period.lower(), "Daily")
                resp = f"{period_str} Statistics from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n\n{table}"
            else:
                resp = f"All time Statistics:\n\n{table}"

            return resp

    def _get_user_stats(self, period: Optional[str] = None, model: Optional[str] = None) -> str:
        with Session(self.pipeline.engine) as session:
            # 计算时间范围
            start_time = None
            if period:
                now = datetime.now()
                if period.lower() == "d":
                    start_time = now - timedelta(days=1)
                elif period.lower() == "w":
                    start_time = now - timedelta(weeks=1)
                elif period.lower() == "m":
                    start_time = now - timedelta(days=30)

            # 构建查询
            query = (
                select(
                    User.name,
                    User.id,
                    func.sum(UsageLog.prompt_tokens).label("total_prompt_tokens"),
                    func.sum(UsageLog.completion_tokens).label("total_completion_tokens"),
                    func.sum(UsageLog.total_tokens).label("total_tokens"),
                    func.sum(UsageLog.cost).label("total_cost"),
                    func.sum(UsageLog.actual_cost).label("total_actual_cost"),
                    func.count(UsageLog.id).label("count"),
                    func.group_concat(distinct(UsageLog.model)).label("used_models"),
                )
                .join(UsageLog, User.id == UsageLog.user_id)
                .where(UsageLog.is_title_generation == False)
            )

            if start_time:
                query = query.where(UsageLog.created_at >= start_time)
            if model:
                query = query.where(UsageLog.model.like(f"%{model}%"))

            query = query.group_by(User.id).order_by(desc("total_actual_cost"))

            results = session.exec(query).all()

            if not results:
                return "No usage records found for the specified time period."

            headers = ["User", "ID", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Base Cost", "Actual Cost", "Usage Count", "Used Models"]
            data = []
            for r in results:
                row = [
                    r.name or "N/A",
                    r.id,
                    r.total_prompt_tokens,
                    r.total_completion_tokens,
                    r.total_tokens,
                    f"{self.pipeline.valves.BASE_COST_CURRENCY_UNIT}{r.total_cost:.6f}",
                    f"{self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{r.total_actual_cost:.6f}",
                    r.count,
                    r.used_models.replace(",", ", "),
                ]
                data.append(row)

            # Add summary row
            sum_row = [
                "Total",
                "-",
                sum(r.total_prompt_tokens for r in results),
                sum(r.total_completion_tokens for r in results),
                sum(r.total_tokens for r in results),
                f"{self.pipeline.valves.BASE_COST_CURRENCY_UNIT}{sum(r.total_cost for r in results):.6f}",
                f"{self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{sum(r.total_actual_cost for r in results):.6f}",
                sum(r.count for r in results),
                "-",
            ]
            data.append(sum_row)

            table = tabulate(data, headers=headers, tablefmt="pipe", colalign=("left",))
            if period:
                period_str = {"d": "Daily", "w": "Weekly", "m": "Monthly"}.get(period.lower(), "Daily")
                resp = f"User {period_str} Statistics (from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n\n{table}"
            else:
                resp = f"All Time User Statistics:\n\n{table}"

            return resp

    def _get_recent_logs(
        self,
        count: int,
        page: int,
        user_id: Optional[int] = None,
        model: Optional[str] = None,
        show_title_generation: bool = False,
        exclude_non_stream: bool = False,
        filter_auxiliary_model: bool = True,
    ) -> str:
        with Session(self.pipeline.engine) as session:
            query = select(UsageLog, User.name, User.id).join(User, UsageLog.user_id == User.id)
            if not show_title_generation:
                query = query.where(UsageLog.is_title_generation == False)
            if user_id:
                query = query.where(UsageLog.user_id == user_id)
            if model:
                query = query.where(UsageLog.model.like(f"%{model}%"))
            if exclude_non_stream:
                query = query.where(UsageLog.is_stream == True)
            if filter_auxiliary_model:
                query = query.where(UsageLog.model.not_like(f"%{self.pipeline.valves.AUXILIARY_MODEL_CODE}%"))

            query = query.order_by(UsageLog.created_at.desc()).offset(page * count).limit(count)

            results = session.exec(query).all()

            if not results:
                return "No recent usage logs found."

            headers = ["Time", "Provider", "Model", "Tokens", "Base Cost", "Actual Cost", "Content", "Stream"]
            if not user_id:
                headers.insert(1, "User")
            if show_title_generation:
                headers.append("Title Generation")

            data = []
            for r in results:
                content = r.UsageLog.content[:20] if r.UsageLog.content else "-"
                content = content.replace("<think>", "").replace("</think>", "")
                if len(r.UsageLog.content) > 20:
                    content += "..."
                content = content.replace("\n", "\\n")
                content = content.replace("\r", "\\r")
                content = f"`{content}`"
                row = [
                    r.UsageLog.created_at.astimezone(pytz.timezone(self.pipeline.valves.TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S"),
                    r.UsageLog.provider,
                    r.UsageLog.model,
                    r.UsageLog.total_tokens,
                    f"{self.pipeline.valves.BASE_COST_CURRENCY_UNIT}{r.UsageLog.cost:.6f}",
                    f"{self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{r.UsageLog.actual_cost:.6f}",
                    content,
                    r.UsageLog.is_stream,
                ]
                if not user_id:
                    row.insert(1, f"{r.name or 'N/A'} (ID: {r.id})")
                if show_title_generation:
                    row.append(r.UsageLog.is_title_generation)
                data.append(row)

            table = tabulate(data, headers=headers, tablefmt="pipe", colalign=("left",))
            return f"Recent Usage Logs (Page {page}):\n\n{table}"
