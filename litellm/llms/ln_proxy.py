# litellm/llms/ln_proxy.py
import os
import json
import requests
from typing import Optional, Dict, Any, List, Union
from dotenv import load_dotenv

# Import LiteLLM utilities
from litellm.utils import ModelResponse, Choices, Message, Usage, CustomStreamWrapper
from litellm.types.utils import ChatCompletionMessageToolCall, Function, ChatCompletionMessageToolCallFunction
import litellm

# Load environment variables
load_dotenv()

class LNProxyProvider:
    def __init__(self):
        self.provider_name = "ln_proxy"

    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_base: str = None,
        api_key: str = None,
        custom_llm_provider: str = "ln_proxy",
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Convert LiteLLM standard format to LN Proxy format with all fixes applied
        """

        # Use environment variables as defaults if not provided
        api_base = api_base or os.getenv("LLM_PROXY_BASE", "")
        api_key = api_key or os.getenv("LLM_PROXY_TENANT", "")

        if not api_base:
            raise ValueError("LLM_PROXY_BASE must be set in environment or passed as api_base")
        if not api_key:
            raise ValueError("LLM_PROXY_TENANT must be set in environment or passed as api_key")

        # Extract LN-specific parameters
        asset_id = kwargs.get("asset_id") or os.getenv("ASSET_ID", "")
        timeout = kwargs.get("timeout", 180)
        tenant = api_key  # Use api_key as tenant

        # Build URL - handle both cases (lesson from Option 3)
        api_base = api_base.rstrip('/')
        if api_base.endswith('/proxy'):
            url = f"{api_base}/{model}"
        else:
            url = f"{api_base}/proxy/{model}"

        # Handle Anthropic system message format (lesson from Option 3)
        system_message = None
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                filtered_messages.append(msg)

        # Determine chat mode - LiteLLM typically uses chat format
        chat_mode = kwargs.get("chat", len(filtered_messages) > 1)

        # Format query based on chat mode (lesson from Option 3)
        if chat_mode:
            query_data = str((filtered_messages,))  # Stringified tuple format
        else:
            # For non-chat, combine system and user message
            if system_message and filtered_messages:
                combined_content = f"{system_message}\n\n{filtered_messages[-1]['content']}"
                query_data = combined_content
            elif filtered_messages:
                query_data = filtered_messages[-1]["content"]
            else:
                query_data = ""

        # Build config with proper parameter names (lesson from Option 3)
        config = {
            "max_tokens_to_sample": kwargs.get("max_tokens", 200),
            "temperature": kwargs.get("temperature", 0.0),
            "instruction": system_message  # Put system message here for Anthropic
        }

        # Add LN-specific config options
        for key in ["secure_check", "non_local_failover_optout", "stream", "append_instruction"]:
            if key in kwargs:
                config[key] = kwargs[key]

        # Build tracing info
        tracing_info = kwargs.get("tracing_info", {
            "asset_id": asset_id,
            "trans_id": "",
            "user_id": "",
            "user_type": "",
            "ext_info": {}
        })

        # Build request payload
        request_data = {
            "query": query_data,
            "config": config,
            "chat": chat_mode,
            "tracing_info": tracing_info
        }

        # Build headers exactly like working example (lesson from Option 3)
        headers = {
            "user-agent": "llm-proxy/0.18.19",
            "timeout": str(timeout),
            "tenant": tenant,
            "disable-cache": "",
            "priority": kwargs.get("priority", "low"),
            "breaker-strategy": kwargs.get("breaker_strategy", "return"),
            "Accept-Encoding": "gzip"
        }

        # Add optional headers
        for header_key, kwarg_key in [
            ("transaction-id", "transaction_id"),
            ("user-agent", "user_agent")
        ]:
            if kwarg_key in kwargs:
                headers[header_key] = str(kwargs[kwarg_key])

        # Make the request
        try:
            response = requests.post(
                url,
                json=request_data,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            response_data = response.json()

            # Convert response back to LiteLLM format
            return self._convert_response(response_data, model, **kwargs)

        except requests.exceptions.RequestException as e:
            raise Exception(f"LN Proxy request failed: {str(e)}")

    def _extract_assistant_content(self, response_data: Dict[str, Any]) -> str:
        """Extract assistant content from LN Proxy response (lesson from Option 3)"""
        if isinstance(response_data, dict):
            # The actual AI response is in the "answer" field
            assistant_content = response_data.get("answer", "")
            if not assistant_content:
                # Fallback to other possible fields
                assistant_content = (
                    response_data.get("content") or
                    response_data.get("text") or
                    response_data.get("message") or
                    str(response_data)
                )
        else:
            assistant_content = str(response_data)
        return assistant_content

    def _extract_tool_calls(self, text: str) -> List[ChatCompletionMessageToolCall]:
        """Extract tool calls from text using balanced brace matching (lesson from Option 3)"""
        def find_balanced_json(text: str) -> Optional[str]:
            start = text.find('{"tool_call":')
            if start == -1:
                return None

            brace_count = 0
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i+1]
            return None

        json_str = find_balanced_json(text)
        if json_str:
            try:
                parsed_tool_call = json.loads(json_str)
                if parsed_tool_call.get("tool_call"):
                    # Convert to LiteLLM tool call format
                    return [
                        ChatCompletionMessageToolCall(
                            id=f"call_{hash(json_str) % 1000000}",  # Generate a simple ID
                            function=ChatCompletionMessageToolCallFunction(
                                name=parsed_tool_call["function_name"],
                                arguments=json.dumps(parsed_tool_call["arguments"])
                            ),
                            type="function"
                        )
                    ]
            except (json.JSONDecodeError, KeyError):
                pass

        return []

    def _convert_response(self, response_data: Dict[str, Any], model: str, **kwargs) -> ModelResponse:
        """
        Convert LN Proxy response to LiteLLM ModelResponse format
        """
        # Extract the assistant content
        assistant_content = self._extract_assistant_content(response_data)

        # Check for tool calls in the content
        tool_calls = self._extract_tool_calls(assistant_content)

        # Create the message
        message = Message(
            content=assistant_content if not tool_calls else None,  # OpenAI spec: null content if tool calls
            role="assistant",
            tool_calls=tool_calls if tool_calls else None
        )

        # Create choices
        choices = [
            Choices(
                finish_reason="tool_calls" if tool_calls else "stop",
                index=0,
                message=message
            )
        ]

        # Extract usage information if available
        usage_data = response_data.get("llm_metadata", {}).get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )

        return ModelResponse(
            id=response_data.get("llm_metadata", {}).get("response_body", {}).get("id", f"ln-{hash(str(response_data)) % 1000000}"),
            choices=choices,
            created=response_data.get("created", 0),
            model=model,
            object="chat.completion",
            usage=usage
        )

# Integration function for LiteLLM
def completion(
    model: str,
    messages: List[Dict[str, str]],
    api_base: str = None,
    api_key: str = None,
    custom_llm_provider: str = "ln_proxy",
    **kwargs
) -> ModelResponse:
    """
    LN Proxy completion function for LiteLLM integration
    """
    provider = LNProxyProvider()
    return provider.completion(
        model=model,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        custom_llm_provider=custom_llm_provider,
        **kwargs
    )

# Example usage with LiteLLM after integration
def example_usage():
    """Example of how to use the LN provider with LiteLLM"""
    from litellm import completion as litellm_completion

    # Your environment variables (same as before!)
    model_name = os.getenv("LLM_PROXY_MODEL", "")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is 12345 plus 5?"
        }
    ]

    # Use with LiteLLM - this would work after integrating into LiteLLM
    response = litellm_completion(
        model=f"ln_proxy/{model_name}",  # Use ln_proxy provider prefix
        messages=messages,
        # Environment variables loaded automatically
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "tool_foo",
                    "description": "Add 5 to an integer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer"}
                        },
                        "required": ["x"]
                    }
                }
            }
        ],
        tool_choice="auto"
    )

    return response

if __name__ == "__main__":
    # Test the provider directly
    provider = LNProxyProvider()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 12345 plus 5?"}
    ]

    try:
        response = provider.completion(
            model=os.getenv("LLM_PROXY_MODEL", ""),
            messages=messages
        )
        print("✅ Provider test successful!")
        print(f"Response: {response.choices[0].message.content}")
        if response.choices[0].message.tool_calls:
            print(f"Tool calls: {response.choices[0].message.tool_calls}")
    except Exception as e:
        print(f"❌ Provider test failed: {e}")

