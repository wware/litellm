import os
import json
from litellm import completion
from typing import Dict, Any, List, Callable
from dotenv import load_dotenv

load_dotenv()

def tool_foo(x: int) -> int:
    """Add 5 to the integer argument"""
    print(f"🔧 Calling tool_foo with x={x}")
    result = x + 5
    print(f"🔧 tool_foo returned: {result}")
    return result

tools = [
    {
        "type": "function",
        "function": {
            "name": "tool_foo",
            "description": "Add 5 to the integer argument",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "The integer to add 5 to"
                    }
                },
                "required": ["x"]
            }
        }
    }
]

function_map: Dict[str, Callable] = {
    "tool_foo": tool_foo
}

def execute_function_call(function_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a function call and return the result."""
    if function_name not in function_map:
        raise ValueError(f"Unknown function: {function_name}")
    
    func = function_map[function_name]
    try:
        result = func(**arguments)
        return result
    except Exception as e:
        return f"Error executing {function_name}: {str(e)}"

def main():
    """Chat using LiteLLM with the custom LN provider"""
    
    # Get model from environment
    model_name = os.getenv("LLM_PROXY_MODEL", "")
    assert model_name, "LLM_PROXY_MODEL must be set"
    
    # Initial conversation
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant. You have access to the following tools:

{json.dumps(tools, indent=2)}

When you need to use a tool, respond with a JSON object containing:
- "tool_call": true
- "function_name": the name of the function to call
- "arguments": the arguments to pass to the function

For example:
{{"tool_call": true, "function_name": "tool_foo", "arguments": {{"x": 10}}}}

Otherwise, respond normally."""
        },
        {
            "role": "user",
            "content": "What is 12345 plus 5?"
        }
    ]
    
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 Iteration {iteration}")
        print(f"💬 Current messages: {len(messages)}")
        
        try:
            # Now you can use LiteLLM with your custom provider!
            # Environment variables are loaded automatically by the provider
            response = completion(
                model=f"ln_proxy/{model_name}",  # Use ln_proxy provider prefix
                messages=messages,
                tools=tools,
                tool_choice="auto",
                # LN-specific parameters can be passed through
                asset_id=os.getenv("ASSET_ID", ""),
                timeout=180,
                priority="low"
            )
            
            # Process the response using standard LiteLLM format
            assistant_message = response.choices[0].message
            print(f"🤖 Assistant response: {assistant_message.content}")
            
            # Handle tool calls if present
            if assistant_message.tool_calls:
                print("🔧 Tool calls detected!")
                
                # Add assistant message to conversation
                messages.append(assistant_message.model_dump())
                
                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"🔧 Executing {function_name}({arguments})")
                    result = execute_function_call(function_name, arguments)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"The calculation is complete: {arguments.get('x', 'input')} + 5 = {result}. Please give this result as your final answer."
                    })
                
                continue
            
            # Regular response - we're done
            print(f"💬 Final answer: {assistant_message.content}")
            break
            
        except Exception as e:
            print(f"❌ API Error: {e}")
            print(f"❌ Error details: {type(e).__name__}: {str(e)}")
            break

if __name__ == "__main__":
    main()
