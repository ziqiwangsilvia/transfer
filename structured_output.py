import json
from typing import List, Literal, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI

# --- 1. Tool Argument Schemas ---

# Define the allowed financial categories
FinancialCategory = Literal[
    "Groceries", "Restaurants", "Coffee & Bars", "Public transport", "Fuel", 
    "Taxi & Car sharing", "Rent/Mortgage", "Utilities", "Home improvement", 
    "Clothing & Shoes", "Electronics", "Home & Garden", "Subscriptions", 
    "Hobbies", "Events", "Insurance", "Bank fees", "Loans", "Pharmacy", 
    "Sport & Fitness", "Medical", "Hair & Beauty", "Spa & Wellness"
]

TimeRange = Literal["this_month", "last_month", "last_30_days", "last_90_days", "this_year"]

# Shared Filter for all tools
class DataFilter(BaseModel):
    categories: Optional[List[FinancialCategory]] = None
    payees: Optional[List[str]] = None

# --- Tool Argument Definitions ---
class LineChartArgs(BaseModel):
    metric: Literal["balance", "net_cash_flow", "income", "spending"]
    time_range: TimeRange
    title: Optional[str] = None
    filter: Optional[DataFilter] = None

class PieChartArgs(BaseModel):
    data_type: Literal["spending", "income"]
    time_range: TimeRange
    group_by: Literal["category", "payee"]
    limit: Optional[int] = Field(None, ge=1)
    filter: Optional[DataFilter] = None

class StackedBarArgs(BaseModel):
    metrics: List[Literal["income", "spending"]]
    time_range: TimeRange
    group_by: Literal["month", "category"]
    filter: Optional[DataFilter] = None

# --- Unified Structure ---
class ToolCall(BaseModel):
    name: Literal["show_line_chart", "show_pie_chart", "show_stacked_bar"]
    args: Union[LineChartArgs, PieChartArgs, StackedBarArgs]

class AssistantResponse(BaseModel):
    content: Optional[str] = Field(None, description="Direct answer if no tool is used")
    tool_calls: Optional[ToolCall] = Field(None, description="Tool data if chart is requested")

# --- 3. Execution ---

client = OpenAI(base_url="http://localhost:8000/v1", api_key="vllm-token")

def get_gemma_response(user_input: str):
    completion = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {
                "role": "system", 
                "content": "You are a financial assistant. Use 'tool_calls' for charts or 'content' for chat."
            },
            {"role": "user", "content": user_input}
        ],
        extra_body={
            "guided_json": AssistantResponse.model_json_schema(),
            "guided_decoding_backend": "xgrammar"
        }
    )
    return json.loads(completion.choices.message.content)

# --- Test Cases ---
# 1. Stacked Bar Chart Trigger
print("Stacked Bar:", get_gemma_response("Compare my income and spending month-by-month for this year using a stacked bar chart."))

# 2. Conversational Trigger
print("\nChat:", get_gemma_response("What is a stacked bar chart useful for?"))


import json
from typing import Dict, Any

def _message_to_dict(message) -> Dict[str, Any]:
    """Convert an API response message into a plain result dict."""
    
    # 1. Handle native tool calls (if any)
    if message.tool_calls:
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        }

    content = message.content or ""

    # 2. Handle Gemma/vLLM Structured JSON Output
    # We check if it's a JSON string and contains our 'tool_calls' key
    if structured_labels and content.strip().startswith("{"):
        try:
            parsed = json.loads(content)
            
            # If the model chose to use a tool (based on our AssistantResponse schema)
            if "tool_calls" in parsed and parsed["tool_calls"]:
                tc = parsed["tool_calls"]
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]), # Stringify to match API standard
                            },
                        }
                    ],
                }
            
            # If the model chose a direct conversational answer
            if "content" in parsed:
                content = parsed["content"] or ""
                
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback to raw content if parsing fails
            pass

    return {"role": "assistant", "content": content}

 def _build_kwargs(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
            kwargs: Dict[str, Any] = {
                "model": self.config.name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            if stop_sequences:
                kwargs["stop"] = stop_sequences
            if tools and self.config.template_has_tool_token:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            if self.config.use_structured:
                kwargs["extra_body"] ={
                "guided_json": AssistantResponse.model_json_schema(),
                "guided_decoding_backend": "xgrammar"
            }
            return kwargs