import json
from typing import List, Literal, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI

# --- 1. Tool Argument Schemas ---

TimeRange = Literal["this_month", "last_month", "last_30_days", "last_90_days", "this_year"]

class LineChartArgs(BaseModel):
    metric: Literal["balance", "net_cash_flow", "income", "spending"]
    time_range: TimeRange
    title: Optional[str] = None

class PieChartArgs(BaseModel):
    data_type: Literal["spending", "income"]
    time_range: TimeRange
    group_by: Literal["category", "payee"]
    limit: Optional[int] = None
    categories: Optional[List[str]] = None 

class StackedBarArgs(BaseModel):
    metrics: List[str] = Field(..., description="Metrics to stack, e.g., ['income', 'spending']")
    time_range: TimeRange
    group_by: Literal["month", "category"] = "month"
    title: Optional[str] = None

# --- 2. Unified Output Structure ---

class ToolCall(BaseModel):
    # Added 'show_stacked_bar' to the literal names
    name: Literal["show_line_chart", "show_pie_chart", "show_stacked_bar"]
    args: Union[LineChartArgs, PieChartArgs, StackedBarArgs]

class AssistantResponse(BaseModel):
    content: Optional[str] = Field(None, description="Direct conversational response")
    tool_calls: Optional[ToolCall] = Field(None, description="Structured tool call")

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
