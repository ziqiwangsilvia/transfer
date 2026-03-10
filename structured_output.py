from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

FinancialCategory = Literal[
    "Groceries",
    "Restaurants",
    "Coffee & Bars",
    "Public transport",
    "Fuel",
    "Taxi & Car sharing",
    "Rent/Mortgage",
    "Utilities",
    "Home improvement",
    "Clothing & Shoes",
    "Electronics",
    "Home & Garden",
    "Subscriptions",
    "Hobbies",
    "Events",
    "Insurance",
    "Bank fees",
    "Loans",
    "Pharmacy",
    "Sport & Fitness",
    "Medical",
    "Hair & Beauty",
    "Spa & Wellness",
]

TimeRange = Literal[
    "this_month", "last_month", "last_30_days", "last_90_days", "this_year"
]


class DataFilter(BaseModel):
    categories: Optional[List[FinancialCategory]] = None
    payees: Optional[List[str]] = None


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


class ToolCall(BaseModel):
    name: Literal["show_line_chart", "show_pie_chart", "show_stacked_bar_chart", "stats_retrieval"]
    args: Union[LineChartArgs, PieChartArgs, StackedBarArgs]


class AssistantResponse(BaseModel):
    content: Optional[str] = Field(None, description="Direct answer if no tool is used")
    tool_calls: Optional[ToolCall] = Field(
        None, description="Tool data if chart is requested"
    )
