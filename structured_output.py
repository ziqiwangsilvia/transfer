from typing import List, Literal, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

FinancialCategory = Literal[
    "Groceries", "Restaurants", "Coffee & Bars", "Public transport", "Fuel", 
    "Taxi & Car sharing", "Rent/Mortgage", "Utilities", "Home improvement", 
    "Clothing & Shoes", "Electronics", "Home & Garden", "Subscriptions", 
    "Hobbies", "Events", "Insurance", "Bank fees", "Loans", "Pharmacy", 
    "Sport & Fitness", "Medical", "Hair & Beauty", "Spa & Wellness"
]

PayeeName = Literal[
    "ASOS", "AXA", "Admiral Insurance", "Aldi", "Amazon Digital", "Amazon Prime", 
    "Apple Store", "Argos", "Art Supplies", "Aviva", "B&Q", "BP", "Beauty Salon", 
    "Bolt", "Boots", "British Gas", "Bupa Healthcare", "Champneys", "CityMapper", 
    "Costa Coffee", "Currys", "David Lloyd", "Day Spa", "Dental Practice", 
    "Dishoom", "Disney+"
]

TimeRange = Literal["this_month", "last_month", "last_30_days", "last_90_days", "this_year"]

class DataFilter(BaseModel):
    categories: Optional[List[FinancialCategory]] = None
    payees: Optional[List[PayeeName]] = None

class CommonDataSource(BaseModel):
    data_type: Literal["spending", "income"]
    time_range: TimeRange
    group_by: Literal["category", "payee"]
    limit: Optional[int] = None
    filter: Optional[DataFilter] = None

class StatsDataSource(BaseModel):
    data_type: Literal["balance", "income", "spending"]
    time_range: TimeRange
    filter: Optional[DataFilter] = None

class LineChartArgs(BaseModel):
    metric: Literal["balance", "net_cash_flow", "income", "spending"]
    time_range: TimeRange
    title: Optional[str] = None

class PieChartArgs(BaseModel):
    data_source: CommonDataSource
    title: Optional[str] = None

class StackedBarArgs(BaseModel):
    data_source: CommonDataSource
    title: Optional[str] = None

class StatsRetrievalArgs(BaseModel):
    data_source: StatsDataSource
    title: Optional[str] = None

class ToolCall(BaseModel):
    name: Literal["show_line_chart", "show_pie_chart", "show_stacked_bar_chart", "stats_retrieval"]
    args: Union[LineChartArgs, PieChartArgs, StackedBarArgs, StatsRetrievalArgs]

class AssistantResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[ToolCall] = None
