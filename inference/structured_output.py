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

PayeeName = Literal[
    "ASOS",
    "AXA",
    "Admiral Insurance",
    "Aldi",
    "Amazon Digital",
    "Amazon Prime",
    "Apple Store",
    "Argos",
    "Art Supplies",
    "Aviva",
    "B&Q",
    "BP",
    "Beauty Salon",
    "Bolt",
    "Boots",
    "British Gas",
    "Bupa Healthcare",
    "Champneys",
    "CityMapper",
    "Costa Coffee",
    "Currys",
    "David Lloyd",
    "Day Spa",
    "Dental Practice",
    "Dishoom",
    "Disney+",
    "Dunelm",
    "Enterprise Car",
    "Esso",
    "Eventbrite",
    "FreeNow",
    "Games Workshop",
    "Garden Center",
    "Golf Club",
    "Gymshark",
    "H&M",
    "Homebase",
    "IKEA",
    "John Lewis",
    "Lidl",
    "Lloyds Pharmacy",
    "Local Barber",
    "Local Italian",
    "Massage Center",
    "Monthly Account Fee",
    "Mortgage_Payment_Direct",
    "Nandos",
    "Netflix",
    "Nike Store",
    "O2 Arena",
    "Octopus Energy",
    "Optical Express",
    "Overdraft Interest",
    "Personal Loan Pymt",
    "Pub",
    "PureGym",
    "RE_MGMT_SVC",
    "Sainsburys",
    "Selfridges",
    "Shell",
    "Sky Digital",
    "Spotify",
    "Starbucks",
    "Student Loan Co",
    "Superdrug",
    "TFL.gov.uk",
    "Tesco",
    "Texaco",
    "Thames Water",
    "The Alchemist",
    "Ticketmaster",
    "Toni & Guy",
    "Trainline",
    "Uber",
    "Vitality",
    "Wagamama",
    "Waitrose",
    "Wayfair",
    "ZARA",
]

TimeRange = Literal[
    "this_month", "last_month", "last_30_days", "last_90_days", "this_year", "last_year"
]


class LineChartArgs(BaseModel):
    chart_type: Literal["balance", "cash_flow"]
    time_range: TimeRange
    by_account: bool


class PieChartArgs(BaseModel):
    data_type: Literal["spending", "income"]
    time_range: TimeRange
    group_by: Literal["category", "payee"]
    limit: Optional[int] = None
    categories: Optional[List[FinancialCategory]] = None
    payees: Optional[List[PayeeName]] = None


class StackedBarArgs(BaseModel):
    data_type: Literal["spending", "income"]
    time_range: TimeRange
    group_by: Literal["category", "payee"]
    limit: Optional[int] = None
    categories: Optional[List[FinancialCategory]] = None
    payees: Optional[List[PayeeName]] = None


class ToolCall(BaseModel):
    name: Literal["show_line_chart", "show_pie_chart", "show_stacked_bar_chart"]
    args: Union[LineChartArgs, PieChartArgs, StackedBarArgs]


class AssistantResponse(BaseModel):
    content: Optional[str] = Field(None, description="Direct answer if no tool is used")
    tool_calls: Optional[ToolCall] = Field(
        None, description="Tool data if chart is requested"
    )
