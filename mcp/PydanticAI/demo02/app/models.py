# app/models.py
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, HttpUrl

class UserDetails(BaseModel):
    """Schema for extracting user details."""
    name: Optional[str] = Field(None, description="The user's full name")
    age: Optional[int] = Field(None, description="The user's age")
    location: Optional[str] = Field(None, description="The user's city and country, if available")
    email: Optional[EmailStr] = Field(None, description="The user's email address")
    sentiment: Optional[str] = Field(None, description="The overall sentiment of the input text (e.g., Positive, Negative, Neutral)") # <-- Added field

class ProductInfo(BaseModel):
    """Schema for extracting product information."""
    product_name: Optional[str] = Field(None, description="The name of the product mentioned")
    price: Optional[float] = Field(None, description="The price of the product")
    currency: Optional[str] = Field(None, description="The currency symbol or code (e.g., $, USD)")
    features: Optional[List[str]] = Field(None, description="A list of key features mentioned")
    sentiment: Optional[str] = Field(None, description="The overall sentiment of the input text (e.g., Positive, Negative, Neutral)") # <-- Added field

class ThoughtProcess(BaseModel):
    thought: str
    # action: Optional[Literal['arithmetic', 'trigonometry']] = None
    action: Optional[str] = None
    action_input: Optional[str] = None
    answer: Optional[str] = None
    sentiment: Optional[str] = Field(None, description="The overall sentiment of the input text (e.g., Positive, Negative, Neutral)") # <-- Added field
    
# Make sure you have an __init__.py in the 'app' directory
# touch app/__init__.py