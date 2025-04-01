# app/models.py
from pydantic import BaseModel, Field
from typing import Optional

class UserDetails(BaseModel):
    """Represents extracted user details."""
    name: str = Field(..., description="The full name of the user.")
    age: Optional[int] = Field(None, description="The age of the user, if mentioned.")
    city: Optional[str] = Field(None, description="The city where the user lives, if mentioned.")
    email: Optional[str] = Field(None, description="Email address, if mentioned.")
    profession: Optional[str] = Field(None, description="The user's profession, if mentioned.")

class ProductInfo(BaseModel):
    """Represents extracted product information."""
    product_name: str = Field(..., description="The name of the product.")
    price: Optional[float] = Field(None, description="The price of the product.")
    currency: Optional[str] = Field(None, description="The currency symbol or code (e.g., $, EUR).")
