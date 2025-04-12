import lmstudio as lms
from pydantic import BaseModel

class Book(BaseModel):
    title: str
    author: str
    year: int

llm = lms.llm()

response = llm.respond(
    "Tell me about The Hobbit.",
    response_format=Book,
)
print(response)
