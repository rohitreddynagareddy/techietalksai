import lmstudio as lms

def multiply(a: float, b: float) -> float:
    """Given two numbers a and b. Returns the product of them."""
    return a * b

llm = lms.llm()

llm.act(
  "What is the result of 2 multiplied by 3?",
  [multiply],
  on_prediction_completed=print
)