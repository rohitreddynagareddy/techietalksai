# agents/booking_agent.py
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from tools.booking_tool import BookingTool

class BookingAgent:
    def __init__(self):
        llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini")
        tools = [
            Tool(
                name="BookingTool",
                func=BookingTool().run,
                description="Handles reservations and bookings"
            )
        ]
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def run(self, input_text: str) -> str:
        return self.agent.run(input_text)