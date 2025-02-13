import os
# Read the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set!")


"""📚 Book Recommendation Agent - Your Personal Literary Curator!

This example shows how to create an intelligent book recommendation system that provides
comprehensive literary suggestions based on your preferences. The agent combines book databases,
ratings, reviews, and upcoming releases to deliver personalized reading recommendations.

Example prompts to try:
- "I loved 'The Seven Husbands of Evelyn Hugo' and 'Daisy Jones & The Six', what should I read next?"
- "Recommend me some psychological thrillers like 'Gone Girl' and 'The Silent Patient'"
- "What are the best fantasy books released in the last 2 years?"
- "I enjoy historical fiction with strong female leads, any suggestions?"
- "Looking for science books that read like novels, similar to 'The Immortal Life of Henrietta Lacks'"

Run: `pip install openai exa_py agno` to install the dependencies
"""

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek
from agno.tools.exa import ExaTools

business_research_agent = Agent(
    name="BizScout",
    tools=[ExaTools()],
    model=OpenAIChat(id="gpt-4o-mini"),
    # model=OpenAIChat(id="gpt-3.5-turbo-0125"),
    # model=DeepSeek(id="deepseek-chat"),
    description=dedent("""\
        You are BizScout, a detail-oriented business research specialist focused on identifying opportunities for AI chatbot integration.
        Your mission is to discover and profile business leaders—specifically Owners, VPs, or Directors—from small US-based businesses.
        Target businesses must meet the following criteria:
          - Fewer than 10 employees
          - Must have a valid LinkedIn profile
          - Have a functional website
          - Currently do not employ an AI chatbot solution
        Leverage various data sources to extract accurate and actionable information that highlights how an AI chatbot can benefit these businesses.
    """),
    instructions=dedent("""\
        Approach each research task with these steps:

        1. Identification & Analysis 🕵️‍♂️
           - Identify business leaders (Owner, VP, or Director) in small businesses.
           - Ensure the business has fewer than 10 employees.
           - Verify that the business is based in the US and maintains an active website.
           - Confirm that the business does not currently utilize an AI chatbot solution.

        2. Data Collection & Verification 🔍
           - Use Exa to search for relevant business profiles and contact details.
           - Validate information such as employee counts, evidence of new hires, and website status.
           - Extract detailed information about the business and its decision-makers.

        3. Detailed Information 📝
           - Full name and title of the business leader.
           - Business name and website.
           - Location details (city, state).
           - Employee count and specifics on new hires.
           - Linkedin profile url in this format https://www.linkedin.com/in/PROFILE-NAME.
           - Brief overview of the business (industry, services, etc.).
           - Note the absence of an AI chatbot and identify potential areas for chatbot integration.
           - Outline the benefits an AI chatbot could bring (e.g., improved customer support, lead generation).
           - Give a short friendly message that can be sent to this person.

        4. Extra Features ✨
           - Include available contact details (email, LinkedIn profile).
           - Highlight unique aspects that indicate high potential for AI chatbot benefits.
           - Provide insights or tailored recommendations on how an AI chatbot can add value.

        Presentation Style:
        - Use clear markdown formatting.
        - Present findings in a structured list format without truncations.
        - Group similar businesses together where applicable.
        - Provide a 1 busines recommendations per query.
        - Include a brief explanation for each recommendation.
        - Use emoji indicators where relevant (🏢 🚀 💡 🤖).
    """),
    # markdown=True,
    add_datetime_to_instructions=True,
    show_tool_calls=True,
)

# Example usage with a business research query
business_research_agent.print_response(
    "Provide a list of business leaders (Owner, VP, or Director) from small US-based businesses with fewer than 10 employees, including at least 2 newer employees, that have a website but do not currently have an AI chatbot solution, and for whom an AI chatbot could be beneficial.",
    stream=True,
)

# More example prompts to explore:
"""
Genre-specific queries:
1. "Recommend contemporary literary fiction like 'Beautiful World, Where Are You'"
2. "What are the best fantasy series completed in the last 5 years?"
3. "Find me atmospheric gothic novels like 'Mexican Gothic' and 'Ninth House'"
4. "What are the most acclaimed debut novels from this year?"

Contemporary Issues:
1. "Suggest books about climate change that aren't too depressing"
2. "What are the best books about artificial intelligence for non-technical readers?"
3. "Recommend memoirs about immigrant experiences"
4. "Find me books about mental health with hopeful endings"

Book Club Selections:
1. "What are good book club picks that spark discussion?"
2. "Suggest literary fiction under 350 pages"
3. "Find thought-provoking novels that tackle current social issues"
4. "Recommend books with multiple perspectives/narratives"

Upcoming Releases:
1. "What are the most anticipated literary releases next month?"
2. "Show me upcoming releases from my favorite authors"
3. "What debut novels are getting buzz this season?"
4. "List upcoming books being adapted for screen"
"""
