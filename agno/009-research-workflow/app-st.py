import os
import json
import streamlit as st
from datetime import datetime
from textwrap import dedent
from typing import Dict, Iterator, Optional
from pathlib import Path
from agno.agent import Agent, RunResponse
from agno.storage.workflow.sqlite import SqliteWorkflowStorage
from agno.utils.log import logger
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from pydantic import BaseModel, Field
from agno.workflow import RunEvent, RunResponse, Workflow

from agno.models.openai import OpenAIChat
from agno.models.xai import xAI
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini
from agno.models.groq import Groq
# --------- LOAD API KEY ---------
import os
# Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
xai_api_key = os.getenv("XAI_API_KEY")
if not xai_api_key:
    st.error("xAI API key not found. Please set the XAI_API_KEY environment variable.")
    st.stop()

# --------------- SETUP ---------------
cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

# --------------- TITLE AND INFO SECTION -------------------
st.title("🎓 Advanced Research Workflow")
st.write("Academic-grade research assistant with smart caching and source analysis")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Research Topics:")
    st.markdown("""
    - Quantum computing breakthroughs
    - CRISPR gene editing advances
    - Fusion energy developments
    - AI ethics frameworks
    - Neuroplasticity research
    """)
    st.markdown("---")
    
    # Cache controls
    st.subheader("Workflow Settings")
    use_search_cache = st.checkbox("Use search cache", True)
    use_scrape_cache = st.checkbox("Use content cache", True)
    use_cached_report = st.checkbox("Use report cache", True)
    stream = st.checkbox("Stream research process", True)
    
    # Stats
    if "report_count" not in st.session_state:
        st.session_state.report_count = 0
    st.caption(f"Reports generated: {st.session_state.report_count}")

# --------------- MODEL DEFINITIONS -------------------

class Article(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(
        ..., description="Summary of the article if available."
    )


class SearchResults(BaseModel):
    articles: list[Article]


class ScrapedArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(
        ..., description="Summary of the article if available."
    )
    content: Optional[str] = Field(
        ...,
        description="Content of the in markdown format if available. Return None if the content is not available or does not make sense.",
    )


class ResearchReportGenerator(Workflow):
    description: str = dedent("""\
    Generate comprehensive research reports that combine academic rigor
    with engaging storytelling. This workflow orchestrates multiple AI agents to search, analyze,
    and synthesize information from diverse sources into well-structured reports.
    """)

    web_searcher: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[DuckDuckGoTools()],
        description=dedent("""\
        You are ResearchBot-X, an expert at discovering and evaluating academic and scientific sources.\
        """),
        instructions=dedent("""\
        You're a meticulous research assistant with expertise in source evaluation! 🔍
        Search for 10-15 sources and identify the 5-7 most authoritative and relevant ones.
        Prioritize:
        - Peer-reviewed articles and academic publications
        - Recent developments from reputable institutions
        - Authoritative news sources and expert commentary
        - Diverse perspectives from recognized experts
        Avoid opinion pieces and non-authoritative sources.\
        """),
        response_model=SearchResults,
        structured_outputs=True,
    )

    article_scraper: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[Newspaper4kTools()],
        description=dedent("""\
        You are ContentBot-X, an expert at extracting and structuring academic content.\
        """),
        instructions=dedent("""\
        You're a precise content curator with attention to academic detail! 📚
        When processing content:
           - Extract content from the article
           - Preserve academic citations and references
           - Maintain technical accuracy in terminology
           - Structure content logically with clear sections
           - Extract key findings and methodology details
           - Handle paywalled content gracefully
        Format everything in clean markdown for optimal readability.\
        """),
        response_model=ScrapedArticle,
        structured_outputs=True,
    )

    writer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        description=dedent("""\
        You are Professor X-2000, a distinguished AI research scientist combining academic rigor with engaging narrative style.\
        """),
        instructions=dedent("""\
        Channel the expertise of a world-class academic researcher!
        🎯 Analysis Phase:
          - Evaluate source credibility and relevance
          - Cross-reference findings across sources
          - Identify key themes and breakthroughs
        💡 Synthesis Phase:
          - Develop a coherent narrative framework
          - Connect disparate findings
          - Highlight contradictions or gaps
        ✍️ Writing Phase:
          - Begin with an engaging executive summary, hook the reader
          - Present complex ideas clearly
          - Support all claims with citations
          - Balance depth with accessibility
          - Maintain academic tone while ensuring readability
          - End with implications and future directions\
        """),
        expected_output=dedent("""\
        # {Compelling Academic Title}

        ## Executive Summary
        {Concise overview of key findings and significance}

        ## Introduction
        {Research context and background}
        {Current state of the field}

        ## Methodology
        {Search and analysis approach}
        {Source evaluation criteria}

        ## Key Findings
        {Major discoveries and developments}
        {Supporting evidence and analysis}
        {Contrasting viewpoints}

        ## Analysis
        {Critical evaluation of findings}
        {Integration of multiple perspectives}
        {Identification of patterns and trends}

        ## Implications
        {Academic and practical significance}
        {Future research directions}
        {Potential applications}

        ## Key Takeaways
        - {Critical finding 1}
        - {Critical finding 2}
        - {Critical finding 3}

        ## References
        {Properly formatted academic citations}

        ---
        Report generated by Professor X-2000
        Advanced Research Division
        Date: {current_date}\
        """),
        markdown=True,
    )

    def run(
        self,
        topic: str,
        use_search_cache: bool = True,
        use_scrape_cache: bool = True,
        use_cached_report: bool = True,
    ) -> Iterator[RunResponse]:
        """
        Generate a comprehensive news report on a given topic.

        This function orchestrates a workflow to search for articles, scrape their content,
        and generate a final report. It utilizes caching mechanisms to optimize performance.

        Args:
            topic (str): The topic for which to generate the news report.
            use_search_cache (bool, optional): Whether to use cached search results. Defaults to True.
            use_scrape_cache (bool, optional): Whether to use cached scraped articles. Defaults to True.
            use_cached_report (bool, optional): Whether to return a previously generated report on the same topic. Defaults to False.

        Returns:
            Iterator[RunResponse]: An stream of objects containing the generated report or status information.

        Steps:
        1. Check for a cached report if use_cached_report is True.
        2. Search the web for articles on the topic:
            - Use cached search results if available and use_search_cache is True.
            - Otherwise, perform a new web search.
        3. Scrape the content of each article:
            - Use cached scraped articles if available and use_scrape_cache is True.
            - Scrape new articles that aren't in the cache.
        4. Generate the final report using the scraped article contents.

        The function utilizes the `session_state` to store and retrieve cached data.
        """
        logger.info(f"Generating a report on: {topic}")
        expander.write(f"Generating a report on: {topic}")

        # Use the cached report if use_cached_report is True
        if use_cached_report:
            cached_report = self.get_cached_report(topic)
            if cached_report:
                yield RunResponse(
                    content=cached_report, event=RunEvent.workflow_completed
                )
                return

        # Search the web for articles on the topic
        search_results: Optional[SearchResults] = self.get_search_results(
            topic, use_search_cache
        )
        # If no search_results are found for the topic, end the workflow
        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Sorry, could not find any articles on the topic: {topic}",
            )
            return

        # Scrape the search results
        scraped_articles: Dict[str, ScrapedArticle] = self.scrape_articles(
            search_results, use_scrape_cache
        )

        # Write a research report
        yield from self.write_research_report(topic, scraped_articles)

    def get_cached_report(self, topic: str) -> Optional[str]:
        logger.info("Checking if cached report exists")
        expander.write("Checking if cached report exists")
        return self.session_state.get("reports", {}).get(topic)

    def add_report_to_cache(self, topic: str, report: str):
        logger.info(f"Saving report for topic: {topic}")
        expander.write(f"Saving report for topic: {topic}")
        self.session_state.setdefault("reports", {})
        self.session_state["reports"][topic] = report
        # Save the report to the storage
        self.write_to_storage()

    def get_cached_search_results(self, topic: str) -> Optional[SearchResults]:
        logger.info("Checking if cached search results exist")
        expander.write("Checking if cached search results exist")
        return self.session_state.get("search_results", {}).get(topic)

    def add_search_results_to_cache(self, topic: str, search_results: SearchResults):
        logger.info(f"Saving search results for topic: {topic}")
        expander.write(f"Saving search results for topic: {topic}")
        self.session_state.setdefault("search_results", {})
        self.session_state["search_results"][topic] = search_results.model_dump()
        # Save the search results to the storage
        self.write_to_storage()

    def get_cached_scraped_articles(
        self, topic: str
    ) -> Optional[Dict[str, ScrapedArticle]]:
        logger.info("Checking if cached scraped articles exist")
        expander.write("Checking if cached scraped articles exist")
        return self.session_state.get("scraped_articles", {}).get(topic)

    def add_scraped_articles_to_cache(
        self, topic: str, scraped_articles: Dict[str, ScrapedArticle]
    ):
        logger.info(f"Saving scraped articles for topic: {topic}")
        expander.write(f"Saving scraped articles for topic: {topic}")
        self.session_state.setdefault("scraped_articles", {})
        self.session_state["scraped_articles"][topic] = scraped_articles
        # Save the scraped articles to the storage
        self.write_to_storage()

    def get_search_results(
        self, topic: str, use_search_cache: bool, num_attempts: int = 3
    ) -> Optional[SearchResults]:
        # Get cached search_results from the session state if use_search_cache is True
        if use_search_cache:
            try:
                search_results_from_cache = self.get_cached_search_results(topic)
                if search_results_from_cache is not None:
                    search_results = SearchResults.model_validate(
                        search_results_from_cache
                    )
                    logger.info(
                        f"Found {len(search_results.articles)} articles in cache."
                    )
                    expander.write(
                        f"Found {len(search_results.articles)} articles in cache."
                    )
                    return search_results
            except Exception as e:
                logger.warning(f"Could not read search results from cache: {e}")
                expander.write(f"Could not read search results from cache: {e}")

        # If there are no cached search_results, use the web_searcher to find the latest articles
        for attempt in range(num_attempts):
            try:
                searcher_response: RunResponse = self.web_searcher.run(topic)
                if (
                    searcher_response is not None
                    and searcher_response.content is not None
                    and isinstance(searcher_response.content, SearchResults)
                ):
                    article_count = len(searcher_response.content.articles)
                    logger.info(
                        f"Found {article_count} articles on attempt {attempt + 1}"
                    )
                    expander.write(
                        f"Found {article_count} articles on attempt {attempt + 1}"
                    )
                    # Cache the search results
                    self.add_search_results_to_cache(topic, searcher_response.content)
                    return searcher_response.content
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}/{num_attempts} failed: Invalid response type"
                    )
                    expander.write(
                        f"Attempt {attempt + 1}/{num_attempts} failed: Invalid response type"
                    )
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")
                expander.write(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")

        logger.error(f"Failed to get search results after {num_attempts} attempts")
        expander.write(f"Failed to get search results after {num_attempts} attempts")
        return None

    def scrape_articles(
        self, search_results: SearchResults, use_scrape_cache: bool
    ) -> Dict[str, ScrapedArticle]:
        scraped_articles: Dict[str, ScrapedArticle] = {}

        # Get cached scraped_articles from the session state if use_scrape_cache is True
        if use_scrape_cache:
            try:
                scraped_articles_from_cache = self.get_cached_scraped_articles(topic)
                if scraped_articles_from_cache is not None:
                    scraped_articles = scraped_articles_from_cache
                    logger.info(
                        f"Found {len(scraped_articles)} scraped articles in cache."
                    )
                    expander.write(
                        f"Found {len(scraped_articles)} scraped articles in cache."
                    )
                    return scraped_articles
            except Exception as e:
                expander.write(f"Could not read scraped articles from cache: {e}")
                logger.warning(f"Could not read scraped articles from cache: {e}")

        # Scrape the articles that are not in the cache
        for article in search_results.articles:
            if article.url in scraped_articles:
                logger.info(f"Found scraped article in cache: {article.url}")
                expander.write(f"Found scraped article in cache: {article.url}")
                continue

            article_scraper_response: RunResponse = self.article_scraper.run(
                article.url
            )
            if (
                article_scraper_response is not None
                and article_scraper_response.content is not None
                and isinstance(article_scraper_response.content, ScrapedArticle)
            ):
                scraped_articles[article_scraper_response.content.url] = (
                    article_scraper_response.content
                )
                logger.info(f"Scraped article: {article_scraper_response.content.url}")
                expander.write(f"Scraped article: {article_scraper_response.content.url}")

        # Save the scraped articles in the session state
        self.add_scraped_articles_to_cache(topic, scraped_articles)
        return scraped_articles

    def write_research_report(
        self, topic: str, scraped_articles: Dict[str, ScrapedArticle]
    ) -> Iterator[RunResponse]:
        logger.info("Writing research report")
        expander.write("Writing research report")
        # Prepare the input for the writer
        writer_input = {
            "topic": topic,
            "articles": [v.model_dump() for v in scraped_articles.values()],
        }
        # Run the writer and yield the response
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)
        # Save the research report in the cache
        self.add_report_to_cache(topic, self.writer.run_response.content)



# --------------- USER INTERFACE -------------------
topic = st.text_input("Enter research topic:", placeholder="e.g., 'Advances in quantum computing'")

# Define the expander at a specific location in your layout
expander = st.expander("Research Details", expanded=True)
# Immediately add some content

# # Execute the workflow with caching enabled
# report_stream: Iterator[RunResponse] = generate_research_report.run(
#     topic=topic,
#     use_search_cache=True,
#     use_scrape_cache=True,
#     use_cached_report=True,
# )

# # Print the response
# pprint_run_response(report_stream, markdown=True)


if topic:
    expander.write(f"Researching: {topic}")
    # Convert the topic to a URL-safe string for use in session_id
    url_safe_topic = topic.lower().replace(" ", "-")
    # Initialize the news report generator workflow
    generate_research_report = ResearchReportGenerator(
        session_id=f"generate-report-on-{url_safe_topic}",
        storage=SqliteWorkflowStorage(
            table_name="generate_research_report_workflow",
            db_file="tmp/workflows.db",
        ),
    )

    # Execute the workflow with caching enabled
    report_stream: Iterator[RunResponse] = generate_research_report.run(
        topic=topic,
        use_search_cache=use_search_cache,
        use_scrape_cache=use_scrape_cache,
        use_cached_report=use_cached_report,
        # use_search_cache=True,
        # use_scrape_cache=True,
        # use_cached_report=True,
    )
            
    with st.spinner("🔬 Conducting deep research analysis..."):
        report_container = st.empty()
        full_report = ""
        
        # for report_chunk in run_research_workflow(topic):
        for report_chunk in report_stream:
            full_report += report_chunk.content
            if stream:
                report_container.markdown(full_report + "▌")
            else:
                report_container.markdown(full_report)
        
        report_container.markdown(full_report)
        
        # Save report
        filename = f"research_{datetime.now():%Y%m%d%H%M%S}.md"
        with open(tmp / filename, "w") as f:
            f.write(full_report)
        
        st.download_button(
            label="Download Full Report",
            data=full_report,
            file_name=filename,
            mime="text/markdown"
        )

# --------------- FOOTER -------------------
st.markdown("---")
st.caption("""
**Workflow Features:**
- Smart caching system 🗄️
- Academic source validation 📚
- Multi-stage quality control ✅
- Technical content extraction 🔍
- Markdown report export 📄
""")

# Hidden dependency note
st.markdown("<!--- Run `pip install openai duckduckgo-search newspaper4k lxml_html_clean sqlalchemy agno` -->", unsafe_allow_html=True)