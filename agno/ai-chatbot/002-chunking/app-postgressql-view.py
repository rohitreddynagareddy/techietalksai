import streamlit as st
from sqlalchemy import create_engine, inspect, MetaData, Table
import pandas as pd
import json
from typing import Dict, Any
from sqlalchemy import text
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = "postgresql+psycopg://ai:ai@pgvector:5432/ai"

def get_engine():
    return create_engine(DB_URL)

def get_tables(engine) -> list:
    inspector = inspect(engine)
    return sorted(inspector.get_table_names())

def safe_fetch_data(engine, table_name: str, offset: int) -> pd.DataFrame:
    """Safely fetch data with error handling and proper table validation"""
    try:
        # Validate table exists using SQLAlchemy Core
        metadata = MetaData()
        metadata.reflect(bind=engine, only=[table_name])
        
        if table_name not in metadata.tables:
            st.error(f"Table '{table_name}' does not exist!")
            return pd.DataFrame()

        # Use SQLAlchemy Core for safe query construction
        table = metadata.tables[table_name]
        
        with engine.connect() as connection:
            query = table.select().limit(10).offset(offset)
            return pd.read_sql_query(query, connection)
            
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()

def display_row(row: Dict[str, Any]):
    st.subheader(f"Chunk Details")
    for key, value in row.items():
        st.write(f"**{key}:**")
        try:
            json_value = json.loads(value)
            st.json(json_value)
        except (TypeError, json.JSONDecodeError):
            st.write(value)
    st.markdown("---")

def truncate_content(content, max_length=70):
    """Truncate content for button display"""
    if pd.isna(content):
        return "[empty]"
    content_str = str(content)
    return (content_str[:max_length] + '...') if len(content_str) > max_length else content_str

def delete_table(engine, table_name: str):
    """Safely delete a table from the database"""
    try:
        with engine.connect() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            connection.commit()
        return True
    except Exception as e:
        st.error(f"Error deleting table: {str(e)}")
        return False

# class SearchType(str, Enum):
#     vector = "vector"
#     keyword = "keyword"
#     hybrid = "hybrid"
def similarity_search(table_name: str, query: str, top_k: int = 5):
	knowledge_base = PDFUrlKnowledgeBase(
	    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
	    vector_db=PgVector(table_name=table_name, db_url=DB_URL),
	    # chunking_strategy=SemanticChunking(),
	    search_method=SearchType.hybrid
	    # search_method=SearchType.keyword
	    # search_method=SearchType.vector
	)
	return knowledge_base.search(query)

def display_search_result(result):
    """Display individual search result with metadata"""
    try:
        # Handle Document object directly
        # st.subheader(f"Score: {getattr(result, 'similarity', 0):.4f}")
        
        logger.info("DISPLAY SEARCH RESULTS")
        logger.info(result)


        # Display core content
        if hasattr(result, 'content'):
            st.write("**Content:**")
            st.write(result.content)
        
        # Display metadata
        if hasattr(result, 'meta_data'):
            st.write("**Metadata:**")
            st.json(result.meta_data)
            
        # Display other important fields
        for field in ['id', 'name', 'page', 'chunk', 'embedding','embedder']:
            if hasattr(result, field):
                if field == "embedding":
                  st.write(f"**{field.capitalize()}:**")
                  em = getattr(result, field)
                  st.write(em)
                else:
                  st.write(f"**{field.capitalize()}:** {getattr(result, field)}")
        
        # Show usage stats if available
        if hasattr(result, 'usage'):
            st.write("**Usage:**")
            st.json(result.usage)
            
        st.markdown("---")
        
    except Exception as e:
        st.error(f"Error displaying result: {str(e)}")

def display_search_result2(result):
    """Display individual search result with metadata"""
    try:
        # Convert result to dict
        result_dict = dict(result._asdict())
        logger.info("===== RESULT =====")
        logger.info(result)
        logger.info("===== RESULT DICT =====")
        logger.info(result_dict)
        
        
        # Display main content
        # st.subheader(f"Similarity Score: {result_dict.get('similarity', 0):.4f}")
        
        # Show available fields
        for key, value in result_dict.items():
            if key in ['embedding', 'similarity']:
                continue
                
            st.write(f"**{key.capitalize()}:**")
            
            # Special handling for different field types
            if key == 'meta_data' and isinstance(value, dict):
                st.json(value)
            elif key == 'content':
                st.write(value)
            else:
                try:
                    json_value = json.loads(value)
                    st.json(json_value)
                except (TypeError, json.JSONDecodeError):
                    st.write(value)
        
        st.markdown("---")
    except Exception as e:
        st.error(f"Error displaying result: {str(e)}")


def main():
    st.set_page_config(page_title="Chunking Explorer", layout="wide")
    st.title("Schogini's Chunking Explorer")

    engine = get_engine()
    
    # Initialize session state
    st.session_state.setdefault('selected_table', None)
    st.session_state.setdefault('page', 0)
    st.session_state.setdefault('selected_row', None)

    # Sidebar - Table selection
    st.sidebar.header("Database Tables")
    tables = get_tables(engine)
    

    # Sidebar - Table selection
    st.sidebar.header("Database Tables")
    
    # Display tables as clickable buttons with fresh list
    current_tables = get_tables(engine)
    
    # Display "Clear Selection" button if a table is selected
    if st.session_state.selected_table:
        if st.sidebar.button("❌ Clear Selection"):
            st.session_state.selected_table = None
            st.session_state.page = 0
            st.session_state.selected_row = None
            st.rerun()
    
    # Display tables with delete buttons
    for table in current_tables:
        cols = st.sidebar.columns([4, 1])
        
        # Table selection button
        btn_style = "" if table != st.session_state.selected_table else "✅ "
        if cols[0].button(
            f"{btn_style}{table}",
            key=f"tbl_{table}",
            use_container_width=True
        ):
            if st.session_state.selected_table != table:
                st.session_state.selected_table = table
                st.session_state.page = 0
                st.session_state.selected_row = None
                st.rerun()
        
        # Delete button
        if cols[1].button(
            "🗑️",
            key=f"del_{table}",
            help=f"Delete table {table}",
            type="secondary",
            use_container_width=True
        ):
            st.session_state.pending_delete = table

    # Delete confirmation dialog
    if 'pending_delete' in st.session_state:
        table_to_delete = st.session_state.pending_delete
        st.warning(f"Are you sure you want to delete table '{table_to_delete}'?")
        col1, col2, col3 = st.columns([1, 1, 3])
        
        if col1.button("✅ Confirm Delete"):
            if delete_table(engine, table_to_delete):
                st.success(f"Table '{table_to_delete}' deleted successfully!")
                # Reset selection if deleted table was selected
                if st.session_state.selected_table == table_to_delete:
                    st.session_state.selected_table = None
                    st.session_state.page = 0
                    st.session_state.selected_row = None
                del st.session_state.pending_delete
                st.rerun()
        
        if col2.button("❌ Cancel"):
            del st.session_state.pending_delete
            st.rerun()
    # Main content area
    if st.session_state.selected_table:
        current_tables = get_tables(engine)
        
        # Validate selected table still exists
        if st.session_state.selected_table not in current_tables:
            st.error("Selected table no longer exists!")
            st.session_state.selected_table = None
            st.session_state.page = 0
            st.session_state.selected_row = None
            st.rerun()
            
        # st.header(f"Table: {st.session_state.selected_table}")

    # Main content area
    if st.session_state.selected_table:
        # ... (Keep existing table validation code)

        st.header(f"Table: {st.session_state.selected_table}")
        
        # Add similarity search section
        st.subheader("Hybrid Similarity Search")
        search_query = st.text_input(
            "Enter search query:", 
            key="search_query",
            placeholder="Find similar content..."
        )
        
        if search_query:
            results = []
            with st.spinner("Searching similar content..."):
                results = similarity_search(
                    st.session_state.selected_table, 
                    search_query
                )
                logger.info("SIMILAITY")
                logger.info(results)
                if results:
                    st.subheader(f"Top {len(results)} Similar Results")
                    for idx, result in enumerate(results, 1):
                        a=truncate_content(result.content)
                        with st.expander(f"Result #{idx}: {a}"):
                            display_search_result(result)
                else:
                    st.info("No similar results found")
            
            st.markdown("---")


        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Previous 10") and st.session_state.page > 0:
                st.session_state.page -= 1
                st.session_state.selected_row = None
        with col3:
            if st.button("Next 10"):
                st.session_state.page += 1
                st.session_state.selected_row = None

        # Fetch and display data
        df = safe_fetch_data(engine, st.session_state.selected_table, st.session_state.page * 10)
        
        if not df.empty:
            # Display clickable rows with content preview
            for index, row in df.iterrows():
                row_number = index + 1 + (st.session_state.page * 10)
                
                # Create button label with content preview if available
                if 'content' in df.columns:
                    content = row.get('content', '')
                    content_len = len(content)
                    preview = truncate_content(content)
                    btn_label = f"Chunk {row_number}: [{content_len}]: {preview}"
                else:
                    btn_label = f"Chunk {row_number}"
                
                # Use a unique key combining page and index
                if st.button(btn_label, key=f"row_{st.session_state.page}_{index}"):
                    st.session_state.selected_row = row.to_dict()

            # Display selected row details
            if st.session_state.selected_row:
                display_row(st.session_state.selected_row)
        else:
            st.info("No rows found in this table")
    else:
        st.info("Select a table from the sidebar to view its contents")

    # Footer section
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1E2D41;
        color: #555;
        text-align: center;
        padding: 10px;
        font-size: 0.8em;
    }
    </style>
    <div class="footer">
        Built by <a href="https://www.schogini.com" target="_blank">Schogini Systems Private Limited</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()