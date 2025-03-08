import streamlit as st
import sqlite3
import pandas as pd
import os
import json
from pandas.api.types import is_string_dtype

st.title("Schogini's Agent Memory Explorer")

DB_DIR = "/app/tmp"
os.makedirs(DB_DIR, exist_ok=True)

# File upload section
st.subheader("Upload Database")
uploaded_file = st.file_uploader("Choose SQLite file", type=["db", "sqlite"])
if uploaded_file is not None:
    save_path = os.path.join(DB_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success(f"Uploaded {uploaded_file.name}")
    st.rerun()

# List database files
db_files = [f for f in os.listdir(DB_DIR) if f.endswith((".db", ".sqlite"))]

if not db_files:
    st.warning("No databases found in directory")
else:
    # File management
    st.subheader("Manage Databases")
    selected_file = st.selectbox("Select database", db_files)
    file_path = os.path.join(DB_DIR, selected_file)
    
    # File controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col2:
        if st.button("❌ Delete"):
            try:
                os.remove(file_path)
                st.success(f"Deleted {selected_file}")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")

    # Database viewer
    try:
        conn = sqlite3.connect(file_path)
        
        # Get tables
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
        
        if not tables:
            st.warning("Database contains no tables")
        else:
            selected_table = st.selectbox("Select table", tables)
            df = pd.read_sql(f"SELECT * FROM {selected_table}", conn)
            
            if not df.empty:
                # Add row selection checkboxes
                df_display = df.copy()
                df_display.insert(0, "Select", False)
                
                # Configure columns
                col_config = {
                    "Select": st.column_config.CheckboxColumn(
                        required=True,
                        help="Select row to expand"
                    )
                }
                for col in df.columns:
                    col_config[col] = st.column_config.Column(disabled=True)

                # Show interactive dataframe
                edited_df = st.data_editor(
                    df_display,
                    column_config=col_config,
                    key=f"editor_{selected_table}",
                    hide_index=True,
                    use_container_width=True
                )

                # Handle selected row
                selected_rows = edited_df[edited_df.Select]
                if not selected_rows.empty:
                    selected_row = selected_rows.iloc[0]
                    
                    st.divider()
                    st.subheader("📖 Expanded Row View")
                    
                    for col in df.columns:
                        value = selected_row[col]
                        st.markdown(f"**{col}**")
                        
                        if pd.isna(value):
                            st.write("NULL")
                        elif col == "memory" or col == "agent_data" or col == "session_data":
                        	 st.json(value)
                        elif is_string_dtype(df[col]):
                            try:
                                json_value = json.loads(value)
                                st.json(json_value)
                            except (json.JSONDecodeError, TypeError):
                                st.write(value)
                        else:
                            if col == "memory":
                        	    # st.json("sss")
                        	    st.write("sss")
                            else:
                                st.write(value)
            else:
                st.warning("Selected table is empty")

    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()