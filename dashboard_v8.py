import streamlit as st
import pandas as pd
import numpy as np
import re

# === Page config ===
st.set_page_config(page_title="Info Arb Tracker", layout="wide") 
st.title("Info Arb Tracker") # Changed from "Info Arb Tracker: Clean Financial Dashboard"

# --- Load dataset ---
@st.cache_data 
def load_data():
    """Attempts to load press_releasesV2.csv. Creates an empty DataFrame if not found."""
    # --- UPDATED FILENAME TO V2 ---
    FILE_NAME = "press_releasesV2.csv"
    
    df = pd.DataFrame()
    try:
        # --- USE NEW FILE_NAME VARIABLE ---
        df = pd.read_csv(FILE_NAME)
    except FileNotFoundError:
        # --- UPDATED ERROR MESSAGE ---
        st.error(f"{FILE_NAME} not found. Please ensure the file is present in the application directory.")
        return pd.DataFrame() 

    # --- Robust Column Header Cleaning ---
    def clean_header(col):
        """Removes quotes, line breaks, non-breaking spaces, and normalizes spacing."""
        col = col.replace('\n', ' ').replace('\xa0', ' ').replace('\u00a0', ' ').strip().replace('"', '')
        col = re.sub(r'\s+', ' ', col).strip()
        return col

    if not df.empty:
        df.columns = [clean_header(col) for col in df.columns]
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        # --- FIX: DATA CLEANING FOR NUMERIC RATIO COLUMNS ---
        ratio_cols = ['EV/TTM SALES', 'TTM PE', 'FY + 1 PE', 'FY + 2 PE']
        
        for col in ratio_cols:
            if col in df.columns:
                # 1. Convert to string, handle missing values early
                df[col] = df[col].astype(str)
                
                # 2. Replace 'None', 'N/A', and empty strings with NaN
                df[col] = df[col].replace(['None', 'N/A', 'nan', ''], np.nan)
                
                # 3. Remove 'x' suffix and other non-numeric characters (except period and sign)
                # This handles '1.67x' -> '1.67'
                df[col] = df[col].str.replace(r'[^\d\.\-]', '', regex=True)
                
                # 4. Convert to numeric (float), coercing any remaining errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# --- Text Cleaning Function (Unchanged) ---
def clean_summary_text(summary_text):
    """
    Cleans up run-on text, fixes list formatting, and escapes dollar signs for Streamlit Markdown.
    """
    if not isinstance(summary_text, str) or summary_text.strip() in ['N/A', '']:
        return summary_text
        
    summary_text = re.sub(r'\s{2,}', ' ', summary_text).strip()
    
    # Targeted string and regex replacements for common run-on issues
    replacements = {
        'plusteleradiology': 'plus teleradiology',
        'contractwitha': 'contract with a', 
        'Level1': 'Level 1', 
        'plusin': 'plus in',
        'Microgrids4AI': 'Microgrids 4 AI', 
    }
    for old, new in replacements.items():
        summary_text = summary_text.replace(old, new)

    summary_text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', summary_text)
    summary_text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', summary_text)
    summary_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', summary_text)
    summary_text = re.sub(r'(\w|[\d\.])([\s]*)([\$%\+\,])', r'\1 \3', summary_text)
    summary_text = re.sub(r'([^\w\s\-])(?=[\w])', r'\1 ', summary_text)
    
    # Fix run-on numbered lists
    summary_text = re.sub(r'(\S)\s+(\d\))', r'\1\n\2', summary_text)
    
    # Escape literal dollar signs for Streamlit Markdown
    summary_text = summary_text.replace('$', r'\$')
    
    summary_text = re.sub(r'\s{2,}', ' ', summary_text).strip()
    
    return summary_text

df = load_data()

# --- NEW ROBUSTNESS CHECK: Stop if data is empty ---
if df.empty:
    st.error("Cannot proceed. The required data file is missing or failed to load. Please ensure 'press_releasesV2.csv' is in the application directory.")
    st.stop()
# --- END NEW ROBUSTNESS CHECK ---

# --- Filtering Section ---
df_filtered = df.copy() 

st.markdown("### Filter Results")
col1, col2, col3, col4 = st.columns(4)

# 1. Ticker Filter
with col1:
    # This line is now safe because we checked if df is empty above
    ticker_options = ['All'] + sorted(df_filtered['Ticker'].dropna().unique().tolist())
    selected_tickers = st.multiselect("Ticker", ticker_options, default='All')
    if 'All' not in selected_tickers:
        df_filtered = df_filtered[df_filtered['Ticker'].isin(selected_tickers)]

# 2. Industry Filter
with col2:
    industry_options = ['All'] + sorted(df_filtered['Industry'].dropna().unique().tolist())
    selected_industries = st.multiselect("Industry", industry_options, default='All')
    if 'All' not in selected_industries:
        df_filtered = df_filtered[df_filtered['Industry'].isin(selected_industries)]

# 3. Week Filter
with col3:
    if 'Week' in df_filtered.columns:
        df_filtered['Week_str'] = df_filtered['Week'].fillna(np.nan).astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', 'N/A')
        week_options = ['All'] + sorted(df_filtered['Week_str'].unique().tolist())
        selected_weeks = st.multiselect("Week #", week_options, default='All')
        
        if 'All' not in selected_weeks:
            df_filtered = df_filtered[df_filtered['Week_str'].isin(selected_weeks)]
        
        df_filtered = df_filtered.drop(columns=['Week_str'], errors='ignore')

# 4. Type of Press Release Filter
with col4:
    pr_type_options = ['All'] + sorted(df_filtered['Type of Press Release'].dropna().unique().tolist())
    selected_pr_types = st.multiselect("PR Type", pr_type_options, default='All')
    if 'All' not in selected_pr_types:
        df_filtered = df_filtered[df_filtered['Type of Press Release'].isin(selected_pr_types)]
    
filtered = df_filtered.copy()

# --- Sorting (Default: Date Descending) ---
if 'Date' in filtered.columns:
    filtered = filtered.sort_values(by="Date", ascending=False, na_position='last')

# -----------------------------------------------------------------------------------
## --- Main Table View ---

# Define columns for the main table
main_table_columns = [
    'Ticker', 'Industry', 'Week', 'Date', 
    'Title', 
    'Type of Press Release', 
    'Currency', 
    'Market Cap', 
    'EV/TTM SALES', 'TTM PE', 'FY + 1 PE', 'FY + 2 PE', 
]

display_df = filtered[[col for col in main_table_columns if col in filtered.columns]].reset_index(drop=True)

if 'Date' in display_df.columns:
    display_df['Date'] = display_df['Date'].dt.date 

# Configuration for table columns
# The format is set here, ensuring the underlying data is numerical first.
ratio_cols_config = {
    "EV/TTM SALES": st.column_config.NumberColumn("EV/TTM SALES", help="Enterprise Value / Trailing Twelve Months Sales", width="small", disabled=True, format="%.2fX"),
    "TTM PE": st.column_config.NumberColumn("TTM PE", help="Trailing Twelve Months P/E Ratio", width="small", disabled=True, format="%.2fX"),
    "FY + 1 PE": st.column_config.NumberColumn("FY + 1 PE", help="Forward P/E Ratio (Next Fiscal Year)", width="small", disabled=True, format="%.2fX"),
    "FY + 2 PE": st.column_config.NumberColumn("FY + 2 PE", help="Forward P/E Ratio (Two Fiscal Years Out)", width="small", disabled=True, format="%.2fX"),
}

full_column_config = {
    "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
    "Week": st.column_config.TextColumn("Week #", help="Fiscal Week Number", width="small", disabled=True), 
    "Title": st.column_config.TextColumn("Title", help="Press Release Title", width="large", disabled=True), 
    **ratio_cols_config 
}

# Display the main data table
st.dataframe(
    display_df,
    key='main_table_select', 
    on_select="rerun", 
    selection_mode="multi-row",
    
    width="stretch", 
    column_config=full_column_config,
    hide_index=True,
)

st.markdown("---")


# =========================================================================
# 2. Details Display Section
# =========================================================================
if 'main_table_select' in st.session_state and st.session_state.main_table_select.selection.rows:
    
    details_container = st.container(height=800) 
    
    with details_container:
        selected_indices = st.session_state.main_table_select.selection.rows
        
        temp_filtered_df = filtered.reset_index(drop=True)
        selected_rows_data = temp_filtered_df.iloc[selected_indices]
        
        st.subheader(f"Details for Selected Press Release(s) ({selected_rows_data['Ticker'].str.cat(sep=', ')})")

        stock_data_columns = [
            'Ticker', 'Industry', 'Date', 'Week', 
            'Type of Press Release', 'Currency', 
            'Market Cap', 'Net Debt', 'EV', 'S/O', 'Price', 'TTM Sales', 'TTM EPS', 
            'FY + 1 EPS', 'FY + 2 EPS', 'EV/TTM SALES', 'TTM PE', 'FY + 1 PE', 'FY + 2 PE', 
        ]
        
        for index, row in selected_rows_data.iterrows():
            
            ticker = row['Ticker']
            pr_type = row.get('Type of Press Release', 'PR')
            date_str = row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else 'N/A'
            
            expander_title = f"**{ticker}** - {pr_type} on {date_str}"
            
            with st.expander(expander_title, expanded=True): 
                
                full_pr_title = row.get('Title', row.get('Type of Press Release', 'N/A'))
                
                # Using markdown for a prominent subtitle effect 
                st.markdown(f'<h2>{full_pr_title}</h2>', unsafe_allow_html=True)
                
                # --- Stock Data Section ---
                st.markdown("#### Stock Data")
                
                # Create a copy of the row for the Stock Data table display
                stock_data_display_row = row[[col for col in stock_data_columns if col in row]].to_frame().T
                
                if 'Date' in stock_data_display_row.columns:
                     stock_data_display_row['Date'] = stock_data_display_row['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A')
                
                # Apply the 'X' format to the display data for clarity.
                for col in ['EV/TTM SALES', 'TTM PE', 'FY + 1 PE', 'FY + 2 PE']:
                    if col in stock_data_display_row.columns and pd.notna(stock_data_display_row[col].iloc[0]):
                        stock_data_display_row[col] = f"{stock_data_display_row[col].iloc[0]:.2f}x"
                    elif col in stock_data_display_row.columns:
                        stock_data_display_row[col] = "None"


                st.dataframe(
                    stock_data_display_row,
                    width='stretch', 
                    hide_index=True
                )
                
                # --- Link ---
                link_url = row.get('Links') 
                
                if pd.notna(link_url) and str(link_url).strip():
                    st.markdown(f"**Full Press Release:** [View Source]({link_url})", unsafe_allow_html=True)
                    st.markdown('<hr style="margin: 5px 0 10px 0; border: 0.5px solid rgba(255, 255, 255, 0.1);">', unsafe_allow_html=True)
                
                else:
                    st.info("Link Missing: URL is missing or blank for this press release.")

                # 1. Display Summary
                st.markdown("#### Summary")
                summary_text = row.get('Summary', 'N/A')
                if summary_text != 'N/A':
                    summary_text = clean_summary_text(summary_text)
                st.info(summary_text) # Uses blue/gray color

                # 2. Positive and Negative Insights
                col_pos, col_neg = st.columns(2)
                
                with col_pos:
                    st.markdown("#### Positive Insights")
                    positive_insights = row.get('Positive Insights', 'N/A')
                    if isinstance(positive_insights, str):
                        positive_insights = clean_summary_text(positive_insights)
                    st.success(positive_insights) # Uses green color

                with col_neg:
                    st.markdown("#### Negative Insights")
                    negative_insights = row.get('Negative Insights', 'N/A')
                    if isinstance(negative_insights, str):
                        negative_insights = clean_summary_text(negative_insights)
                    
                    st.error(negative_insights) # Uses red/pink color
            
else:
    st.markdown("---")
    st.info("Select one or more rows in the table above to view the Summary and Insights.")
