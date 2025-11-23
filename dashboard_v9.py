import streamlit as st
import pandas as pd
import numpy as np
import re

# === Page config ===
st.set_page_config(page_title="Info Arb Tracker", layout="wide") 
st.title("Info Arb Tracker")

# --- Load dataset ---
@st.cache_data 
def load_data():
    """Attempts to load press_releasesV2.csv. Creates an empty DataFrame if not found."""
    FILE_NAME = "press_releasesV2.csv"
    df = pd.DataFrame()
    
    # 1. ROBUST FILE READING BLOCK (Encoding/Delimiter/File Not Found)
    try:
        df = pd.read_csv(
            FILE_NAME, 
            encoding='utf-8', # Explicitly set UTF-8
            sep=',',         # Explicitly set comma delimiter
        )
    except FileNotFoundError:
        st.error(f"{FILE_NAME} not found. Please ensure the file is present in the application directory.")
        return pd.DataFrame()
    except UnicodeDecodeError:
        st.warning("UTF-8 failed. Trying 'latin-1' encoding...")
        try:
            df = pd.read_csv(
                FILE_NAME, 
                encoding='latin-1',
                sep=',',
            )
        except Exception as e:
            st.error(f"Failed to load data with latin-1. Error: {e}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load data due to a general error. Check file structure (e.g., delimiter). Error: {e}")
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
            # Explicit Date Parsing: Use the confirmed MM/DD/YYYY format first
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce') 
            except ValueError:
                # If that fails, let pandas infer (as a last resort)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
        # --- FIX: DATA CLEANING FOR NUMERIC RATIO COLUMNS (CORRECTED) ---
        ratio_cols = ['EV/TTM SALES', 'TTM PE', 'FY + 1 PE', 'FY + 2 PE']
        
        for col in ratio_cols:
            if col in df.columns:
                # 1. Fill NaN with a sentinel, convert to string, replace 'None' and empty strings with sentinel.
                df[col] = df[col].fillna('NaN_TEMP').astype(str).replace('None', 'NaN_TEMP').replace('', 'NaN_TEMP')

                # 2. Aggressive regex: Remove anything that isn't a digit, decimal point, or leading minus sign.
                df[col] = df[col].str.replace(r'[^\d\.\-]', '', regex=True)
                
                # 3. Replace the sentinel string with a true numeric NaN
                df[col] = df[col].replace('NaN_TEMP', np.nan)
                
                # 4. Final conversion to numeric. Coerce errors turns any remaining invalid string into NaN.
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
    return df

# --- Text Cleaning Function (General purpose for all Summary/Insights) ---
def clean_summary_text(summary_text):
    """
    General cleanup for run-on text and list formatting issues.
    """
    if not isinstance(summary_text, str) or summary_text.strip() in ['N/A', '', 'None']:
        return 'N/A'
        
    summary_text = re.sub(r'\s{2,}', ' ', summary_text).strip()
    
    # Ensure all list items are on a new line:
    summary_text = re.sub(r'(\S)\s+([1-9]\.)', r'\1\n\2', summary_text)
    
    # Handle run-on word fixes
    replacements = {
        'plusteleradiology': 'plus teleradiology',
        'contractwitha': 'contract with a', 
        'Level1': 'Level 1', 
        'plusin': 'plus in',
        'Microgrids4AI': 'Microgrids 4 AI',
        'millionvs.CAD': ' million vs. CAD ', # Fix for your specific example
    }
    for old, new in replacements.items():
        summary_text = summary_text.replace(old, new)
        
    # Add space between letters and numbers
    summary_text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', summary_text)
    summary_text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', summary_text)
    # Add space between lower and upper case letters (camel case fix)
    summary_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', summary_text)
    
    # Add space before currency/symbols if directly following a word/number
    summary_text = re.sub(r'(\w|[\d\.])([\s]*)([\$%\+\,])', r'\1 \3', summary_text)
    
    # Fix run-on numbered lists (parenthesis style)
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

# ----------------- FIX: FILTER OUT EMPTY ROWS -----------------
if not df.empty:
    df.dropna(subset=['Ticker', 'Title'], how='all', inplace=True)
    df.reset_index(drop=True, inplace=True) 
# ----------------- END FIX -----------------


# --- Filtering Section ---
df_filtered = df.copy() 

st.markdown("### Filter Results")
col1, col2, col3, col4 = st.columns(4)

# 1. Ticker Filter
with col1:
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

# DATE FIX for MAIN TABLE 
if 'Date' in display_df.columns:
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d').fillna('N/A') 

# Configuration for table columns
ratio_cols_config = {
    "EV/TTM SALES": st.column_config.NumberColumn("EV/TTM SALES", help="Enterprise Value / Trailing Twelve Months Sales", width="small", disabled=True, format="%.2fX"),
    "TTM PE": st.column_config.NumberColumn("TTM PE", help="Trailing Twelve Months P/E Ratio", width="small", disabled=True, format="%.2fX"),
    "FY + 1 PE": st.column_config.NumberColumn("FY + 1 PE", help="Forward P/E Ratio (Next Fiscal Year)", width="small", disabled=True, format="%.2fX"),
    "FY + 2 PE": st.column_config.NumberColumn("FY + 2 PE", help="Forward P/E Ratio (Two Fiscal Years Out)", width="small", disabled=True, format="%.2fX"),
}

full_column_config = {
    "Date": st.column_config.TextColumn("Date", help="Date of Press Release", width="small"),
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
        
        if selected_rows_data.empty:
            st.error("No data found for selected rows.")
            st.stop() 

        st.subheader(f"Details for Selected Press Release(s) ({selected_rows_data['Ticker'].str.cat(sep=', ')})")

        # Define the columns for the details stock data table (Price removed)
        stock_data_columns = [
            'Ticker', 'Industry', 'Date', 
            'Type of Press Release', 
            'EV/TTM SALES', 'TTM PE', 'FY + 1 PE', 'FY + 2 PE', 
            'Currency', 
            'Market Cap', 'Net Debt', 'EV', 
            'TTM Sales', 'TTM EPS', 
            'FY + 1 EPS', 'FY + 2 EPS',
            
        ]
        
        FINANCIAL_SUMMARY_COL = 'Financial Summary'
        OPERATIONAL_SUMMARY_COL = 'Operational Summary'
        
        for index, row in selected_rows_data.iterrows():
            
            ticker = row['Ticker']
            pr_type = row.get('Type of Press Release', 'PR')
            
            # 🌟 FIX 1: Retrieve the formatted date string for the expander title directly from the display_df content
            date_str_formatted_from_display = display_df.iloc[index].get('Date', 'N/A')
            
            expander_title = f"**{ticker}** - {pr_type} on {date_str_formatted_from_display}"
            
            with st.expander(expander_title, expanded=True): 
                
                full_pr_title = row.get('Title', row.get('Type of Press Release', 'N/A'))
                st.markdown(f'<h2>{full_pr_title}</h2>', unsafe_allow_html=True)
                
                # --- Stock Data Section ---
                st.markdown("#### Stock Data")
                
                # Filter row to only include the columns we want to display and in the new order
                stock_data_display_row = row[[col for col in stock_data_columns if col in row]].to_frame().T
                
                # 🌟 FIX 2: Ensure the 'Date' column content in this new small DataFrame is a clean string
                if 'Date' in stock_data_display_row.columns:
                    if pd.api.types.is_datetime64_any_dtype(stock_data_display_row['Date']):
                        stock_data_display_row['Date'] = stock_data_display_row['Date'].dt.strftime('%Y-%m-%d').fillna('N/A')
                    elif stock_data_display_row['Date'].dtype == 'object':
                         # Convert to datetime and then format to string, which is the most robust way to clean it here
                         stock_data_display_row['Date'] = pd.to_datetime(stock_data_display_row['Date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('N/A')
                    
                # Format ratio columns
                for col in ['EV/TTM SALES', 'TTM PE', 'FY + 1 PE', 'FY + 2 PE']:
                    if col in stock_data_display_row.columns:
                        value = stock_data_display_row[col].iloc[0]
                        if pd.notna(value):
                            stock_data_display_row[col] = f"{value:.2f}x"
                        elif col in stock_data_display_row.columns:
                            stock_data_display_row[col] = np.nan 

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

                
                # --- CONDITIONAL SUMMARY LOGIC ---
                
                show_fs = pr_type == "Earnings Release" and FINANCIAL_SUMMARY_COL in row
                show_os = OPERATIONAL_SUMMARY_COL in row
                
                # 1. Determine the content available for display
                financial_summary = row.get(FINANCIAL_SUMMARY_COL, 'N/A')
                operational_summary = row.get(OPERATIONAL_SUMMARY_COL, 'N/A')
                general_summary = row.get('Summary', 'N/A')

                cleaned_fs = clean_summary_text(financial_summary)
                cleaned_os = clean_summary_text(operational_summary)
                cleaned_gs = clean_summary_text(general_summary)
                
                
                # 2. Display the summaries
                # NOTE: The main "### Summaries" header is intentionally removed.
                if (show_fs and cleaned_fs != 'N/A') or (show_os and cleaned_os != 'N/A'):
                    
                    if show_fs and cleaned_fs != 'N/A' and show_os and cleaned_os != 'N/A':
                        # Case 1: Earnings Release - Show Financial and Operational side-by-side
                        summary_cols = st.columns(2)
                        
                        with summary_cols[0]:
                            st.markdown("#### Financial Summary")
                            st.info(cleaned_fs if cleaned_fs != 'N/A' else "Financial Summary not available.")

                        with summary_cols[1]:
                            st.markdown("#### Operational Summary")
                            st.info(cleaned_os if cleaned_os != 'N/A' else "Operational Summary not available.")
                            
                    elif show_os and cleaned_os != 'N/A':
                        # Case 2: Not Earnings Release (or FS missing/empty) - Show Operational Summary full width
                        st.markdown("#### Operational Summary")
                        st.info(cleaned_os if cleaned_os != 'N/A' else "Operational Summary not available.")
                        
                    st.markdown('<hr style="margin: 5px 0 10px 0; border: 0.5px solid rgba(255, 255, 255, 0.1);">', unsafe_allow_html=True)


                # 3. Fallback to General Summary
                if cleaned_gs != 'N/A' and not (show_fs and cleaned_fs != 'N/A'):
                    st.markdown("#### General Summary")
                    st.info(cleaned_gs)

                
                # 4. Positive and Negative Insights (always shown, always cleaned)
                col_pos, col_neg = st.columns(2)
                
                with col_pos:
                    st.markdown("#### Positive Insights")
                    positive_insights = row.get('Positive Insights', 'N/A')
                    cleaned_pos = clean_summary_text(positive_insights)
                    st.success(cleaned_pos if cleaned_pos != 'N/A' else "Positive Insights not available.")

                with col_neg:
                    st.markdown("#### Negative Insights")
                    negative_insights = row.get('Negative Insights', 'N/A')
                    cleaned_neg = clean_summary_text(negative_insights)
                    st.error(cleaned_neg if cleaned_neg != 'N/A' else "Negative Insights not available.") 
            
else:
    st.markdown("---")
    st.info("Select one or more rows in the table above to view the Summary and Insights.")