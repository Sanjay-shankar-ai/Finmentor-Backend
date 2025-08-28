"""
Enhanced SQL Agent to Convert Natural Language to SQL and Execute on Supabase
"""

import os
import re
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from supabase import create_client
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------- ENV Setup ----------
load_dotenv()

# Supabase Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# LLM Setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

# ---------- Enhanced PROMPT Templates ----------
ENHANCED_SQL_PROMPT = PromptTemplate.from_template("""
You are an expert SQL assistant specializing in financial transaction queries.

Database Schema:
- Table: transactions
- Columns: id (bigint), user_id (text), description (text), amount (numeric), 
          type (text: 'sent', 'received', 'expense', 'income'), 
          category (text), created_at (timestamp), contact_name (text)

CRITICAL SQL RULES:
1. ALWAYS filter by user_id = '{user_id}'
2. When using aggregate functions (SUM, COUNT, AVG), DO NOT include non-aggregated columns in SELECT unless they are in GROUP BY
3. For total/sum queries, use only: SELECT SUM(amount) as total_amount FROM transactions WHERE ...
4. For current month: WHERE EXTRACT(MONTH FROM created_at) = EXTRACT(MONTH FROM CURRENT_DATE) AND EXTRACT(YEAR FROM created_at) = EXTRACT(YEAR FROM CURRENT_DATE)
5. For contact searches: WHERE contact_name ILIKE '%CONTACT_NAME%' (use partial matching)
6. For individual transaction details: SELECT id, amount, description, contact_name, created_at FROM transactions WHERE ... ORDER BY created_at DESC
7. Never mix aggregate and non-aggregate columns without proper GROUP BY

QUERY TYPE PATTERNS:

A) TOTAL/SUM QUERIES (How much did I...):
SELECT SUM(amount) as total_amount 
FROM transactions 
WHERE user_id = '{user_id}' AND [conditions]

B) INDIVIDUAL TRANSACTIONS (Show me transactions...):
SELECT id, amount, description, contact_name, created_at::date as date
FROM transactions 
WHERE user_id = '{user_id}' AND [conditions]
ORDER BY created_at DESC

C) GROUPED SUMMARIES (spending by category/contact):
SELECT contact_name, SUM(amount) as total_amount
FROM transactions 
WHERE user_id = '{user_id}' AND [conditions]
GROUP BY contact_name
ORDER BY total_amount DESC

EXAMPLES:
- "How much I sent to John this month" → Type A (SUM query)
- "Show my transactions with John" → Type B (detail query)  
- "Spending by contact" → Type C (grouped query)

Current date: {current_date}
Question: {question}

Return ONLY the PostgreSQL SELECT query. No explanations, markdown, or extra text.

SQL:
""")

ENHANCED_RESULT_PROMPT = ChatPromptTemplate.from_template("""
You are a financial advisor analyzing transaction data for a user in India.
for data results Respond in Markdown format as a table.
Query Context: {query_context}
Data Results:
{table}

Provide a clear, insightful analysis including:
1. 📊 **Summary**: Key numbers and totals (format amounts in ₹)
2. 🔍 **Key Insights**: Patterns, trends, or notable findings
3. 💡 **Observations**: Actionable insights or recommendations if relevant
4. 📈 **Context**: How this data relates to financial habits

Keep the response conversational and helpful. Use Indian Rupee (₹) formatting for amounts.

Response:
""")

# ---------- Enhanced Query Classification ----------
QUERY_CATEGORIES = {
    'spending': ['spent', 'expense', 'paid', 'bought', 'spending', 'cost'],
    'income': ['received', 'earned', 'income', 'got', 'credited'],
    'balance': ['balance', 'total', 'net', 'summary'],
    'contacts': ['from', 'to', 'contact', 'person', 'friend'],
    'time': ['month', 'year', 'week', 'day', 'recent', 'last', 'this'],
    'category': ['category', 'food', 'transport', 'entertainment', 'bills']
}

def classify_query(question: str) -> List[str]:
    """Classify the type of query to provide better context"""
    question_lower = question.lower()
    categories = []
    
    for category, keywords in QUERY_CATEGORIES.items():
        if any(keyword in question_lower for keyword in keywords):
            categories.append(category)
    
    return categories

# ---------- Enhanced SQL Sanitizer ----------
def sanitize_sql(sql: str) -> str:
    """Enhanced SQL sanitization with better error handling"""
    sql = sql.strip()
    
    # Remove code blocks if present
    sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'```\s*', '', sql)
    
    # Extract SELECT statement
    sql = re.sub(r"(?is)^.*?(SELECT)", r"\1", sql)
    
    # Remove trailing semicolon and whitespace
    sql = sql.split(";")[0].strip()
    
    # Enhanced security checks - check for dangerous keywords as standalone words
    dangerous_patterns = [
        r'\bDROP\s+TABLE\b',
        r'\bDELETE\s+FROM\b', 
        r'\bUPDATE\s+\w+\s+SET\b',
        r'\bINSERT\s+INTO\b',
        r'\bCREATE\s+TABLE\b',
        r'\bCREATE\s+DATABASE\b',
        r'\bALTER\s+TABLE\b',
        r'\bTRUNCATE\s+TABLE\b',
        r'\bGRANT\b',
        r'\bREVOKE\b'
    ]
    
    sql_upper = sql.upper()
    
    for pattern in dangerous_patterns:
        if re.search(pattern, sql_upper):
            keyword = pattern.split(r'\\b')[1].split(r'\\s')[0]
            raise ValueError(f"🛑 Unsafe SQL operation detected: {keyword}")
    
    # Additional check for multiple statements
    if ';' in sql and sql.count(';') > 0:
        raise ValueError("🛑 Multiple SQL statements not allowed")
    
    if not sql.upper().startswith("SELECT"):
        raise ValueError(f"🛑 Invalid SQL query. Must start with SELECT: {sql[:100]}...")
    
    return sql

def execute_sql(sql: str) -> Dict[str, Any]:
    """Execute SQL with enhanced error handling"""
    try:
        print(f"🔍 Executing SQL: {sql}")
        response = supabase.rpc("exec_sql", {"sql": sql}).execute()
        
        if hasattr(response, 'data') and response.data is not None:
            return {"success": True, "data": response.data}
        else:
            return {"success": False, "error": "No data returned from query"}
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ SQL Execution Error: {error_msg}")
        return {"success": False, "error": error_msg}

def format_currency(amount: float) -> str:
    """Format amount in Indian Rupees"""
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.1f} Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.1f} L"
    elif amount >= 1000:
        return f"₹{amount/1000:.1f}k"
    else:
        return f"₹{amount:.2f}"

def enhance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Enhance dataframe with better formatting"""
    df_enhanced = df.copy()
    
    # Format amount columns
    amount_columns = ['amount', 'total', 'sum', 'total_amount']
    for col in df_enhanced.columns:
        if any(amt_col in col.lower() for amt_col in amount_columns):
            if df_enhanced[col].dtype in ['float64', 'int64']:
                df_enhanced[f"{col}_formatted"] = df_enhanced[col].apply(format_currency)
    
    # Format dates if present
    date_columns = ['created_at', 'date', 'transaction_date']
    for col in df_enhanced.columns:
        if any(date_col in col.lower() for date_col in date_columns):
            try:
                df_enhanced[col] = pd.to_datetime(df_enhanced[col]).dt.strftime('%Y-%m-%d')
            except:
                pass
    
    return df_enhanced

def interpret_result(table_text: str, query_context: str = "") -> str:
    """Enhanced result interpretation with context"""
    try:
        result_chain = (
            {"table": RunnablePassthrough(), "query_context": lambda _: query_context}
            | ENHANCED_RESULT_PROMPT
            | llm
            | StrOutputParser()
        )
        return result_chain.invoke(table_text)
    except Exception as e:
        return f"⚠️ Could not generate insights: {str(e)}"

def get_query_suggestions(question: str) -> List[str]:
    """Provide helpful query suggestions based on input"""
    suggestions = []
    categories = classify_query(question)
    
    if 'spending' in categories:
        suggestions.extend([
            "Show my top 5 expenses this month",
            "What did I spend on food category?",
            "Show spending by category"
        ])
    
    if 'contacts' in categories:
        suggestions.extend([
            "How much did I send to [contact name]?",
            "Show all transactions with [contact name]",
            "Who did I send money to most?"
        ])
    
    if 'time' in categories:
        suggestions.extend([
            "Show last month's transactions",
            "What's my spending this year?",
            "Show recent transactions"
        ])
    
    return suggestions[:3]  # Return top 3 suggestions

def sql_agent(question: str, user_id: str = "") -> str:
    """Enhanced SQL agent with better error handling and insights"""
    
    if not user_id.strip():
        return "❌ Error: User ID is required for security."
    
    if not question.strip():
        return "❌ Error: Please provide a question about your transactions."
    
    try:
        # Classify query for context
        categories = classify_query(question)
        query_context = f"Query type: {', '.join(categories) if categories else 'general'}"
        
        print(f"🔍 Query categories: {categories}")
        
        # Step 1: Generate SQL with retry mechanism
        current_date = datetime.now().strftime('%Y-%m-%d')
        sql_chain = (
            {"question": RunnablePassthrough()}
            | ENHANCED_SQL_PROMPT.partial(user_id=user_id, current_date=current_date)
            | llm
            | StrOutputParser()
        )
        
        raw_sql = sql_chain.invoke(question)
        print("🧠 Raw SQL:\n", raw_sql)
        
        # Step 2: Sanitize SQL
        try:
            clean_sql = sanitize_sql(raw_sql)
            print("✅ Clean SQL:\n", clean_sql)
        except ValueError as e:
            return str(e)
        
        # Step 3: Execute SQL with error handling
        result = execute_sql(clean_sql)
        
        if not result["success"]:
            error_msg = result["error"]
            
            # Handle common PostgreSQL GROUP BY errors
            if "must appear in the GROUP BY clause" in error_msg:
                print("🔄 Retrying with corrected GROUP BY logic...")
                
                # Create a simpler, safer query for aggregation
                question_lower = question.lower()
                if any(word in question_lower for word in ['how much', 'total', 'sum']):
                    # This is likely a SUM query - create a simple aggregation
                    fallback_sql = f"""
                    SELECT SUM(amount) as total_amount 
                    FROM transactions 
                    WHERE user_id = '{user_id}' 
                    AND type = 'sent'
                    AND EXTRACT(MONTH FROM created_at) = EXTRACT(MONTH FROM CURRENT_DATE)
                    AND EXTRACT(YEAR FROM created_at) = EXTRACT(YEAR FROM CURRENT_DATE)
                    """
                    
                    # Add contact filter if mentioned in question
                    import re
                    contact_match = re.search(r'(?:to|sent)\s+([A-Za-z\s]+?)(?:\s+in|\s+this|\s+last|$)', question, re.IGNORECASE)
                    if contact_match:
                        contact_name = contact_match.group(1).strip()
                        fallback_sql += f" AND contact_name ILIKE '%{contact_name}%'"
                    
                    print(f"🔄 Fallback SQL: {fallback_sql}")
                    result = execute_sql(fallback_sql.strip())
            
            if not result["success"]:
                suggestions = get_query_suggestions(question)
                suggestion_text = "\n\n💡 **Try asking:**\n" + "\n".join(f"• {s}" for s in suggestions) if suggestions else ""
                return f"❌ **Database Error:** {result['error']}{suggestion_text}"
        
        # Step 4: Process results
        data = result["data"]
        
        if not data:
            suggestions = get_query_suggestions(question)
            suggestion_text = "\n\n💡 **Try asking:**\n" + "\n".join(f"• {s}" for s in suggestions) if suggestions else ""
            return f"📭 **No data found** for your query.{suggestion_text}"
        
        # Step 5: Format and enhance data
        df = pd.DataFrame(data)
        df_enhanced = enhance_dataframe(df)
        
        # Create table output
        if len(df_enhanced) > 20:
            table_text = df_enhanced.head(20).to_markdown(index=False)
            table_text += f"\n\n*Showing first 20 of {len(df_enhanced)} results*"
        else:
            table_text = df_enhanced.to_markdown(index=False)
        
        print("📊 Table Output:\n", table_text)
        
        # Step 6: Generate insights
        insights = interpret_result(table_text, query_context)
        
        return f"📊 **Query Results:**\n\n{table_text}\n\n{insights}"
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return f"❌ **Unexpected Error:** {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."

# ---------- Additional Utility Functions ----------
def get_schema_info() -> str:
    """Get database schema information for debugging"""
    return """
    Database Schema:
    Table: transactions
    - id: Unique identifier (bigint)
    - user_id: User identifier (text) - ALWAYS required in WHERE clause
    - description: Transaction description (text)
    - amount: Transaction amount (numeric)
    - type: Transaction type (text: 'sent', 'received', 'expense', 'income')
    - category: Transaction category (text)
    - created_at: Transaction timestamp (timestamp)
    - contact_name: Contact name for the transaction (text)
    """

def validate_environment() -> Tuple[bool, str]:
    """Validate environment setup"""
    missing = []
    
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    
    if missing:
        return False, f"Missing environment variables: {', '.join(missing)}"
    
    return True, "Environment setup is valid"

# ---------- Demo and Testing ----------
if __name__ == "__main__":
    # Validate environment
    is_valid, message = validate_environment()
    if not is_valid:
        print(f"❌ {message}")
        exit(1)
    
    print("✅ Environment validated")
    print("🔍 Enhanced SQL Agent Demo\n")
    print(get_schema_info())
    
    # Demo queries
    demo_queries = [
        "How much did I spend on food this month?",
        "Show my top 5 expenses",
        "What did I receive from contacts last week?",
        "Show spending by category",
        "How much did I send to John?",
        "What's my total balance this year?"
    ]
    
    print("\n💡 **Example queries you can try:**")
    for i, query in enumerate(demo_queries, 1):
        print(f"{i}. {query}")
    
    # Interactive demo (uncomment to test)
    # while True:
    #     user_input = input("\n🤖 Ask about your transactions (or 'quit' to exit): ")
    #     if user_input.lower() in ['quit', 'exit', 'q']:
    #         break
    #     
    #     result = sql_agent(user_input, user_id="demo_user")
    #     print(f"\n{result}\n")