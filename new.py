
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from supabase import create_client
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
from collections import defaultdict
import datetime
import re
import matplotlib.pyplot as plt
import io
import base64
import json
import pytz
from dateutil import parser
from sqlagent import sql_agent, interpret_result, execute_sql

try:
    from agent12 import  generate_investment_summary, SmartInvestmentAdvisor, handle_sip_query
    INVESTMENT_ADVISOR_AVAILABLE = True
    print("✅ sip advisor module loaded successfully")
except ImportError as e:
    print(f"⚠️ Investment advisor module not found: {e}")
    INVESTMENT_ADVISOR_AVAILABLE = False

try:
    from loan import loan_optimization_pipeline as run_loan_optimization
    LOAN_OPTIMIZATION_AVAILABLE = True
    print("✅ Loan optimization module loaded successfully")
except ImportError as e:
    print(f"⚠️ Loan optimization module not found or has errors: {e}")
    LOAN_OPTIMIZATION_AVAILABLE = False
# Import from book3.py for tax calculations
try:
    from book3 import (
        handle_enhanced_query, 
        analyze_income_by_heads, 
        calculate_tax_liability_enhanced,
        generate_detailed_tax_report,
        calculate_tax_for_specific_income,
        get_user_transactions as get_tax_transactions,
        query_transaction_insights
    )
    TAX_MODULE_AVAILABLE = True
    print("✅ Enhanced tax module loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced tax module not found: {e}")
    print("Tax calculations will be disabled.")
    TAX_MODULE_AVAILABLE = False

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize LLM once
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_index"
RESOURCE_FOLDER = "Resources"
FOOD_KEYWORDS = {"zomato", "swiggy", "domino", "pizza", "food", "restaurant", "cafe", "meal", "dining"}
INVESTMENT_KEYWORDS = {"lic", "elss", "nps", "ppf", "insurance", "mutual fund", "sip", "fd", "fixed deposit"}
AVOIDABLE_CATEGORIES = {"entertainment", "shopping", "luxury", "gaming", "subscription"}
PERSON_KEYWORDS = {"to", "sent to", "paid to", "transfer to", "payment to"}

# Cache for expensive operations
@lru_cache(maxsize=128)
def get_cached_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

# Modified to get ALL transactions (no date filtering)
def get_user_transactions(user_id: str, limit: int = 5000) -> List[dict]:
    """Get all user transactions without date filtering"""
    try:
        result = (
            supabase
            .table("transactions")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data
    except Exception as e:
        print(f"❌ Error fetching transactions: {e}")
        return []

def get_transactions_by_date_range(user_id: str, start_date: str = None, end_date: str = None) -> List[dict]:
    """Get transactions within specific date range"""
    try:
        query = supabase.table("transactions").select("*").eq("user_id", user_id)
        
        if start_date:
            query = query.gte("created_at", start_date)
        if end_date:
            query = query.lte("created_at", end_date)
            
        result = query.order("created_at", desc=True).limit(5000).execute()
        return result.data
    except Exception as e:
        print(f"❌ Error fetching transactions by date: {e}")
        return []

def get_recent_loans(user_id: str) -> List[dict]:
    try:
        result = supabase.table("loan_info").select("*").eq("user_id", user_id).limit(10).execute()
        print(result)
        return result.data
    except:
        return []

@lru_cache(maxsize=32)
def get_user_balance(user_id: str) -> float:
    try:
        result = supabase.table("users").select("balance").eq("id", user_id).single().execute()
        return float(result.data['balance']) if result.data and 'balance' in result.data else 0.0
    except:
        return 0.0

def find_person_transactions(transactions: List[dict], person_name: str) -> List[dict]:
    """Find all transactions related to a specific person"""
    person_transactions = []
    person_name_lower = person_name.lower()
    
    for txn in transactions:
        desc = txn.get("description", "").lower()
        if (person_name_lower in desc and 
            any(keyword in desc for keyword in PERSON_KEYWORDS)):
            person_transactions.append(txn)
    
    return person_transactions
def handle_loan_optimization_query(user_input: str, user_id: str) -> str:
    """
    Handler function to fetch data and run the loan optimization pipeline.
    """
    if not LOAN_OPTIMIZATION_AVAILABLE:
        return "❌ Loan optimization feature is currently unavailable."

    try:
        print(f"🔄 Initiating loan optimization for user {user_id}...")
        # 1. Fetch user transactions using existing function
        transactions = get_user_transactions(user_id)
        if not transactions:
            return "❌ No transaction data found for loan optimization analysis."

        # 2. Call the loan optimization pipeline from the separate module
        # Pass the raw transaction data list
        results = run_loan_optimization(user_id, user_input, transactions)

        # 3. Check for errors from the pipeline
        if "error" in results:
            return f"❌ Loan Optimization Error: {results['error']}"

        # 4. Format the results into a user-friendly response string
        # Extract key information
        lifestyle = results.get('lifestyle', 'Unknown')
        monthly_emi = results.get('monthly_emi', 0.0)
        monthly_summary_df = results.get('monthly_summary', pd.DataFrame())
        normal_schedule_df = results.get('normal_schedule', pd.DataFrame())
        optimized_schedule_df = results.get('optimized_schedule', pd.DataFrame())

        # Basic summary
        response = f"💰 Loan Optimization Analysis for User: {user_id}\n"
        response += f"🏠 Inferred Lifestyle: {lifestyle}\n"
        response += f"💳 Monthly Loan EMI: ₹{monthly_emi:,.2f}\n\n"

        # Monthly Summary (last few months)
        if not monthly_summary_df.empty:
            response += "📊 Recent Monthly Summary:\n"
            recent_summary = monthly_summary_df.tail() # Show last 3 months
            for _, row in recent_summary.iterrows():
                 response += (f"  • {row['Month']}: Salary ₹{row['Salary']:,.2f}, "
                              f"Spent ₹{row['Unnecessary_Spending']:,.2f} unnecessarily.\n "
                            )
            response += "\n"

        # Loan Schedule Comparison (simplified)
        if not normal_schedule_df.empty and not optimized_schedule_df.empty:
            normal_duration = len(normal_schedule_df)
            optimized_duration = len(optimized_schedule_df)
            
            # Safely sum interest, handling potential missing column
            normal_interest = normal_schedule_df['Interest'].sum() if 'Interest' in normal_schedule_df.columns else 0
            optimized_interest = optimized_schedule_df['Interest'].sum() if 'Interest' in optimized_schedule_df.columns else 0

            response += "📉 Loan Repayment Comparison:\n"
            response += f"  • Standard Plan: {normal_duration} months, Total Interest: ₹{normal_interest:,.2f}\n"
            response += f"  • Optimized Plan: {optimized_duration} months, Total Interest: ₹{optimized_interest:,.2f}\n"
            
            months_saved = normal_duration - optimized_duration
            interest_saved = normal_interest - optimized_interest
            
            if months_saved > 0:
                response += f"  💡 Optimization saves approximately {months_saved} months and ₹{interest_saved:,.2f} in interest.\n"
            elif months_saved == 0 and interest_saved > 0:
                 response += f"  💡 Optimization saves ₹{interest_saved:,.2f} in interest (same duration).\n"
            else:
                 response += "  ℹ️  No significant time or interest savings found with current spending pattern. Keep optimizing!\n"
            response += "\n"

        # --- NEW: Add Full Loan Schedules ---
        # Option 1: Add them directly to the text response (can be very long)
        # response += "📅 Standard Loan Repayment Schedule:\n"
        # response += normal_schedule_df.head(15).to_string(index=False) # Show first 15 months
        # response += "\n\n"
        # response += "📅 Optimized Loan Repayment Schedule:\n"
        # response += optimized_schedule_df.head(15).to_string(index=False) # Show first 15 months
        # response += "\n\n(Note: Full schedules might be truncated for brevity.)\n\n"

        # Option 2: Better formatting for inclusion in text (showing start and end)
        def format_schedule_snippet(schedule_df, title, max_rows=10):
            """Helper to format a snippet of the schedule table."""
            if schedule_df.empty:
                return f"{title}: No data available.\n"
            snippet = f"{title}:\n"
            # Show header
            snippet += " | ".join(schedule_df.columns) + "\n"
            snippet += "-" * len(" | ".join(schedule_df.columns)) + "\n"
            
            # Show first few rows
            head_rows = schedule_df.head(max_rows // 2)
            for _, row in head_rows.iterrows():
                snippet += " | ".join([f"{v:,.2f}" if isinstance(v, (int, float)) else str(v) for v in row]) + "\n"
            
            if len(schedule_df) > max_rows:
                 snippet += "... (middle rows omitted) ...\n"
            
            # Show last few rows if there are more
            if len(schedule_df) > max_rows:
                tail_rows = schedule_df.tail(max_rows // 2)
                for _, row in tail_rows.iterrows():
                     snippet += " | ".join([f"{v:,.2f}" if isinstance(v, (int, float)) else str(v) for v in row]) + "\n"
            
            snippet += f"(Total months: {len(schedule_df)})\n\n"
            return snippet

        response += format_schedule_snippet(normal_schedule_df, "📅 Standard Loan Repayment Schedule")
        response += format_schedule_snippet(optimized_schedule_df, "📅 Optimized Loan Repayment Schedule")


        print(f"✅ Loan optimization completed for user ")
        return response

    except Exception as e:
        print(f"❌ Error in handle_loan_optimization_query: {e}")
        import traceback
        traceback.print_exc() # For debugging
        return f"❌ An unexpected error occurred while running the loan optimization: {str(e)}"
def analyze_spending_by_category(transactions: List[dict], category: str) -> Dict:
    """Analyze spending in a specific category"""
    category_transactions = []
    total_amount = 0
    
    category_lower = category.lower()
    
    for txn in transactions:
        desc = txn.get("description", "").lower()
        txn_category = txn.get("category", "").lower()
        
        if (category_lower in desc or 
            category_lower in txn_category or
            (category_lower == "food" and any(keyword in desc for keyword in FOOD_KEYWORDS))):
            
            amount = float(txn.get("amount", 0))
            category_transactions.append({
                "date": txn["created_at"],
                "description": txn["description"],
                "amount": amount,
                "type": txn.get("type", "")
            })
            
            if txn.get("type") in ["sent", "expense"]:
                total_amount += amount
    
    return {
        "transactions": category_transactions,
        "total_spent": total_amount,
        "transaction_count": len(category_transactions)
    }

def analyze_transactions_comprehensive(transactions: List[dict]) -> Dict:
    """Comprehensive analysis for all transaction insights (non-tax)"""
    if not transactions:
        return {}
    
    category_totals = defaultdict(float)
    monthly_totals = defaultdict(float)
    yearly_totals = defaultdict(float)
    person_totals = defaultdict(float)
    total_income = 0
    total_expense = 0
    reminders = []
    investment_entries = []
    avoidable_expenses = []
    
    for txn in transactions:
        try:
            amount = float(txn["amount"])
            desc = txn["description"]
            desc_lower = desc.lower()
            txn_type = txn.get("type", "")
            category = txn.get("category", "Other")
            
            # Enhanced categorization
            if any(keyword in desc_lower for keyword in FOOD_KEYWORDS):
                category = "food"
            elif any(keyword in desc_lower for keyword in INVESTMENT_KEYWORDS):
                category = "investment"
            
            category_totals[category] += amount
            
            # Monthly and yearly grouping
            try:
                created = txn["created_at"]
                dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
                month_key = dt.strftime("%B %Y")
                year_key = dt.strftime("%Y")
                monthly_totals[month_key] += amount
                yearly_totals[year_key] += amount
            except:
                pass
            
            if any(keyword in desc_lower for keyword in PERSON_KEYWORDS):
                person_match = re.search(r'(?:to|sent to|paid to|transfer to|payment to)\s+([a-zA-Z\s]+)', desc_lower)
                if person_match:
                    person_name = person_match.group(1).strip().title()
                    person_totals[person_name] += amount
            
            if txn_type == "received":
                total_income += amount
            elif txn_type in ["sent", "expense"]:
                total_expense += amount
            
            if txn_type == 'loan_payment':
                date = txn["created_at"][:10]
                reminders.append(f"Loan payment was due on {date} for ₹{amount}")
            
            if any(keyword in desc_lower for keyword in INVESTMENT_KEYWORDS):
                investment_entries.append({
                    "date": txn["created_at"],
                    "amount": amount,
                    "description": desc
                })
            
            if (category.lower() in AVOIDABLE_CATEGORIES and amount > 1000):
                avoidable_expenses.append(txn)
                
        except Exception as e:
            print(f"Error processing transaction: {e}")
            continue
    
    return {
        "category_totals": dict(category_totals),
        "monthly_totals": dict(monthly_totals),
        "yearly_totals": dict(yearly_totals),
        "person_totals": dict(person_totals),
        "total_income": total_income,
        "total_expense": total_expense,
        "reminders": reminders,
        "investment_entries": investment_entries,
        "avoidable_expenses": avoidable_expenses,
        "net_savings": total_income - total_expense
    }

def generate_spending_summary(analysis: Dict) -> str:
    """Generate spending summary from pre-analyzed data"""
    if not analysis or not analysis.get("category_totals"):
        return "No spending data available."
    
    summary = "\n".join([
        f"{cat.title()}: ₹{amt:,.2f}" 
        for cat, amt in analysis["category_totals"].items()
    ])
    return f"💰 Spending Summary:\n{summary}"

def generate_person_summary(analysis: Dict) -> str:
    """Generate person-wise spending summary"""
    person_totals = analysis.get("person_totals", {})
    if not person_totals:
        return "No person-wise transactions found."
    
    summary = "\n".join([
        f"{person}: ₹{amt:,.2f}" 
        for person, amt in sorted(person_totals.items(), key=lambda x: x[1], reverse=True)
    ])
    return f"👥 Person-wise Transactions:\n{summary}"

def generate_yearly_summary(analysis: Dict) -> str:
    """Generate yearly spending summary"""
    yearly_totals = analysis.get("yearly_totals", {})
    if not yearly_totals:
        return "No yearly data available."
    
    summary = "\n".join([
        f"{year}: ₹{amt:,.2f}" 
        for year, amt in sorted(yearly_totals.items(), reverse=True)
    ])
    return f"📅 Yearly Summary:\n{summary}"

def gst_summary_basic(transactions: List[dict]) -> str:
    """Basic GST summary from transactions"""
    gst_collected = 0
    gst_paid = 0
    
    for txn in transactions:
        desc = txn.get("description", "").lower()
        amount = float(txn.get("amount", 0))
        
        if "gst collected" in desc or "gst received" in desc:
            gst_collected += amount
        elif "gst paid" in desc or "gst payment" in desc:
            gst_paid += amount
    
    liability = gst_collected - gst_paid
    return f"📊 GST Summary: Collected ₹{gst_collected:,.2f}, Paid ₹{gst_paid:,.2f}, Net Liability ₹{liability:,.2f}"

def simulate_loan_repayment_impact(analysis: Dict, loans: List[dict]) -> str:
    """Loan repayment simulation"""
    if not loans or not analysis.get("avoidable_expenses"):
        return "No loan optimization insights available."
    
    loan = loans[0]
    monthly_interest = float(loan.get('monthly_interest', 0))
    if monthly_interest == 0:
        return "No loan interest data available."
    
    messages = []
    for txn in analysis["avoidable_expenses"][:3]:
        alt_payment = float(txn['amount'])
        months_saved = alt_payment / monthly_interest
        if months_saved >= 1:
            messages.append(
                f"💡 If you had not spent ₹{alt_payment:.2f} on '{txn['description']}', "
                f"you could have reduced your loan tenure by approximately {int(months_saved)} month(s)."
            )
    
    return "\n".join(messages) if messages else "No significant optimization insights found."

def generate_spending_chart(analysis: Dict) -> str:
    """Generate spending chart"""
    category_totals = analysis.get("category_totals", {})
    if not category_totals:
        return ""
    
    sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    categories = [cat for cat, _ in sorted_categories]
    values = [amt for _, amt in sorted_categories]
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='teal', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Amount (₹)')
    plt.title('Top 10 Spending Categories')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def generate_gamified_dashboard(transactions: List[dict]) -> str:
    """Generate gamified dashboard"""
    from collections import defaultdict
    
    category_totals = defaultdict(float)
    total_income = 0
    total_expense = 0
    
    for txn in transactions:
        amt = float(txn["amount"])
        desc = txn["description"].lower()
        if txn["type"] == "received":
            total_income += amt
        elif txn["type"] in ["sent", "expense"]:
            total_expense += amt
            cat = txn.get("category", "Other")
            category_totals[cat] += amt
    
    net_savings = total_income - total_expense
    daily_limit = round((total_income - total_expense) / 30, 2) if total_income > 0 else 0
    
    sorted_cats = sorted(category_totals.items(), key=lambda x: -x[1])[:3]
    cat_labels = [f"{cat}" for cat, amt in sorted_cats]
    cat_values = [amt for cat, amt in sorted_cats]
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax0 = fig.add_subplot(gs[0, 0])
    if total_expense > 0:
        wedges, _ = ax0.pie([total_expense, max(1, total_income - total_expense)],
                            colors=["#FF6B6B", "#4ECDC4"], startangle=90)
        ax0.set_title("💸 Total Spending")
        ax0.text(0, 0, f"₹{total_expense:,.0f}", ha='center', va='center', fontsize=12, weight='bold')
    
    ax1 = fig.add_subplot(gs[0, 1])
    if cat_values:
        ax1.barh(cat_labels, cat_values, color=["#FF9F43", "#10AC84", "#5F27CD"])
        ax1.set_title("🏆 Top Categories")
        ax1.invert_yaxis()
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    savings_color = "#26de81" if net_savings > 0 else "#fc5c65"
    ax2.text(0.5, 0.5, f"₹{net_savings:,.0f}\n💰 Net Savings", 
             ha='center', va='center', fontsize=12, weight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor=savings_color, alpha=0.3))
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    monthly_avg = total_expense / 12 if total_expense > 0 else 0
    ax3.text(0.5, 0.5, f"₹{monthly_avg:,.0f}\n📅 Monthly Avg", 
             ha='center', va='center', fontsize=12, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#a55eea", alpha=0.3))
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(['Income', 'Expense'], [total_income, total_expense], 
            color=['#26de81', '#fc5c65'], alpha=0.7)
    ax4.set_title("💹 Income vs Expense")
    ax4.set_ylabel('Amount (₹)')
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ax5.text(0.5, 0.5, f"₹{daily_limit:,.0f}\n🎯 Daily Budget", 
             ha='center', va='center', fontsize=12, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#feca57", alpha=0.3))
    
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    ax6.text(0.5, 0.5, "🤖 Generated by Enhanced Financial Assistant", 
             ha='center', va='center', fontsize=10, style='italic')
    
    fig.suptitle("📊 Complete Financial Dashboard", fontsize=16, weight='bold')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

@lru_cache(maxsize=1)
def load_documents_cached():
    """Cache document loading to avoid repeated file I/O"""
    if not os.path.exists(RESOURCE_FOLDER):
        return []
    
    documents = []
    for filename in os.listdir(RESOURCE_FOLDER):
        path = os.path.join(RESOURCE_FOLDER, filename)
        try:
            if filename.endswith(".txt"):
                documents.extend(TextLoader(path, encoding='utf-8').load())
            elif filename.endswith(".pdf"):
                documents.extend(PyPDFLoader(path).load())
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return documents

def create_retriever_optimized():
    """Optimized retriever creation with caching"""
    embeddings = get_cached_embeddings()
    
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("🔁 Loading existing ChromaDB...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
    else:
        print("🆕 Creating new ChromaDB...")
        docs = load_documents_cached()
        if not docs:
            print("⚠️ No documents found.")
            return None
        
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            texts, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH
        )
        print("✅ ChromaDB created and saved.")
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

_chain_cache = None
_retriever_cache = None

def get_or_create_chain() -> Tuple:
    """Get cached chain or create new one"""
    global _chain_cache, _retriever_cache
    
    if _chain_cache is None or _retriever_cache is None:
        _retriever_cache = create_retriever_optimized()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a multilingual financial assistant. Provide concise, accurate responses based on transaction data."),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ])
        base_chain = prompt | llm
        
        def get_memory(session_id: str):
            return InMemoryChatMessageHistory()
        
        _chain_cache = RunnableWithMessageHistory(
            base_chain,
            get_memory,
            input_messages_key="input",
            history_messages_key="history"
        )
    
    return _chain_cache, _retriever_cache

def classify_query_intent(user_input: str) -> str:
    """Classify the intent of the user query using LLM"""
    intent_prompt = PromptTemplate.from_template("""
    Analyze the following user query and classify its intent into one of the following categories:
    
    - sql_query: Queries that require database operations like:
      * Counting, summing, averaging transactions
      * Finding specific transactions by criteria
      * Listing/showing all transactions
      * Complex filtering or aggregation
      * Any query that needs to search through transaction data systematically
    - loan_optimization_query: Specifically asking for loan optimization, budgeting to pay off loans faster, or analyzing spending to reduce loan tenure/interest.
    - tax_query: Related to taxes, ITR, tax calculations, or income heads
    - balance_query: Asking about current balance or funds
    - sip_query: Specifically asking for SIP (Systematic Investment Plan) recommendations, suggestions, or advice                                            
    
    - category_query: Asking about spending in specific categories (e.g., food, investment)
    - yearly_query: Asking for yearly or annual summaries
    - dashboard_query: Asking for dashboard, summary, or overview
    - chart_query: Asking for charts, graphs, or visualizations
    - general_query: Any other financial query requiring context analysis

    Examples of sql_query:
    - "Show me all transactions above 5000"
    - "How many transactions did I make last month?"
    - "Find all payments to John"
    - "List transactions between January and March"
    - "What's the total amount I spent on food?"
    - "Show me my largest transactions"
    Examples of sip_query:
    - "Suggest some good SIPs"
    - "Best SIP for me"
    - "Which SIP should I start?"
    - "SIP recommendations"
    - "How much SIP should I do?"
    - "Good SIPs to invest in"
    - "SIP advice"
    
    Examples of investment_query:
    - "What should I invest in?"
    - "How to build my investment portfolio?"
    - "Best mutual funds for beginners"
    - "Investment advice based on my savings"
    - "Portfolio allocation strategy"

    Query: {query}

    Return only the intent category name.
    """)
    
    prompt = intent_prompt.format(query=user_input)
    result = llm.invoke(prompt)
    return result.content.strip()

def route_query(user_input: str, user_id: str, plain_text_mode: bool) -> str:
    """Route the query to the appropriate handler based on classified intent"""
    intent = classify_query_intent(user_input)
    transactions = get_user_transactions(user_id)
    loans = get_recent_loans(user_id)
    analysis = analyze_transactions_comprehensive(transactions) if transactions else {}

    if intent == "tax_query" and TAX_MODULE_AVAILABLE:
        return handle_enhanced_query(user_input, user_id)
    
    elif intent == "sql_query":
        print("🔄 Handling SQL query...")
        return sql_agent(user_input, user_id)
    
    elif intent == "sip_query" and INVESTMENT_ADVISOR_AVAILABLE:
        print("🔄 Handling SIP-specific query...")
        return handle_sip_query(user_input, user_id, transactions)
    elif intent == "balance_query":
        balance = get_user_balance(user_id)
        return f"💰 Your current balance is ₹{balance:,.2f}."
    
    elif intent == "person_query":
        return sql_agent(user_input, user_id)
    
    elif intent == "category_query":
        # Extract category from query (simple heuristic for demo purposes)
        category = "food" if "food" in user_input.lower() else "investment" if "investment" in user_input.lower() else "other"
        category_analysis = analyze_spending_by_category(transactions, category)
        if category_analysis["total_spent"] > 0:
            response = f"📊 {category.title()} Spending Analysis:\n"
            response += f"Total spent on {category}: ₹{category_analysis['total_spent']:,.2f}\n"
            response += f"Number of transactions: {category_analysis['transaction_count']}\n\n"
            response += f"Recent {category} transactions:\n"
            for txn in category_analysis["transactions"][:5]:
                response += f"• {txn['date'][:10]}: {txn['description']} - ₹{txn['amount']:,.2f}\n"
            return response
        return f"No {category}-related expenses found in your transactions."
    
    elif intent == "yearly_query":
        return generate_yearly_summary(analysis)
    
    elif intent == "dashboard_query":
        dashboard_img = generate_gamified_dashboard(transactions)
        return f"📊 Here's your complete financial dashboard:\n\n{generate_spending_summary(analysis)}\n\n{generate_person_summary(analysis)}\n\n<img src=\"{dashboard_img}\" width=\"700\"/>"
    
    elif intent == "chart_query":
        chart_img = generate_spending_chart(analysis)
        if chart_img and not plain_text_mode:
            return f"📈 Here's your spending visualization:\n\n<img src=\"{chart_img}\" width=\"600\"/>"
    elif intent == "loan_optimization_query":
        print("🔄 Handling Loan Optimization query...")
        return handle_loan_optimization_query(user_input, user_id) # <-- Call the new handler
    else:  # general_query
        chain, retriever = get_or_create_chain()
        context_parts = []
        
        recent_activity = "\n".join([
            f"{t['created_at'][:10]}: {t['description']} - ₹{t['amount']}" 
            for t in transactions[:10]
        ])
        context_parts.extend([
            f"Recent Transactions:\n{recent_activity}",
            generate_spending_summary(analysis),
            generate_person_summary(analysis),
            generate_yearly_summary(analysis),
            f"Total Income: ₹{analysis['total_income']:,.2f}",
            f"Total Expense: ₹{analysis['total_expense']:,.2f}",
            f"Net Savings: ₹{analysis['net_savings']:,.2f}",
            f"Reminders:\n{chr(10).join(analysis.get('reminders', []))}",
            gst_summary_basic(transactions)
        ])
        
        if loans:
            context_parts.append(f"Loan Optimization:\n{simulate_loan_repayment_impact(analysis, loans)}")
        
        if retriever:
            try:
                docs = retriever.invoke(user_input)
                combined_context = "\n\n".join([doc.page_content for doc in docs])
                context_parts.append(f"Knowledge Base:\n{combined_context}")
            except Exception as e:
                print(f"Retriever error: {e}")
        
        context = "\n\n".join(context_parts)
        prompt_template = PromptTemplate.from_template("""
        ### USER REQUEST:
        {query}
        
        ### FINANCIAL CONTEXT:
        {context}
        
        ### RESPONSE (Be specific and use the transaction data):
        """)
        
        prompt = prompt_template.format(query=user_input, context=context)
        result = chain.invoke(
            {"input": prompt}, 
            config={"configurable": {"session_id": user_id}}
        )
        
        return result.content

def handle_query(user_input: str, user_id: str = "demo-user", plain_text_mode: bool = False) -> str:
    """Main entry point for query handling with agentic routing"""
    if not user_input.strip():
        return "Please provide a valid query."
    return route_query(user_input, user_id, plain_text_mode)