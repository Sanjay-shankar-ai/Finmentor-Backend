
import os
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
from book3 import handle_enhanced_query, analyze_income_by_heads, calculate_tax_liability
import datetime
import re
import matplotlib.pyplot as plt
import io
import base64
import json
import pytz
from dateutil import parser


load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_index"
RESOURCE_FOLDER = "Resources"
FOOD_KEYWORDS = {"zomato", "swiggy", "domino", "pizza", "food"}
INVESTMENT_KEYWORDS = {"lic", "elss", "nps", "ppf", "insurance"}
AVOIDABLE_CATEGORIES = {"entertainment", "shopping", "luxury"}

# Cache for expensive operations
@lru_cache(maxsize=128)
def get_cached_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

# Optimized transaction fetching with better query
def get_user_transactions(user_id: str) -> List[dict]:
    try:
        # Pre-calculate dates once
        fy_start = datetime.datetime(2024, 4, 1, tzinfo=pytz.UTC).isoformat()
        fy_end = datetime.datetime.now(pytz.UTC).replace(hour=23, minute=59, second=59).isoformat()

        # Single optimized query with all needed data
        result = (
            supabase
            .table("transactions")
            .select("*")
            .eq("user_id", user_id)
            .gte("created_at", fy_start)
            .lte("created_at", fy_end)
            .order("created_at", desc=True)
            .limit(1000)  # Prevent excessive data loading
            .execute()
        )
        return result.data
    except Exception as e:
        print(f"❌ Error fetching transactions: {e}")
        return []

def get_recent_loans(user_id: str) -> List[dict]:
    try:
        result = supabase.table("loan_info").select("*").eq("user_id", user_id).limit(10).execute()
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

# Optimized analysis functions using single pass through data
def analyze_transactions_comprehensive(transactions: List[dict]) -> Dict:
    """Single pass analysis for all transaction insights"""
    if not transactions:
        return {}
    
    # Initialize all counters
    category_totals = defaultdict(float)
    monthly_totals = defaultdict(float)
    total_income = 0
    total_expense = 0
    gst_collected = 0
    gst_paid = 0
    tds_paid = 0
    reminders = []
    salary_sources = []
    investment_entries = []
    avoidable_expenses = []
    
    # Pre-compile regex for better performance
    salary_pattern = re.compile(r"(?i)salary\s*[\-–—]?\s*([a-zA-Z]+)\s+(\d{4})")
    
    # Single pass through transactions
    for txn in transactions:
        try:
            amount = float(txn["amount"])
            desc = txn["description"]
            desc_lower = desc.lower()
            txn_type = txn.get("type", "")
            category = txn.get("category", "Other")
            
            # Basic categorization
            if any(keyword in desc_lower for keyword in FOOD_KEYWORDS):
                category = "food"
            
            category_totals[category] += amount
            
            # Monthly grouping
            try:
                created = txn["created_at"]
                dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
                month_key = dt.strftime("%B %Y")
                monthly_totals[month_key] += amount
            except:
                pass
            
            # Income/Expense tracking
            if txn_type == "received":
                total_income += amount
            elif txn_type in ["sent", "expense"]:
                total_expense += amount
            
            # GST tracking
            if category.lower() == "gst_collected":
                gst_collected += amount
            elif category.lower() == "gst_paid":
                gst_paid += amount
            
            # TDS tracking
            if "tds" in desc_lower:
                tds_paid += amount
            
            # Loan payment reminders
            if txn_type == 'loan_payment':
                date = txn["created_at"][:10]
                reminders.append(f"Loan payment was due on {date} for ₹{amount}")
            
            # Salary identification
            match = salary_pattern.search(desc)
            if match:
                month_str = match.group(1).capitalize()
                year_str = match.group(2)
                try:
                    salary_sources.append({
                        "date": txn["created_at"],
                        "amount": amount,
                        "description": desc
                    })
                except:
                    pass
            
            # Investment tracking
            if any(keyword in desc_lower for keyword in INVESTMENT_KEYWORDS):
                investment_entries.append({
                    "date": txn["created_at"],
                    "amount": amount,
                    "description": desc
                })
            
            # Avoidable expenses
            if (category.lower() in AVOIDABLE_CATEGORIES and amount > 1000):
                avoidable_expenses.append(txn)
                
        except Exception as e:
            print(f"Error processing transaction: {e}")
            continue
    
    return {
        "category_totals": dict(category_totals),
        "monthly_totals": dict(monthly_totals),
        "total_income": total_income,
        "total_expense": total_expense,
        "gst_collected": gst_collected,
        "gst_paid": gst_paid,
        "tds_paid": tds_paid,
        "reminders": reminders,
        "salary_sources": salary_sources,
        "investment_entries": investment_entries,
        "avoidable_expenses": avoidable_expenses,
        "net_savings": total_income - total_expense
    }

def generate_spending_summary(analysis: Dict) -> str:
    """Generate spending summary from pre-analyzed data"""
    if not analysis or not analysis.get("category_totals"):
        return "No spending data available."
    
    summary = "\n".join([
        f"{cat.title()}: ₹{amt:.2f}" 
        for cat, amt in analysis["category_totals"].items()
    ])
    return f"Spending Summary:\n{summary}"

def estimate_advance_tax_optimized(analysis: Dict) -> str:
    """Optimized advance tax calculation"""
    income = analysis.get("total_income", 0)
    expenses = analysis.get("total_expense", 0)
    profit = income - expenses
    
    presumptive_rate = 0.06 if income < 20000000 else 0.08
    taxable = profit * presumptive_rate
    advance_tax = taxable * 0.30
    
    return f"Estimated Advance Tax: ₹{advance_tax:.2f} on presumptive profit ₹{taxable:.2f}"

def gst_summary_optimized(analysis: Dict) -> str:
    """Optimized GST summary"""
    gst_collected = analysis.get("gst_collected", 0)
    gst_paid = analysis.get("gst_paid", 0)
    liability = gst_collected - gst_paid
    
    return f"GST Summary: Collected ₹{gst_collected:.2f}, Paid ₹{gst_paid:.2f}, Net Liability ₹{liability:.2f}"

def simulate_loan_repayment_impact_optimized(analysis: Dict, loans: List[dict]) -> str:
    """Optimized loan repayment simulation"""
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
                f"If you had not spent ₹{alt_payment:.2f} on '{txn['description']}', "
                f"you could have reduced your loan tenure by approximately {int(months_saved)} month(s)."
            )
    
    return "\n".join(messages) if messages else "No significant optimization insights found."

def generate_spending_chart_optimized(analysis: Dict) -> str:
    """Generate spending chart with optimized rendering"""
    category_totals = analysis.get("category_totals", {})
    if not category_totals:
        return ""
    
    # Limit to top 10 categories for better visualization
    sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    categories = [cat for cat, _ in sorted_categories]
    values = [amt for _, amt in sorted_categories]
    
    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color='teal')
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

def calculate_full_year_tax_optimized(analysis: Dict, user_id: str) -> str:
    """Optimized tax calculation using pre-analyzed data"""
    income = analysis.get("total_income", 0)
    
    # Sum investment deductions
    deductions = sum(entry["amount"] for entry in analysis.get("investment_entries", []))
    tds_paid = analysis.get("tds_paid", 0)
    
    standard_deduction = 50000
    taxable_income = max(0, income - deductions - standard_deduction)
    
    # Tax calculation functions
    def calculate_old_regime_tax(taxable):
        if taxable <= 250000:
            return 0
        elif taxable <= 500000:
            return (taxable - 250000) * 0.05
        elif taxable <= 1000000:
            return (250000 * 0.05) + (taxable - 500000) * 0.2
        else:
            return (250000 * 0.05) + (500000 * 0.2) + (taxable - 1000000) * 0.3
    
    def calculate_new_regime_tax(taxable):
        slabs = [(300000, 0), (300000, 0.05), (300000, 0.1), (300000, 0.15), (300000, 0.2)]
        tax = 0
        remaining = taxable
        
        for slab_limit, rate in slabs:
            if remaining <= 0:
                break
            taxable_in_slab = min(remaining, slab_limit)
            tax += taxable_in_slab * rate
            remaining -= taxable_in_slab
        
        if remaining > 0:
            tax += remaining * 0.3
        
        return tax
    
    tax_old = calculate_old_regime_tax(taxable_income)
    tax_new = calculate_new_regime_tax(taxable_income)
    
    # Apply rebate for old regime
    if taxable_income <= 700000:
        tax_old = 0
    
    selected_regime = "Old" if tax_old <= tax_new else "New"
    estimated_tax = min(tax_old, tax_new)
    
    
    summary = {
        "user_id": user_id,
        "financial_year": "2024-25",
        "gross_income": income,
        "deductions": deductions,
        "standard_deduction": standard_deduction,
        "taxable_income": taxable_income,
        "estimated_tax_old_regime": round(tax_old, 2),
        "estimated_tax_new_regime": round(tax_new, 2),
        "selected_regime": selected_regime,
        "final_estimated_tax": round(estimated_tax, 2),
        "tds_paid": tds_paid,
        "salary_entries": analysis.get("salary_sources", []),
        "investment_entries": analysis.get("investment_entries", [])
    }
    
    os.makedirs("reports", exist_ok=True)
    file_path = f"reports/ITR_summary_{user_id}.json"
    with open(file_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return (
        f"🧾 Annual Tax Report (FY 2024–25):\n"
        f"Gross Income: ₹{income:,.2f}\n"
        f"Deductions (80C): ₹{deductions:,.2f}\n"
        f"Standard Deduction: ₹{standard_deduction:,.0f}\n"
        f"Taxable Income: ₹{taxable_income:,.2f}\n"
        f"Estimated Tax (Old Regime): ₹{tax_old:,.2f}\n"
        f"Estimated Tax (New Regime): ₹{tax_new:,.2f}\n"
        f"Selected Regime: {selected_regime}\n"
        f"Final Estimated Tax: ₹{estimated_tax:,.2f}\n"
        f"TDS Paid: ₹{tds_paid:,.2f}\n\n"
        f"✅ Report saved: {file_path}"
    )

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
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})  # Limit results


_chain_cache = None
_retriever_cache = None

def get_or_create_chain() -> Tuple:
    """Get cached chain or create new one"""
    global _chain_cache, _retriever_cache
    
    if _chain_cache is None or _retriever_cache is None:
        _retriever_cache = create_retriever_optimized()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a multilingual financial assistant. Provide concise, accurate responses."),
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
def generate_gamified_dashboard(transactions: List[dict]) -> str:
    from collections import defaultdict
    from matplotlib.patches import Wedge

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


    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(3, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])

    # Circular chart for total spend
    wedges, _ = ax0.pie([total_expense, max(1, total_income - total_expense)],
                        colors=["#1E90FF", "#DDDDDD"], startangle=90)
    ax0.set_title("Spending Overview")
    ax0.text(0, 0, f"₹{total_expense:,.0f}", ha='center', va='center', fontsize=12)

    # Bar for categories
    ax1.barh(cat_labels, cat_values, color="#6495ED")
    ax1.set_title("Top Categories")
    ax1.invert_yaxis()

    # Savings box
    ax2.axis("off")
    ax2.text(0.5, 0.5, f"₹{net_savings:,.0f}\nSavings", ha='center', va='center', fontsize=12, weight='bold', bbox=dict(facecolor='#E0FFFF'))

    # Daily limit box
    ax3.axis("off")
    ax3.text(0.5, 0.5, f"₹{daily_limit:,.0f}\nDaily Limit", ha='center', va='center', fontsize=12, weight='bold', bbox=dict(facecolor='#F5F5DC'))

    # Visual Footer
    ax4.axis("off")
    ax4.text(0.5, 0.5, "📊 Auto-generated by Fin Mentor", ha='center', fontsize=10)

    fig.suptitle("💰 Gamified Financial Dashboard", fontsize=14, weight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def handle_query_optimized(user_input: str, user_id: str = "demo-user", plain_text_mode: bool = False) -> str:
    
    """Optimized query handler with single-pass analysis"""
    user_text = user_input.strip().lower()
    def handle_query_optimized(user_input: str, user_id: str = "demo-user", plain_text_mode: bool = False) -> str:
        user_text = user_input.strip().lower()
    

    if any(k in user_text for k in ["annual tax", "generate tax", "itr", "tax amount", "yearly tax", "final tax", "income tax", "section 14", "tax calculation"]):
        return handle_enhanced_query(user_input, user_id)
    

    if any(word in user_text for word in ["balance", "current balance", "my funds"]):
        balance = get_user_balance(user_id)
        return f"💰 Your current balance is ₹{balance:,.2f}."

    
    if any(word in user_text for word in ["balance", "current balance", "my funds"]):
        balance = get_user_balance(user_id)
        return f"💰 Your current balance is ₹{balance:,.2f}."
    
    transactions = get_user_transactions(user_id)
    loans = get_recent_loans(user_id)
    
    if not transactions:
        return "No transaction data available for analysis."
   
    analysis = analyze_transactions_comprehensive(transactions)
    
    if any(k in user_text for k in ["annual tax", "generate tax", "itr", "tax amount", "yearly tax", "final tax", "income tax"]):
        return calculate_full_year_tax_optimized(analysis, user_id)
    
    if user_text in ["yes", "yeah", "sure", "please do"]:
        return calculate_full_year_tax_optimized(analysis, user_id)
    
    if "tax" in user_text:
        return "Would you like me to generate your full income tax report for FY 2024–25? Just reply with 'Yes'."
    
    chain, retriever = get_or_create_chain()
    
    context_parts = []
    
    if transactions:
        recent_activity = "\n".join([
            f"{t['created_at']}: {t['description']} - ₹{t['amount']}" 
            for t in transactions[:5]
        ])
        context_parts.extend([
            f"Recent Transactions:\n{recent_activity}",
            generate_spending_summary(analysis),
            f"Reminders:\n{chr(10).join(analysis.get('reminders', []))}",
            f"Advance Tax:\n{estimate_advance_tax_optimized(analysis)}",
            f"GST Info:\n{gst_summary_optimized(analysis)}"
        ])
    
    if loans:
        context_parts.append(f"Loan Optimization:\n{simulate_loan_repayment_impact_optimized(analysis, loans)}")
    
    if retriever:
        try:
            docs = retriever.invoke(user_input)
            combined_context = "\n\n".join([doc.page_content for doc in docs])
            context_parts.append(f"Docs:\n{combined_context}")
        except Exception as e:
            print(f"Retriever error: {e}")
    
    context = "\n\n".join(context_parts)
    prompt_template = PromptTemplate.from_template("""
    ### USER REQUEST:
    {query}
    
    ### CONTEXT:
    {context}
    
    ### RESPONSE:
    """)
    
    prompt = prompt_template.format(query=user_input, context=context)
    result = chain.invoke(
        {"input": prompt}, 
        config={"configurable": {"session_id": user_id}}
    )
    
    if any(keyword in user_text for keyword in ["dashboard"]):
        dashboard_img = generate_gamified_dashboard(transactions)
        return f"\n\nGamified Dashboard:\n<img src=\"{dashboard_img}\" width=\"500\"/>"
    if any(keyword in user_text for keyword in ["chart", "graph", "visual", "plot"]):
        chart_img = generate_spending_chart_optimized(analysis)
        if chart_img and not plain_text_mode:
            return result.content + f"\n\nSpending Visualization:\n<img src=\"{chart_img}\" width=\"500\"/>"
    
    return result.content

def handle_query(user_input: str, user_id: str = "demo-user", plain_text_mode: bool = False) -> str:
    return handle_query_optimized(user_input, user_id, plain_text_mode)