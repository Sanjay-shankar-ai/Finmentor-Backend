# complaint_handler.py (Restored Full Version with LangChain 0.2+ Compatibility)
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
from typing import List
import datetime
import re
import matplotlib.pyplot as plt
import io
import base64

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_index"
RESOURCE_FOLDER = "Resources"

prompt_template = PromptTemplate.from_template("""
### USER REQUEST:
{query}

### INSTRUCTION:
Act as a multilingual, human-like financial assistant and virtual tax advisor.
Understand the user request and respond contextually using financial data, reminders, document insights, and intelligent routing.

### CONTEXT:
{context}

### RESPONSE:
""")

from datetime import datetime
import pytz
from typing import List

def get_user_transactions(user_id: str) -> List[dict]:
    try:
        fy_start = datetime(2024, 4, 1, tzinfo=pytz.UTC).isoformat()
        fy_end = datetime.now(pytz.UTC).replace(hour=23, minute=59, second=59).isoformat()

        result = (
            supabase
            .table("transactions")
            .select("*")
            .eq("user_id", user_id)
            .gte("created_at", fy_start)
            .lte("created_at", fy_end)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data
    except Exception as e:
        print(f"❌ Error fetching transactions: {e}")
        return []

def get_recent_loans(user_id: str) -> List[dict]:
    try:
        result = supabase.table("loan_info").select("*").eq("user_id", user_id).execute()
        return result.data
    except:
        return []

def analyze_spending(transactions: List[dict]) -> str:
    category_totals = {}
    food_keywords = ["zomato", "swiggy", "domino", "pizza", "food"]
    for txn in transactions:
        desc = txn["description"].lower()
        amt = float(txn["amount"])
        cat = "food" if any(k in desc for k in food_keywords) else txn["category"]
        category_totals[cat] = category_totals.get(cat, 0) + amt
    summary = "\n".join([f"{cat.title()}: ₹{amt:.2f}" for cat, amt in category_totals.items()])
    return f"Spending Summary:\n{summary}"

def generate_spending_chart(transactions: List[dict]) -> str:
    category_totals = {}
    food_keywords = ["zomato", "swiggy", "domino", "pizza", "food"]
    for txn in transactions:
        desc = txn["description"].lower()
        amt = float(txn["amount"])
        cat = "food" if any(k in desc for k in food_keywords) else txn["category"]
        category_totals[cat] = category_totals.get(cat, 0) + amt
    if not category_totals:
        return ""
    categories = list(category_totals.keys())
    values = list(category_totals.values())
    plt.figure(figsize=(6, 4))
    plt.bar(categories, values, color='teal')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Amount (₹)')
    plt.title('Spending by Category')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def extract_due_dates(transactions: List[dict]) -> List[str]:
    reminders = []
    for txn in transactions:
        if txn['type'] == 'loan_payment':
            date = txn["created_at"][:10]
            reminders.append(f"Loan payment was due on {date} for ₹{txn['amount']}")
    return reminders

def estimate_advance_tax(transactions: List[dict]) -> str:
    income = sum(float(txn['amount']) for txn in transactions if txn['type'] == 'received')
    expenses = sum(float(txn['amount']) for txn in transactions if txn['type'] in ['sent', 'expense'])
    profit = income - expenses
    presumptive_rate = 0.06 if income < 20000000 else 0.08
    taxable = profit * presumptive_rate
    advance_tax = taxable * 0.30
    return f"Estimated Advance Tax: ₹{advance_tax:.2f} on presumptive profit ₹{taxable:.2f}"

def generate_tax_checklist(user_id: str) -> str:
    return ("Checklist: PAN, GST summary, P&L sheet, investment proofs, Form 26AS. Rental receipts or trade logs may also be needed.")

def gst_summary(transactions: List[dict]) -> str:
    gst_collected = sum(float(txn['amount']) for txn in transactions if txn['category'].lower() == 'gst_collected')
    gst_paid = sum(float(txn['amount']) for txn in transactions if txn['category'].lower() == 'gst_paid')
    liability = gst_collected - gst_paid
    return f"GST Summary: Collected ₹{gst_collected:.2f}, Paid ₹{gst_paid:.2f}, Net Liability ₹{liability:.2f}"

def simulate_loan_repayment_impact(transactions: List[dict], loans: List[dict]) -> str:
    if not loans:
        return "No loan data found."
    avoidable = [txn for txn in transactions if txn['category'].lower() in ['entertainment', 'shopping', 'luxury'] and float(txn['amount']) > 1000]
    if not avoidable:
        return "No high-value avoidable expenses found."
    loan = loans[0]
    monthly_interest = float(loan['monthly_interest'])
    messages = []
    for txn in avoidable[:3]:
        alt_payment = float(txn['amount'])
        months_saved = alt_payment / monthly_interest if monthly_interest else 0
        if months_saved >= 1:
            messages.append(f"If you had not spent ₹{alt_payment:.2f} on '{txn['description']}', you could have reduced your loan tenure by approximately {int(months_saved)} month(s).")
    return "\n".join(messages) if messages else "No significant optimization insights found."

def load_documents():
    if not os.path.exists(RESOURCE_FOLDER):
        return []
    documents = []
    for filename in os.listdir(RESOURCE_FOLDER):
        path = os.path.join(RESOURCE_FOLDER, filename)
        if filename.endswith(".txt"):
            documents.extend(TextLoader(path, encoding='utf-8').load())
        elif filename.endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())
    return documents

def create_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    
    # If vectorstore already exists, load it
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("🔁 Loading existing ChromaDB...")
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    else:
        print("🆕 Creating new ChromaDB...")
        docs = load_documents()
        if not docs:
            print("⚠️ No documents found.")
            return None
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(texts, embedding=embeddings, persist_directory=CHROMA_PATH)
        vectorstore.persist()
        print("✅ ChromaDB created and saved.")

    return vectorstore.as_retriever()

def create_chain():
    retriever = create_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a multilingual, multimodal AI assistant. Answer only what's asked and route intelligently."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])
    base_chain = prompt | llm
    def get_memory(session_id: str):
        return InMemoryChatMessageHistory()
    return RunnableWithMessageHistory(
        base_chain,
        get_memory,
        input_messages_key="input",
        history_messages_key="history"
    ), retriever
def get_user_balance(user_id: str) -> float:
    try:
        result = supabase.table("users").select("balance").eq("id", user_id).single().execute()
        return float(result.data['balance']) if result.data and 'balance' in result.data else 0.0
    except:
        return 0.0

def handle_query(user_input: str, user_id: str = "demo-user", plain_text_mode: bool = False) -> str:
    chain, retriever = create_chain()
    transactions = get_user_transactions(user_id)
    loans = get_recent_loans(user_id)
    context_parts = []

    if transactions:
        recent_activity = "\n".join([f"{t['created_at']}: {t['description']} - ₹{t['amount']}" for t in transactions[:5]])
        context_parts.append(f"Recent Transactions:\n{recent_activity}")
        context_parts.append(analyze_spending(transactions))
        context_parts.append("Reminders:\n" + "\n".join(extract_due_dates(transactions)))
        context_parts.append("Tax Estimation:\n" + estimate_advance_tax(transactions))
        context_parts.append("Checklist:\n" + generate_tax_checklist(user_id))
        context_parts.append("GST Info:\n" + gst_summary(transactions))

    if transactions and loans:
        context_parts.append("Loan Optimization:\n" + simulate_loan_repayment_impact(transactions, loans))

    if retriever:
        docs = retriever.invoke(user_input)
        combined_context = "\n\n".join([doc.page_content for doc in docs])
        context_parts.append("Docs:\n" + combined_context)

    context = "\n\n".join(context_parts)
    prompt = prompt_template.format(query=user_input, context=context)
    result = chain.invoke({"input": prompt}, config={"configurable": {"session_id": user_id}})

    # only show chart if user asks for it
    show_chart = any(keyword in user_input.lower() for keyword in ["chart", "graph", "visual", "plot"])
    chart_img = generate_spending_chart(transactions) if show_chart else None

    if chart_img and not plain_text_mode:
        return result.content + f"\n\nSpending Visualization:\n<img src=\"{chart_img}\" width=\"500\"/>"
    elif chart_img and plain_text_mode:
        return result.content + "\n\n[Spending chart available on the web dashboard]"
    else:
        return result.content
def calculate_full_year_tax(transactions: list, user_id: str) -> str:
    from datetime import datetime
    from dateutil import parser

    import pytz
    import re
    import json

    fy_start = datetime(2024, 4, 1, tzinfo=pytz.UTC)
    fy_end = datetime(2025, 3, 31, 23, 59, 59, tzinfo=pytz.UTC)
    
    income = 0
    deductions = 0
    tds_paid = 0
    salary_sources = []
    investment_entries = []
    print(len(transactions[5]))
    count = 0

# Loop through all transactions
    for txn in transactions:
        try:
            txn_date = parser.isoparse(txn["created_at"])
            if fy_start <= txn_date <= fy_end:
                print(f"📝 Description: {txn['description']} — 📅 Date: {txn['created_at']}")
                count += 1
        except Exception as e:
            print("⚠️ Skipped due to error:", e)

    print(f"\n✅ Total transactions in FY 2024–25: {count}")
    for txn in transactions:
        
        amount = float(txn["amount"])
        desc = txn["description"]
        desc_lower = desc.lower()
        


        # 👇 More reliable pattern
        match = re.search(r"(?i)salary\s*[\-–—]?\s*([a-zA-Z]+)\s+(\d{4})", desc)
        if match:
            month_str = match.group(1).capitalize()
            year_str = match.group(2)
            try:
                txn_date = datetime.strptime(f"{month_str} {year_str}", "%B %Y").replace(tzinfo=pytz.UTC)
                if fy_start <= txn_date <= fy_end:
                    income += amount
                    salary_sources.append({
                        "date": txn["created_at"],
                        "amount": amount,
                        "description": desc
                    })
                    print(f"✅ Counted: {desc} — ₹{amount}")
            except Exception as e:
                print("❌ Date parse failed for:", desc)
                continue

        # 💸 Investment Deductions
        if any(k in desc for k in ["lic", "elss", "nps", "ppf", "insurance"]):
            deductions += amount
            investment_entries.append({
                "date": txn["created_at"],
                "amount": amount,
                "description": txn["description"]
            })

        # 🏦 TDS
        if "tds" in desc:
            tds_paid += amount

    standard_deduction = 50000
    taxable_income = max(0, income - deductions - standard_deduction)

    # Old Regime
    tax_old = 0
    if taxable_income <= 250000:
        tax_old = 0
    elif taxable_income <= 500000:
        tax_old = (taxable_income - 250000) * 0.05
    elif taxable_income <= 1000000:
        tax_old = (250000 * 0.05) + (taxable_income - 500000) * 0.2
    else:
        tax_old = (250000 * 0.05) + (500000 * 0.2) + (taxable_income - 1000000) * 0.3
    if taxable_income <= 700000:
        tax_old = 0  # 87A rebate

    # New Regime
    tax_new = 0
    slabs = [(300000, 0), (300000, 0.05), (300000, 0.1), (300000, 0.15), (300000, 0.2), (float('inf'), 0.3)]
    temp_income = taxable_income
    for slab, rate in slabs:
        if temp_income > 0:
            slab_amt = min(temp_income, slab)
            tax_new += slab_amt * rate
            temp_income -= slab_amt
        else:
            break

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
        "salary_entries": salary_sources,
        "investment_entries": investment_entries
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
        f"✅ JSON ready for ITR uploaded: {file_path}"
    )
def generate_monthly_comparison_chart(transactions: List[dict]) -> str:
    from collections import defaultdict

    # Group by month
    monthly_totals = defaultdict(float)
    for txn in transactions:
        created = txn["created_at"]
        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        month_key = dt.strftime("%B %Y")
        monthly_totals[month_key] += float(txn["amount"])

    # Sort and select last 2
    sorted_months = sorted(monthly_totals.keys(), key=lambda x: datetime.strptime(x, "%B %Y"))[-2:]
    if len(sorted_months) < 2:
        return ""

    values = [monthly_totals[month] for month in sorted_months]
    plt.figure(figsize=(6, 4))
    plt.bar(sorted_months, values, color=['#4682B4', '#87CEFA'])
    plt.title("Spending Comparison (Last 2 Months)")
    plt.ylabel("Amount (₹)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
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

    # Sort categories by spend
    sorted_cats = sorted(category_totals.items(), key=lambda x: -x[1])[:3]
    cat_labels = [f"{cat}" for cat, amt in sorted_cats]
    cat_values = [amt for cat, amt in sorted_cats]

    # Begin figure
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
    ax2.text(0.5, 0.5, f"₹{net_savings:,.0f}\nTheoretical Savings", ha='center', va='center', fontsize=12, weight='bold', bbox=dict(facecolor='#E0FFFF'))

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


def handle_query(user_input: str, user_id: str = "demo-user", plain_text_mode: bool = False) -> str:
    chain, retriever = create_chain()
    transactions = get_user_transactions(user_id)
    loans = get_recent_loans(user_id)
    context_parts = []

    # Normalize input
    user_text = user_input.strip().lower()
    if any(word in user_text for word in ["balance", "current balance", "my funds"]):
        balance = get_user_balance(user_id)
        return f"\U0001F4B0 Your current balance is ₹{balance:,.2f}."

    if any(k in user_text for k in ["annual tax", "generate tax", "itr", "tax amount", "yearly tax", "final tax", "income tax"]):
        return calculate_full_year_tax(transactions, user_id)

    if user_text in ["yes", "yeah", "sure", "please do"]:
        return calculate_full_year_tax(transactions, user_id)

    if "tax" in user_text:
        return "Would you like me to generate your full income tax report for FY 2024–25? Just reply with 'Yes'."

    if transactions:
        recent_activity = "\n".join([f"{t['created_at']}: {t['description']} - ₹{t['amount']}" for t in transactions[:5]])
        context_parts.append(f"Recent Transactions:\n{recent_activity}")
        context_parts.append(analyze_spending(transactions))
        context_parts.append("Reminders:\n" + "\n".join(extract_due_dates(transactions)))
        context_parts.append("Advance Tax:\n" + estimate_advance_tax(transactions))
        context_parts.append("Checklist:\n" + generate_tax_checklist(user_id))
        context_parts.append("GST Info:\n" + gst_summary(transactions))

    if transactions and loans:
        context_parts.append("Loan Optimization:\n" + simulate_loan_repayment_impact(transactions, loans))

    if retriever:
        docs = retriever.invoke(user_input)
        combined_context = "\n\n".join([doc.page_content for doc in docs])
        context_parts.append("Docs:\n" + combined_context)

    context = "\n\n".join(context_parts)
    prompt = prompt_template.format(query=user_input, context=context)
    result = chain.invoke({"input": prompt}, config={"configurable": {"session_id": user_id}})

    if any(keyword in user_text for keyword in ["dashboard"]):
        dashboard_img = generate_gamified_dashboard(transactions)
        return result.content + f"\n\nGamified Dashboard:\n<img src=\"{dashboard_img}\" width=\"500\"/>"

    if any(keyword in user_text for keyword in ["comparison", "compare", "visual", "monthly", "difference"]):
        comparison_img = generate_monthly_comparison_chart(transactions)
        return result.content + f"\n\nSpending Comparison:\n<img src=\"{comparison_img}\" width=\"500\"/>"
        

    return result.content
