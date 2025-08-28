import os, datetime, re, io, base64, json
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from supabase import create_client

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ---------- Book3 tax bridge ----------
try:
    from book3 import (
        handle_enhanced_query,
        analyze_income_by_heads,
        calculate_tax_liability_enhanced,
        generate_detailed_tax_report,
        calculate_tax_for_specific_income,
        get_user_transactions as get_tax_transactions,
        query_transaction_insights,
    )
    TAX_MODULE_AVAILABLE = True
except ImportError:
    TAX_MODULE_AVAILABLE = False

load_dotenv()
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # optional (only for live investment)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
llm        = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

EMBEDDING_PATH   = "./chroma_index"
RESOURCE_FOLDER  = "Resources"
FOOD_KEYWORDS    = {"zomato","swiggy","domino","pizza","food","restaurant","cafe","meal","dining"}
INVEST_KEYWORDS  = {"sip","mutual fund","elss","ppf","nps","fd","gold","etf","stocks","bonds"}
AVOIDABLE        = {"entertainment","shopping","luxury","gaming","subscription"}

# ---------- UTIL ----------
@lru_cache(maxsize=32)
def get_user_balance(user_id: str) -> float:
    try:
        return float(supabase.table("users").select("balance").eq("id", user_id).single().execute().data["balance"])
    except:
        return 0.0

def get_user_transactions(user_id: str, limit: int = 5000) -> List[dict]:
    try:
        return (supabase.table("transactions")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit).execute().data)
    except Exception as e:
        print("❌ fetch error:", e)
        return []

def analyse_txns(txns: List[dict]) -> dict:
    cat_tot, monthly, yearly = defaultdict(float), defaultdict(lambda: {"in": 0, "out": 0}), defaultdict(float)
    invest, income, exp = [], 0, 0
    for t in txns:
        amt, desc, typ, cat = float(t["amount"]), t["description"].lower(), t.get("type"), t.get("category", "Other")
        if any(k in desc for k in FOOD_KEYWORDS): cat = "food"
        if any(k in desc for k in INVEST_KEYWORDS):
            cat, invest = "investment", invest + [{"date": t["created_at"], "amt": amt, "desc": t["description"]}]
        cat_tot[cat] += amt
        month_key = t["created_at"][:7]
        if typ == "received":
            income += amt
            monthly[month_key]["in"] += amt
        else:
            exp += amt
            monthly[month_key]["out"] += amt
        yearly[t["created_at"][:4]] += amt
    return {"cat_tot": dict(cat_tot), "monthly": dict(monthly), "yearly": dict(yearly),
            "invest": invest, "net": income - exp, "income": income, "exp": exp}

# ---------- VIS ----------
def pie_base64(labels, sizes, title):
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close()
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def bar_base64(labels, values, title, color='teal'):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=labels, y=values, palette=[color] * len(labels))
    plt.xticks(rotation=45, ha='right'); plt.title(title)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close()
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
