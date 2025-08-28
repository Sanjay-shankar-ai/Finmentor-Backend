# enhanced_complaint_handler.py (Updated with Head-wise Deductions and Tax Rebate)
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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_index"
RESOURCE_FOLDER = "Resources"

INCOME_HEAD_KEYWORDS = {
    "salary": {
        "keywords": ["salary", "wages", "allowance", "hra", "da", "bonus", "incentive", "commission", "gratuity", "pension"],
        "patterns": [r"salary\s*[\-–—]?\s*([a-zA-Z]+)\s+(\d{4})", r"pay.*slip", r"monthly.*salary"]
    },
    "house_property": {
        "keywords": ["rent", "rental", "property", "house", "apartment", "maintenance", "municipal", "property tax"],
        "patterns": [r"rent.*received", r"rental.*income", r"house.*rent"]
    },
    "business_profession": {
        "keywords": ["business", "profession", "consultancy", "freelance", "service", "contract", "invoice", "fees"],
        "patterns": [r"professional.*fees", r"consultancy.*income", r"business.*income"]
    },
    "capital_gains": {
        "keywords": ["shares", "stocks", "mutual fund", "sip", "equity", "bond", "securities", "investment", "capital gain"],
        "patterns": [r"share.*sale", r"stock.*profit", r"mutual.*fund.*redemption"]
    },
    "other_sources": {
        "keywords": ["interest", "dividend", "fd", "fixed deposit", "savings", "lottery", "gift", "other income"],
        "patterns": [r"interest.*earned", r"dividend.*received", r"fd.*maturity"]
    }
}

STANDARD_DEDUCTION = 75000
TAX_REBATE_LIMIT = 1200000
TAX_REBATE_AMOUNT = 60000

DEDUCTIONS = {
    "salary": {
        "standard_deduction": 75000,
        "description": "Standard Deduction (applicable only to salary income)"
    },
    "house_property": {
        "standard_deduction": 0.30,
        "description": "30% standard deduction for house property"
    },
    "business_profession": {
        "standard_deduction": 0,
        "description": "No standard deduction (actual business expenses to be claimed)"
    },
    "capital_gains": {
        "standard_deduction": 0,
        "description": "Separate tax treatment for capital gains"
    },
    "other_sources": {
        "standard_deduction": 0,
        "description": "No standard deduction for other sources"
    }
}

NEW_REGIME_SLABS = [
    (400000, 0.0),
    (800000, 0.05),
    (1200000, 0.10),
    (1600000, 0.15),
    (2000000, 0.20),
    (2400000, 0.25),
    (float('inf'), 0.30)
]

@lru_cache(maxsize=128)
def get_cached_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

def classify_transaction_by_section14(description: str, amount: float, txn_type: str) -> str:
    desc_lower = description.lower()
    
    for head, criteria in INCOME_HEAD_KEYWORDS.items():
        if any(keyword in desc_lower for keyword in criteria["keywords"]):
            return head
        
        for pattern in criteria["patterns"]:
            if re.search(pattern, desc_lower, re.IGNORECASE):
                return head
    
    if txn_type == "received":
        return "other_sources"
    
    return "expense"

def calculate_head_wise_taxable_income(income_heads: Dict[str, float]) -> Dict:
    head_wise_calculation = {}
    total_taxable_income = 0
    
    for head, gross_income in income_heads.items():
        if gross_income <= 0:
            continue
            
        deduction_info = DEDUCTIONS.get(head, {"standard_deduction": 0, "description": "No deduction"})
        
        if head == "salary":
            deduction_amount = min(STANDARD_DEDUCTION, gross_income)
            taxable_income = max(0, gross_income - deduction_amount)
            
        elif head == "house_property":
            deduction_amount = gross_income * 0.30
            taxable_income = max(0, gross_income - deduction_amount)
            
        elif head == "business_profession":
            deduction_amount = 0
            taxable_income = gross_income
            
        elif head == "capital_gains":
            deduction_amount = 0
            taxable_income = 0
            
        elif head == "other_sources":
            deduction_amount = 0
            taxable_income = gross_income
            
        else:
            deduction_amount = 0
            taxable_income = gross_income
        
        head_wise_calculation[head] = {
            "gross_income": gross_income,
            "deduction_amount": deduction_amount,
            "taxable_income": taxable_income,
            "deduction_description": deduction_info["description"]
        }
        
        if head != "capital_gains":
            total_taxable_income += taxable_income
    
    return {
        "head_wise_details": head_wise_calculation,
        "total_taxable_income": total_taxable_income
    }

def calculate_tax_liability_enhanced(income_heads: Dict[str, float]) -> Dict:
    def calculate_tax_slab(income: float, slabs: List[Tuple]) -> Tuple[float, List[Dict]]:
        tax = 0.0
        prev_limit = 0
        slab_breakdown = []
        
        for limit, rate in slabs:
            if income <= prev_limit:
                break
            
            taxable_in_slab = min(income - prev_limit, limit - prev_limit)
            tax_in_slab = taxable_in_slab * rate
            tax += tax_in_slab
            
            if taxable_in_slab > 0:
                slab_breakdown.append({
                    "range": f"₹{prev_limit:,} - ₹{min(limit, income):,}",
                    "rate": f"{rate * 100:.0f}%",
                    "taxable_amount": taxable_in_slab,
                    "tax_amount": tax_in_slab
                })
            
            prev_limit = limit
            
            if income <= limit:
                break
        
        return tax, slab_breakdown
    
    head_wise_calc = calculate_head_wise_taxable_income(income_heads)
    total_taxable_income = head_wise_calc["total_taxable_income"]
    
    tax_liability, slab_breakdown = calculate_tax_slab(total_taxable_income, NEW_REGIME_SLABS)
    
    tax_rebate = 0
    if total_taxable_income <= TAX_REBATE_LIMIT:
        tax_rebate = min(TAX_REBATE_AMOUNT, tax_liability)
    
    tax_after_rebate = tax_liability - tax_rebate
    
    cess = tax_after_rebate * 0.04
    total_tax = tax_after_rebate + cess
    
    total_gross_income = sum(income_heads.values())
    
    return {
        "head_wise_calculation": head_wise_calc["head_wise_details"],
        "total_gross_income": total_gross_income,
        "total_taxable_income": total_taxable_income,
        "tax_before_rebate": tax_liability,
        "tax_rebate": tax_rebate,
        "tax_after_rebate": tax_after_rebate,
        "health_education_cess": cess,
        "total_tax_liability": total_tax,
        "effective_tax_rate": (total_tax / total_gross_income * 100) if total_gross_income > 0 else 0,
        "slab_breakdown": slab_breakdown
    }

def calculate_tax_for_specific_income(salary: float, house: float, business: float, other: float) -> str:
    income_heads = {
        "salary": salary,
        "house_property": house,
        "business_profession": business,
        "other_sources": other,
        "capital_gains": 0
    }
    
    tax_calc = calculate_tax_liability_enhanced(income_heads)
    
    report = f"""
🧾 INCOME TAX CALCULATION - NEW REGIME FY 2025-26
{'='*60}

📊 HEAD-WISE INCOME BREAKDOWN & DEDUCTIONS:
{'='*50}
"""
    
    for head, details in tax_calc["head_wise_calculation"].items():
        if details["gross_income"] > 0:
            head_name = head.replace('_', ' ').title()
            report += f"""
{head_name}:
  Gross Income:           ₹{details['gross_income']:,.2f}
  Deduction:              ₹{details['deduction_amount']:,.2f}
  ({details['deduction_description']})
  Taxable Income:         ₹{details['taxable_income']:,.2f}
"""
    
    report += f"""
{'='*60}
TOTAL TAXABLE INCOME:     ₹{tax_calc['total_taxable_income']:,.2f}

💰 TAX CALCULATION:
{'='*20}
"""
    
    for slab in tax_calc['slab_breakdown']:
        report += f"{slab['range']:<25} @ {slab['rate']:<4} = ₹{slab['tax_amount']:,.2f}\n"
    
    report += f"""
{'='*60}
Tax before Rebate:        ₹{tax_calc['tax_before_rebate']:,.2f}
Tax Rebate u/s 87A:       ₹{tax_calc['tax_rebate']:,.2f}
Tax after Rebate:         ₹{tax_calc['tax_after_rebate']:,.2f}
Health & Education Cess (4%): ₹{tax_calc['health_education_cess']:,.2f}
{'='*60}
Total Tax Payable:        ₹{tax_calc['total_tax_liability']:,.2f}
Effective Tax Rate:       {tax_calc['effective_tax_rate']:.2f}%

📅 TAX SLABS (FY 2025-26):
{'='*30}
₹0 - ₹4,00,000:          Nil
₹4,00,001 - ₹8,00,000:   5%
₹8,00,001 - ₹12,00,000:  10%
₹12,00,001 - ₹16,00,000: 15%
₹16,00,001 - ₹20,00,000: 20%
₹20,00,001 - ₹24,00,000: 25%
Above ₹24,00,000:        30%

💡 DEDUCTION SUMMARY:
{'='*25}
• Salary: ₹75,000 standard deduction (max)
• House Property: 30% of rental income
• Business/Profession: No standard deduction (actual expenses)
• Other Sources: No standard deduction
• Capital Gains: Separate tax treatment (excluded)

🎯 TAX REBATE u/s 87A:
{'='*25}
• Available for total income up to ₹12,00,000
• Maximum rebate: ₹60,000
• Rebate amount: Minimum of (Tax liability, ₹60,000)
"""
    
    return report

def get_user_transactions(user_id: str) -> List[dict]:
    try:
        fy_start = datetime.datetime(2024, 4, 1, tzinfo=pytz.UTC).isoformat()
        fy_end = datetime.datetime.now(pytz.UTC).replace(hour=23, minute=59, second=59).isoformat()

        result = (
            supabase
            .table("transactions")
            .select("*")
            .eq("user_id", user_id)
            .gte("created_at", fy_start)
            .lte("created_at", fy_end)
            .order("created_at", desc=True)
            .limit(1000)
            .execute()
        )
        return result.data
    except Exception as e:
        print(f"❌ Error fetching transactions: {e}")
        return []

def analyze_income_by_heads(transactions: List[dict]) -> Dict:
    income_heads = {
        "salary": [],
        "house_property": [],
        "business_profession": [],
        "capital_gains": [],
        "other_sources": []
    }
    
    head_totals = {head: 0.0 for head in income_heads.keys()}
    total_expenses = 0.0
    tds_deducted = 0.0
    
    for txn in transactions:
        try:
            amount = float(txn["amount"])
            desc = txn["description"]
            txn_type = txn.get("type", "")
            
            if txn_type == "received":
                head = classify_transaction_by_section14(desc, amount, txn_type)
                if head in income_heads:
                    income_heads[head].append({
                        "date": txn["created_at"],
                        "description": desc,
                        "amount": amount
                    })
                    head_totals[head] += amount
            
            elif txn_type in ["sent", "expense"]:
                total_expenses += amount
            
            if "tds" in desc.lower() or "tax deducted" in desc.lower():
                tds_deducted += amount
                
        except Exception as e:
            print(f"Error processing transaction: {e}")
            continue
    
    return {
        "income_heads": income_heads,
        "head_totals": head_totals,
        "total_income": sum(head_totals.values()),
        "total_expenses": total_expenses,
        "tds_deducted": tds_deducted,
        "net_savings": sum(head_totals.values()) - total_expenses
    }

def generate_detailed_tax_report(user_id: str, analysis: Dict) -> str:
    income_heads = analysis["income_heads"]
    head_totals = analysis["head_totals"]
    tds_deducted = analysis["tds_deducted"]
    
    tax_calc = calculate_tax_liability_enhanced(head_totals)
    
    report = f"""
🧾 ENHANCED TAX REPORT (FY 2025-26) - NEW REGIME ONLY
{'='*60}

📊 HEAD-WISE INCOME ANALYSIS (Section 14):
{'='*45}
"""
    
    for head, details in tax_calc["head_wise_calculation"].items():
        if details["gross_income"] > 0:
            head_name = head.replace('_', ' ').title()
            report += f"""
{head_name}:
  Gross Income:     ₹{details['gross_income']:,.2f}
  Deduction:        ₹{details['deduction_amount']:,.2f}
  ({details['deduction_description']})
  Taxable Income:   ₹{details['taxable_income']:,.2f}
  
  Recent Transactions:"""
            
            transactions = income_heads[head][:3]
            for txn in transactions:
                report += f"\n    • {txn['description'][:40]}... - ₹{txn['amount']:,.2f}"
            
            report += "\n"
    
    report += f"""
{'='*60}
TOTAL GROSS INCOME:       ₹{tax_calc['total_gross_income']:,.2f}
TOTAL TAXABLE INCOME:     ₹{tax_calc['total_taxable_income']:,.2f}
TDS DEDUCTED:             ₹{tds_deducted:,.2f}

📋 TAX CALCULATION (NEW REGIME FY 2025-26):
{'='*45}
"""
    
    for slab in tax_calc['slab_breakdown']:
        report += f"{slab['range']:<25} @ {slab['rate']:<4} = ₹{slab['tax_amount']:,.2f}\n"
    
    report += f"""
Tax Before Rebate:        ₹{tax_calc['tax_before_rebate']:,.2f}
Tax Rebate u/s 87A:       ₹{tax_calc['tax_rebate']:,.2f}
Tax After Rebate:         ₹{tax_calc['tax_after_rebate']:,.2f}
Health & Education Cess (4%): ₹{tax_calc['health_education_cess']:,.2f}
Total Tax Liability:      ₹{tax_calc['total_tax_liability']:,.2f}
Effective Tax Rate:       {tax_calc['effective_tax_rate']:.2f}%

🎯 HEAD-WISE DEDUCTION BENEFITS:
{'='*35}
• Salary: Standard deduction of ₹75,000 applied
• House Property: 30% standard deduction applied
• Business/Profession: No standard deduction (claim actual expenses)
• Other Sources: No standard deduction available
• Capital Gains: Excluded from regular income tax calculation

🎯 TAX REBATE u/s 87A:
{'='*25}
• Available for total income up to ₹12,00,000
• Maximum rebate: ₹60,000
• Your rebate: ₹{tax_calc['tax_rebate']:,.2f}
"""
    
    return report

def query_transaction_insights(user_query: str, analysis: Dict) -> str:
    query_lower = user_query.lower()
    income_heads = analysis["income_heads"]
    head_totals = analysis["head_totals"]
    
    tax_calc = calculate_tax_liability_enhanced(head_totals)
    
    if any(word in query_lower for word in ["salary", "wages", "pay", "income from job"]):
        salary_details = tax_calc["head_wise_calculation"].get("salary", {})
        
        if salary_details and salary_details["gross_income"] > 0:
            response = f"💼 SALARY INCOME ANALYSIS:\n"
            response += f"Gross Salary Income: ₹{salary_details['gross_income']:,.2f}\n"
            response += f"Standard Deduction: ₹{salary_details['deduction_amount']:,.2f}\n"
            response += f"Taxable Salary: ₹{salary_details['taxable_income']:,.2f}\n\n"
            
            salary_transactions = income_heads.get("salary", [])
            response += "Recent Salary Transactions:\n"
            
            for txn in salary_transactions[:5]:
                response += f"• {txn['date'][:10]}: {txn['description']} - ₹{txn['amount']:,.2f}\n"
            
            monthly_avg = salary_details['gross_income'] / 12
            response += f"\nMonthly Average: ₹{monthly_avg:,.2f}"
            
            return response
        else:
            return "No salary income found in your transactions."
    
    if any(word in query_lower for word in ["property", "house", "rent", "rental"]):
        property_details = tax_calc["head_wise_calculation"].get("house_property", {})
        
        if property_details and property_details["gross_income"] > 0:
            response = f"🏠 HOUSE PROPERTY INCOME ANALYSIS:\n"
            response += f"Gross Property Income: ₹{property_details['gross_income']:,.2f}\n"
            response += f"30% Standard Deduction: ₹{property_details['deduction_amount']:,.2f}\n"
            response += f"Taxable Property Income: ₹{property_details['taxable_income']:,.2f}\n\n"
            
            property_transactions = income_heads.get("house_property", [])
            response += "Property Transactions:\n"
            
            for txn in property_transactions:
                response += f"• {txn['date'][:10]}: {txn['description']} - ₹{txn['amount']:,.2f}\n"
            
            return response
        else:
            return "No house property income found in your transactions."
    
    if any(word in query_lower for word in ["tax", "liability", "how much tax", "tax calculation"]):
        return generate_detailed_tax_report("demo-user", analysis)
    
    if any(word in query_lower for word in ["income", "earnings", "total", "summary"]):
        response = f"📊 ENHANCED INCOME SUMMARY:\n\n"
        
        for head, details in tax_calc["head_wise_calculation"].items():
            if details["gross_income"] > 0:
                head_name = head.replace('_', ' ').title()
                response += f"{head_name}:\n"
                response += f"  Gross: ₹{details['gross_income']:,.2f}\n"
                response += f"  Deduction: ₹{details['deduction_amount']:,.2f}\n"
                response += f"  Taxable: ₹{details['taxable_income']:,.2f}\n\n"
        
        response += f"Total Gross Income: ₹{tax_calc['total_gross_income']:,.2f}\n"
        response += f"Total Taxable Income: ₹{tax_calc['total_taxable_income']:,.2f}\n"
        response += f"Tax Before Rebate: ₹{tax_calc['tax_before_rebate']:,.2f}\n"
        response += f"Tax Rebate: ₹{tax_calc['tax_rebate']:,.2f}\n"
        response += f"Tax Liability: ₹{tax_calc['total_tax_liability']:,.2f}\n"
        response += f"Effective Tax Rate: {tax_calc['effective_tax_rate']:.2f}%"
        
        return response
    
    return "I can help you with queries about salary, property income, business income, capital gains, other sources, tax calculations, and income summaries with head-wise deductions!"

def handle_enhanced_query(user_input: str, user_id: str = "demo-user") -> str:
    transactions = get_user_transactions(user_id)
    
    if not transactions:
        return "No transaction data available for analysis."
    
    analysis = analyze_income_by_heads(transactions)
    
    if any(keyword in user_input.lower() for keyword in ["detailed tax", "full tax report", "tax calculation", "itr"]):
        return generate_detailed_tax_report(user_id, analysis)
    
    response = query_transaction_insights(user_input, analysis)
    
    return response

def handle_query(user_input: str, user_id: str = "demo-user", plain_text_mode: bool = False) -> str:
    return handle_enhanced_query(user_input, user_id)

def calculate_tax_for_income_heads(salary: float = 0, house: float = 0, business: float = 0, other: float = 0) -> str:
    return calculate_tax_for_specific_income(salary, house, business, other)