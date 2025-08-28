import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List

# --- Constants ---
LOAN_OPTIMIZATION_BASE_CATEGORIES = {
    'food': 0.10,
    'entertainment': 0.05,
    'shopping': 0.06,
    'travel': 0.05,
    'health': 0.10,
    'festival': 0.05,
    'emergency': 0.10,
    'investment': 0.10,
    'gifts': 0.05,
    'utilities': 0.10,
    'salary': 0.0  # Flagged as necessary income
}

LOAN_OPTIMIZATION_UNNECESSARY_KEYWORDS = ['entertainment', 'shopping', 'movies', 'party', 'gadgets', 'luxury', 'travel', 'gaming', 'subscriptions', 'gifts']
LOAN_OPTIMIZATION_NECESSARY_KEYWORDS = ['rent', 'food', 'health', 'education', 'festival', 'emergency', 'investment', 'utilities', 'salary', 'income']

LOAN_OPTIMIZATION_DEFAULT_LOAN_AMOUNT = 1200000
LOAN_OPTIMIZATION_DEFAULT_INTEREST_RATE = 0.09
LOAN_OPTIMIZATION_DEFAULT_LOAN_YEARS = 5

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

def enhance_dataframe(df: pd.DataFrame, include_columns: List[str] = None, is_schedule: bool = False) -> pd.DataFrame:
    """Enhance DataFrame with user-friendly formatting"""
    df_enhanced = df.copy()
    
    # Rename columns for clarity
    column_map = {
        'transaction_date': 'Date',
        'category': 'Category',
        'amount': 'Amount (₹)',
        'description': 'Description',
        'type': 'Type',
        'month': 'Month',
        'necessary': 'Is Necessary',
        'non_essential_amount': 'Non-Essential Amount (₹)',
        'necessity_flag': 'Necessity',
        'Year': 'Year',
        'Month': 'Month',
        'Beginning Balance': 'Starting Balance (₹)',
        'EMI': 'EMI (₹)',
        'Principal': 'Principal (₹)',
        'Interest': 'Interest (₹)',
        'Ending Balance': 'Ending Balance (₹)',
        'Salary': ' Nonlinear (₹)',
        'Loan_EMI': 'Loan EMI (₹)',
        'Remaining_Salary': 'Remaining Salary (₹)',
        'Unnecessary_Spending': 'Unnecessary Spending (₹)',
        'Advice': 'Advice'
    }
    df_enhanced.rename(columns={k: v for k, v in column_map.items() if k in df_enhanced.columns}, inplace=True)
    
    # Format amount columns
    amount_columns = ['Amount (₹)', 'Non-Essential Amount (₹)', 'Starting Balance (₹)', 'EMI (₹)', 'Principal (₹)', 'Interest (₹)', 
                     'Ending Balance (₹)', 'Salary (₹)', 'Loan EMI (₹)', 'Remaining Salary (₹)', 'Unnecessary Spending (₹)']
    for col in df_enhanced.columns:
        if any(amt_col in col for amt_col in amount_columns):
            if df_enhanced[col].dtype in ['float64', 'int64']:
                df_enhanced[col] = df_enhanced[col].apply(lambda x: format_currency(x) if pd.notnull(x) else '₹0.00')
    
    # Format dates
    if 'Date' in df_enhanced.columns:
        df_enhanced['Date'] = pd.to_datetime(df_enhanced['Date'], errors='coerce').dt.strftime('%b %d, %Y')
    
    # Format schedule-specific date (Year and Month columns)
    if is_schedule and 'Year' in df_enhanced.columns and 'Month' in df_enhanced.columns:
        df_enhanced['Year'] = df_enhanced['Year'].astype(str)
        df_enhanced['Month'] = df_enhanced['Month'].apply(lambda x: x.strip())
        df_enhanced['Date'] = df_enhanced['Month'] + ' ' + df_enhanced['Year'].str.replace('.0', '')
        df_enhanced = df_enhanced.drop(columns=['Year', 'Month'], errors='ignore')
    
    # Format boolean
    if 'Is Necessary' in df_enhanced.columns:
        df_enhanced['Is Necessary'] = df_enhanced['Is Necessary'].apply(lambda x: 'Yes' if x else 'No')
    
    # Select only specified columns if provided
    if include_columns:
        df_enhanced = df_enhanced[[col for col in include_columns if col in df_enhanced.columns]]
    
    return df_enhanced

def loan_optimization_pipeline(user_id: str, user_prompt: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main entry point for the loan optimization. Takes user ID, prompt, and raw transaction data.
    Converts data to DataFrame, runs analysis, and formats output as markdown tables.
    """
    try:
        if not transactions:
            return {"error": "No transaction data provided for loan optimization."}

        # 1. Convert transaction list (from Supabase) to Pandas DataFrame
        df_data = []
        for txn in transactions:
            df_data.append({
                'transaction_date': txn.get('created_at', ''),
                'category': txn.get('category', 'Other'),
                'amount': float(txn.get('amount', 0.0)),
                'description': txn.get('description', ''),
                'type': txn.get('type', 'unknown')
            })
        df = pd.DataFrame(df_data)

        if df.empty:
            return {"error": "Unable to process transaction data for loan optimization."}

        # 2. Run the core analysis logic
        print("🔄 Running core loan optimization logic...")
        results = _run_loan_optimization_core(user_id, user_prompt, df)
        print("✅ Loan optimization logic completed.")
        
        # 3. Format results as markdown tables
        output = []

        # Lifestyle Information
        output.append(f"### Loan Optimization Results\n\n**Lifestyle Detected**: {results['lifestyle']}\n")
        output.append(f"**Monthly EMI**: {format_currency(results['monthly_emi'])}\n")

        # Categorized Transactions
        if not results['categorized_transactions'].empty:
            categorized_df = enhance_dataframe(results['categorized_transactions'], 
                include_columns=['Date', 'Category', 'Description', 'Type', 'Amount (₹)', 'Necessity', 'Non-Essential Amount (₹)'])
            output.append("### Categorized Transactions\n")
            output.append(categorized_df.head(20).to_markdown(index=False, tablefmt='grid'))
            output.append(f"\n*Showing first 20 of {len(results['categorized_transactions'])} transactions*")
            output.append("\n")

        # Monthly Summary
        if not results['monthly_summary'].empty:
            summary_df = enhance_dataframe(results['monthly_summary'], 
                include_columns=['Month', 'Salary (₹)', 'Loan EMI (₹)', 'Remaining Salary (₹)', 'Unnecessary Spending (₹)', 'Advice'])
            output.append("### Monthly Summary\n")
            output.append(summary_df.to_markdown(index=False, tablefmt='grid'))
            output.append("\n")

        # Normal Repayment Schedule
        if not results['normal_schedule'].empty:
            normal_schedule_df = enhance_dataframe(results['normal_schedule'], 
                include_columns=['Date', 'Starting Balance (₹)', 'EMI (₹)', 'Principal (₹)', 'Interest (₹)', 'Ending Balance (₹)'],
                is_schedule=True)
            output.append("### Standard Loan Repayment Schedule\n")
            output.append(normal_schedule_df.head(12).to_markdown(index=False, tablefmt='grid'))
            output.append(f"\n*Showing first 12 months of {len(normal_schedule_df)} total payments*")
            output.append("\n")

        # Optimized Repayment Schedule
        if not results['optimized_schedule'].empty:
            optimized_schedule_df = enhance_dataframe(results['optimized_schedule'], 
                include_columns=['Date', 'Starting Balance (₹)', 'EMI (₹)', 'Principal (₹)', 'Interest (₹)', 'Ending Balance (₹)'],
                is_schedule=True)
            output.append("### Optimized Loan Repayment Schedule\n")
            output.append(optimized_schedule_df.head(12).to_markdown(index=False, tablefmt='grid'))
            output.append(f"\n*Showing first 12 months of {len(optimized_schedule_df)} total payments*")
            output.append("\n")

        # Summary Insights
        total_unnecessary = results['categorized_transactions']['non_essential_amount'].sum() if not results['categorized_transactions'].empty else 0.0
        output.append("### Key Insights\n")
        output.append(f"- **Total Unnecessary Spending**: {format_currency(total_unnecessary)}\n")
        output.append(f"- **Loan Amount**: {format_currency(LOAN_OPTIMIZATION_DEFAULT_LOAN_AMOUNT)}\n")
        output.append(f"- **Interest Rate**: {LOAN_OPTIMIZATION_DEFAULT_INTEREST_RATE*100:.1f}%\n")
        output.append(f"- **Loan Term**: {LOAN_OPTIMIZATION_DEFAULT_LOAN_YEARS} years\n")
        output.append("- **Recommendation**: By redirecting unnecessary spending to extra loan payments, you can reduce the loan term, as shown in the optimized schedule.")

        results['formatted_output'] = "\n".join(output)
        return results

    except Exception as e:
        print(f"❌ Error in loan_optimization_pipeline: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"An error occurred during loan optimization: {str(e)}"}

def _run_loan_optimization_core(user_id: str, user_prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
    """The core loan optimization logic provided."""
    BASE_CATEGORIES = LOAN_OPTIMIZATION_BASE_CATEGORIES
    UNNECESSARY_KEYWORDS = LOAN_OPTIMIZATION_UNNECESSARY_KEYWORDS
    NECESSARY_KEYWORDS = LOAN_OPTIMIZATION_NECESSARY_KEYWORDS
    LOAN_AMOUNT = LOAN_OPTIMIZATION_DEFAULT_LOAN_AMOUNT
    INTEREST_RATE = LOAN_OPTIMIZATION_DEFAULT_INTEREST_RATE
    LOAN_YEARS = LOAN_OPTIMIZATION_DEFAULT_LOAN_YEARS

    def get_best_match(text, keyword_list):
        if not text or not keyword_list or pd.isna(text):
            return None, 0.0
        try:
            text_str = str(text)
            vectorizer = TfidfVectorizer().fit([text_str] + keyword_list)
            vectors = vectorizer.transform([text_str] + keyword_list)
            cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            if len(cosine_sim) == 0:
                return None, 0.0
            max_sim = cosine_sim.max()
            matched_keyword = keyword_list[cosine_sim.argmax()] if max_sim > 0.3 else None
            return matched_keyword, max_sim
        except Exception as e:
            print(f"Error in get_best_match: {e}")
            return None, 0.0

    def categorize_nlp(text):
        if pd.isna(text):
            return 'Necessary'
        text_clean = re.sub(r"[^a-zA-Z ]", "", str(text)).lower()
        if 'salary' in text_clean or 'income' in text_clean:
            return 'Necessary'
        necessary_match, necessary_sim = get_best_match(text_clean, NECESSARY_KEYWORDS)
        unnecessary_match, unnecessary_sim = get_best_match(text_clean, UNNECESSARY_KEYWORDS)
        if necessary_match and (necessary_sim >= unnecessary_sim):
            return 'Necessary'
        elif unnecessary_match:
            return 'Not Necessary'
        return 'Necessary'

    def infer_lifestyle(df):
        if df.empty:
            return 'Unmarried_but_Living_with_family'
        categories = df['category'].str.lower().fillna('').tolist()
        if any("school" in str(c) or "education" in str(c) for c in categories):
            return 'Married_Living_with_family_with_children'
        if any("rent" in str(c) for c in categories) and any("food" in str(c) for c in categories):
            return 'Bachelor_Living_Alone'
        return 'Unmarried_but_Living_with_family'
    
    def dynamic_threshold_adjuster(salary, lifestyle):
        P = LOAN_AMOUNT
        R = INTEREST_RATE / 12
        N = LOAN_YEARS * 12

        if R == 0:
            emi = P / N if N > 0 else 0
        else:
            emi = (P * R * (1 + R)**N) / ((1 + R)**N - 1)
        
        remaining_amount = max(salary - emi, 0)
        adjustments = BASE_CATEGORIES.copy()

        if lifestyle == 'Unmarried_but_Living_with_family':
            adjustments['food'] -= 0.01
            adjustments['festival'] -= 0.02
            adjustments['entertainment'] += 0.01
        elif lifestyle == 'Bachelor_Living_Alone':
            adjustments['food'] += 0.02
            adjustments['entertainment'] += 0.01
        elif lifestyle == 'Married_Living_with_family_with_children':
            adjustments['food'] += 0.02
            adjustments['festival'] += 0.02
            adjustments['health'] += 0.03

        thresholds = {
            cat: max(0, round(remaining_amount * perc, 2))
            for cat, perc in adjustments.items()
            if cat != 'salary'
        }
        return thresholds

    def calculate_not_necessary_spending(df, thresholds):
        df = df.sort_values(by='transaction_date').reset_index(drop=True)
        df['necessary'] = True
        df['non_essential_amount'] = 0.0
        category_cumulative = {cat: 0.0 for cat in thresholds}

        for idx, row in df.iterrows():
            cat = str(row['category']).strip().lower()
            amount = float(row['amount'])
            if cat in thresholds:
                current_total = category_cumulative[cat]
                threshold = thresholds[cat]
                if current_total + amount <= threshold:
                    category_cumulative[cat] += amount
                else:
                    allowed = max(threshold - current_total, 0)
                    non_essential = amount - allowed
                    df.at[idx, 'necessary'] = False
                    df.at[idx, 'non_essential_amount'] = non_essential
                    category_cumulative[cat] = threshold
        return df

    def generate_repayment_schedule(loan_amount, interest_rate, years, extra_annual_payment=0):
        balance = loan_amount
        monthly_interest = interest_rate / 12
        n_months = years * 12
        if monthly_interest == 0:
            emi = loan_amount / n_months if n_months > 0 else 0
        else:
            emi = (balance * monthly_interest * (1 + monthly_interest) ** n_months) / ((1 + monthly_interest) ** n_months - 1)
        
        schedule = []
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        year = 2025
        month = 6
        months_processed = 0
        while balance > 0 and months_processed < n_months:
            interest = balance * monthly_interest if monthly_interest > 0 else 0
            principal = emi - interest if monthly_interest > 0 else emi
            if balance < emi:
                principal = balance
                emi = balance + interest if monthly_interest > 0 else balance
            end_balance = balance - principal
            schedule.append({
                'Year': year,
                'Month': month_names[month % 12],
                'Beginning Balance': round(balance, 2),
                'EMI': round(emi, 2),
                'Principal': round(principal, 2),
                'Interest': round(interest, 2),
                'Ending Balance': round(end_balance, 2)
            })
            balance = end_balance
            months_processed += 1
            month += 1
            if month % 12 == 0 and extra_annual_payment > 0 and balance > 0:
                payment = min(balance, extra_annual_payment)
                balance -= payment
            if month % 12 == 0:
                year += 1
        return pd.DataFrame(schedule)

    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=['transaction_date'])
    df['category'] = df['category'].fillna('Other').str.lower().str.strip()
    df['month'] = df['transaction_date'].dt.to_period('M')

    lifestyle = infer_lifestyle(df)
    schedule_df = generate_repayment_schedule(LOAN_AMOUNT, INTEREST_RATE, LOAN_YEARS)
    monthly_emi = round(schedule_df.iloc[0]['EMI'], 2) if not schedule_df.empty else 0.0

    final_results = []
    monthly_summary = []

    for period, group in df.groupby('month'):
        salary_transactions = group[(group['type'].str.lower() == 'received') & (group['description'].str.contains("salary|income", case=False, na=False))]
        salary = salary_transactions['amount'].sum() if not salary_transactions.empty else 0.0

        remaining = max(salary - monthly_emi, 0)
        thresholds = dynamic_threshold_adjuster(remaining, lifestyle)
        group = calculate_not_necessary_spending(group.copy(), thresholds)
        
        group['necessity_flag'] = group.apply(
            lambda row: 'Necessary - Income' if (row['type'].lower() == 'received' and ('salary' in str(row['description']).lower() or 'income' in str(row['description']).lower()))
            else ('Not Necessary - Threshold Exceed' if not row['necessary'] else categorize_nlp(row['description'])),
            axis=1
        )
        
        unnecessary_spending = group['non_essential_amount'].sum()
        advice = "✅ Good control on spending!"
        if salary > 0 and unnecessary_spending > 0.15 * salary:
            advice = "⚠ High unnecessary expenses. Consider reviewing your budget."
        monthly_summary.append({
            'Month': str(period),
            'Salary': round(salary, 2),
            'Loan_EMI': monthly_emi,
            'Remaining_Salary': remaining,
            'Unnecessary_Spending': unnecessary_spending,
            'Advice': advice
        })
        final_results.append(group)

    if not final_results:
        final_df = pd.DataFrame()
    else:
        final_df = pd.concat(final_results, ignore_index=True)
    summary_df = pd.DataFrame(monthly_summary)
    
    total_extra_payment = final_df['non_essential_amount'].sum() if not final_df.empty else 0.0
    
    optimized_schedule = generate_repayment_schedule(LOAN_AMOUNT, INTEREST_RATE, LOAN_YEARS, extra_annual_payment=total_extra_payment)

    return {
        'user_id': user_id,
        'user_prompt': user_prompt,
        'lifestyle': lifestyle,
        'monthly_emi': monthly_emi,
        'categorized_transactions': final_df,
        'monthly_summary': summary_df,
        'normal_schedule': schedule_df,
        'optimized_schedule': optimized_schedule
    }

# --- Optional: Standalone runner if needed ---
# if __name__ == "__main__":
#     pass