# investment_advisor.py - Smart Investment Advisory Module with Agent Integration
import os
from typing import List, Dict, Optional
from collections import defaultdict
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io

# LangChain imports for agent functionality
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults

class SmartInvestmentAdvisor:
    """
    Smart Investment Advisory System with AI Agent Integration
    Provides personalized investment recommendations with real-time market data
    """
    
    def __init__(self):
        # API Keys (should be moved to environment variables)
        os.environ["TAVILY_API_KEY"] = "TAVILY_API_KEY"
        os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"
        
        # Initialize components
        self.llm = ChatGroq(model="llama3-70b-8192")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.tavily_search = TavilySearchResults(k=3)
        
        # Investment parameters
        self.risk_profiles = {
            'conservative': {'equity': 20, 'debt': 70, 'gold': 10},
            'moderate': {'equity': 50, 'debt': 40, 'gold': 10},
            'aggressive': {'equity': 70, 'debt': 20, 'gold': 10}
        }
        
        self.investment_options = {
            'ELSS': {'risk': 'moderate', 'returns': 12, 'lock_in': 3, 'tax_benefit': True},
            'Large Cap MF': {'risk': 'conservative', 'returns': 10, 'lock_in': 0, 'tax_benefit': False},
            'Mid Cap MF': {'risk': 'aggressive', 'returns': 15, 'lock_in': 0, 'tax_benefit': False},
            'PPF': {'risk': 'conservative', 'returns': 7.1, 'lock_in': 15, 'tax_benefit': True},
            'NSC': {'risk': 'conservative', 'returns': 6.8, 'lock_in': 5, 'tax_benefit': True},
            'FD': {'risk': 'conservative', 'returns': 5.5, 'lock_in': 0, 'tax_benefit': False},
            'Gold ETF': {'risk': 'moderate', 'returns': 8, 'lock_in': 0, 'tax_benefit': False}
        }
        
        # User financial data (will be populated from transactions)
        self.user_finance_data = {}
        self.avg_monthly_leftover = 0
        
        # Initialize agent
        self._setup_agent()

    def _setup_agent(self):
        """Setup the AI agent with tools"""
        tools = [
            Tool(
                name="PersonalizedSIPAdvisor",
                func=self._personalized_sip_advice,
                description="Gives SIP suggestions based on user savings and current SIP trends."
            ),
            Tool(
                name="FinancialSearchTool",
                func=self.tavily_search.run,
                description="Use this tool to search real-time finance topics like best SIPs, investments, etc."
            ),
            Tool(
                name="InvestmentPortfolioAnalyzer",
                func=self._analyze_investment_portfolio,
                description="Analyze user's current investment portfolio and suggest improvements."
            )
        ]
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            memory=self.memory,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False  # Set to True for debugging
        )

    def analyze_transactions_for_investment(self, transactions: List[Dict]) -> Dict:
        """Convert transaction data to monthly finance data for investment analysis"""
        monthly_data = defaultdict(lambda: {'income': 0, 'expenses': 0})
        investment_history = []
        
        for txn in transactions:
            try:
                amount = float(txn.get("amount", 0))
                txn_type = txn.get("type", "")
                description = txn.get("description", "").lower()
                created_at = txn.get("created_at", "")
                
                # Parse date
                try:
                    dt = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    month_key = dt.strftime("%B")
                except:
                    continue
                
                if txn_type == "received":
                    monthly_data[month_key]['income'] += amount
                elif txn_type in ["sent", "expense"]:
                    monthly_data[month_key]['expenses'] += amount
                    
                    # Track existing investments
                    if any(inv_keyword in description for inv_keyword in 
                          ['sip', 'mutual fund', 'elss', 'ppf', 'nsc', 'fd', 'investment']):
                        investment_history.append({
                            'date': created_at,
                            'amount': amount,
                            'description': txn.get('description', ''),
                            'type': self._identify_investment_type(description)
                        })
                        
            except Exception as e:
                continue
        
        # Update user finance data
        self.user_finance_data = dict(monthly_data)
        
        # Calculate key metrics
        total_income = sum(data['income'] for data in monthly_data.values())
        total_expenses = sum(data['expenses'] for data in monthly_data.values())
        total_savings = total_income - total_expenses
        
        return {
            'monthly_data': dict(monthly_data),
            'total_income': total_income,
            'total_expenses': total_expenses,
            'total_savings': total_savings,
            'investment_history': investment_history,
            'avg_monthly_savings': total_savings / max(len(monthly_data), 1)
        }

    def _identify_investment_type(self, description: str) -> str:
        """Identify investment type from description"""
        description = description.lower()
        
        if 'elss' in description:
            return 'ELSS'
        elif 'sip' in description or 'mutual fund' in description:
            return 'Mutual Fund'
        elif 'ppf' in description:
            return 'PPF'
        elif 'fd' in description or 'fixed deposit' in description:
            return 'FD'
        elif 'nsc' in description:
            return 'NSC'
        else:
            return 'Other'

    def _generate_financial_summary(self, _=None) -> str:
        """Generate financial summary from user data"""
        if not self.user_finance_data:
            return "No financial data available for analysis."
        
        total_savings = 0
        for month, values in self.user_finance_data.items():
            income = values.get("income", 0)
            expenses = values.get("expenses", 0)
            leftover = income - expenses
            total_savings += leftover
        
        self.avg_monthly_leftover = total_savings // max(len(self.user_finance_data), 1)
        
        # Calculate SIP returns (12% annual return)
        if self.avg_monthly_leftover > 0:
            monthly_rate = 0.12 / 12  # 12% annual = 1% monthly
            sip_return = round(self.avg_monthly_leftover * (((1 + monthly_rate)**12 - 1) / monthly_rate))
        else:
            sip_return = 0
        
        summary = f"""
📊 Financial Summary:
- Total yearly savings: ₹{total_savings:,}
- Average monthly savings: ₹{self.avg_monthly_leftover:,}

💡 Suggested SIP:
You can start a SIP of ₹{self.avg_monthly_leftover:,}/month.
Estimated return in 1 year (at 12% p.a.): ₹{sip_return:,}
"""
        return summary

    def _personalized_sip_advice(self, _=None) -> str:
        """Generate personalized SIP advice with real-time data"""
        try:
            # Get savings summary
            savings_summary = self._generate_financial_summary()
            
            # Search for current SIP trends
            search_query = "Best SIPs to invest in India 2024 mutual funds"
            search_results = self.tavily_search.run(search_query)
            
            # Format search results
            if isinstance(search_results, list):
                formatted_results = "\n".join(f"- {res}" for res in search_results[:3])
            else:
                formatted_results = str(search_results)[:500]  # Limit length
            
            combined_advice = f"""
🧾 Your Savings Analysis:
{savings_summary}

✅ Personalized Recommendation:
Based on your monthly savings of ₹{self.avg_monthly_leftover:,}, here's my advice:

1. 🎯 **Emergency Fund First**: Keep 6 months expenses (₹{self.avg_monthly_leftover * 6:,}) in liquid funds
2. 💪 **SIP Strategy**: Start with ₹{max(5000, self.avg_monthly_leftover // 2):,}/month in diversified equity funds
3. 🛡️ **Risk Management**: Consider 70% equity, 20% debt, 10% gold allocation
4. 📈 **Tax Saving**: Include ELSS funds for Section 80C benefits

Remember: Start small, stay consistent, and increase SIP amount annually!
"""
            return combined_advice
            
        except Exception as e:
            return f"Error generating SIP advice: {str(e)}"

    def _analyze_investment_portfolio(self, _=None) -> str:
        """Analyze current investment portfolio"""
        if not hasattr(self, 'current_investments') or not self.current_investments:
            return """
📊 Portfolio Analysis:
No existing investments found in your transactions.

💡 Recommendations:
1. Start with a diversified equity mutual fund SIP
2. Consider ELSS for tax benefits
3. Add debt funds for stability
4. Maintain emergency fund in liquid funds
"""
        
        # Analyze existing investments
        portfolio_summary = "📊 Current Investment Portfolio:\n"
        total_investment = 0
        
        for investment in self.current_investments:
            portfolio_summary += f"- {investment['type']}: ₹{investment['amount']:,}\n"
            total_investment += investment['amount']
        
        portfolio_summary += f"\nTotal Investment: ₹{total_investment:,}\n"
        portfolio_summary += "\n💡 Optimization Suggestions:\n"
        portfolio_summary += "- Ensure proper asset allocation based on your risk profile\n"
        portfolio_summary += "- Review and rebalance quarterly\n"
        portfolio_summary += "- Consider tax-efficient instruments\n"
        
        return portfolio_summary

    def get_investment_recommendations(self, transactions: List[Dict]) -> str:
        """Main function to get investment recommendations"""
        # Analyze transactions
        analysis = self.analyze_transactions_for_investment(transactions)
        
        # Store investment history for portfolio analysis
        self.current_investments = analysis.get('investment_history', [])
        
        # Generate comprehensive recommendation using AI agent
        query = f"""
        Based on my financial data:
        - Monthly savings: ₹{analysis['avg_monthly_savings']:,.0f}
        - Total income: ₹{analysis['total_income']:,.0f}
        - Total expenses: ₹{analysis['total_expenses']:,.0f}
        
        Please provide personalized investment advice including SIP recommendations, 
        portfolio allocation, and current market opportunities.
        """
        
        try:
            response = self.agent.run(query)
            return response
        except Exception as e:
            # Fallback to basic recommendation
            return self._personalized_sip_advice()

    def generate_investment_chart(self, analysis: Dict) -> str:
        """Generate investment allocation chart"""
        try:
            # Sample allocation based on moderate risk profile
            categories = ['Equity Funds', 'Debt Funds', 'Gold ETF', 'Emergency Fund']
            allocation = [50, 30, 10, 10]  # Percentages
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Pie chart for allocation
            ax1.pie(allocation, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Recommended Portfolio Allocation')
            
            # Bar chart for monthly investment
            monthly_savings = analysis.get('avg_monthly_savings', 0)
            investment_amounts = [monthly_savings * (pct/100) for pct in allocation]
            
            ax2.bar(categories, investment_amounts, color=colors, alpha=0.7)
            ax2.set_title('Monthly Investment Distribution')
            ax2.set_ylabel('Amount (₹)')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            return ""

    def chat_with_advisor(self, user_query: str) -> str:
        """Chat interface for investment advice"""
        try:
            response = self.agent.run(user_query)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}. Please try rephrasing your question."

def handle_sip_query(user_input: str, user_id: str, transactions: List[Dict]) -> str:
    """
    Dedicated handler for SIP-specific queries
    This function should be called when user specifically asks about SIPs
    """
    try:
        advisor = SmartInvestmentAdvisor()
        
        # Analyze user's financial profile
        analysis = advisor.analyze_transactions_for_investment(transactions)
        
        # Generate SIP-focused advice
        sip_advice = advisor._personalized_sip_advice()
        
        # Add SIP-specific recommendations based on savings
        monthly_savings = analysis.get('avg_monthly_savings', 0)
        
        if monthly_savings <= 0:
            return """
🚫 SIP Recommendation Alert:
You don't currently have positive monthly savings for SIP investments.

💡 First Steps:
1. Analyze your expenses and reduce unnecessary spending
2. Create a monthly budget with savings target
3. Build an emergency fund (6 months expenses)
4. Then start SIP with even ₹1,000/month

📞 Would you like help with expense analysis? Ask me to "analyze my spending" or "show spending categories"
"""
        
        # Determine SIP recommendations based on savings amount
        if monthly_savings < 5000:
            sip_recommendation = f"""
🎯 **Beginner SIP Strategy** (Based on ₹{monthly_savings:,.0f} monthly savings):

💪 **Recommended SIP Amount**: ₹{min(2000, monthly_savings//2):,.0f}/month

📈 **Top SIP Options for You**:
1. **Large Cap Fund SIP**: ₹{min(1000, monthly_savings//3):,.0f}/month
   - Low risk, steady returns (10-12% annually)
   - Perfect for beginners

2. **ELSS SIP**: ₹{min(1000, monthly_savings//3):,.0f}/month (if eligible for tax saving)
   - Tax benefits under Section 80C
   - 3-year lock-in period

💡 **Start Small Strategy**: Begin with ₹1,000/month and increase by ₹500 every 6 months!
"""
        
        elif monthly_savings < 15000:
            sip_recommendation = f"""
🎯 **Balanced SIP Strategy** (Based on ₹{monthly_savings:,.0f} monthly savings):

💪 **Recommended SIP Amount**: ₹{monthly_savings//2:,.0f}/month

📈 **Diversified SIP Portfolio**:
1. **Large Cap Fund**: ₹{(monthly_savings//2) * 0.4:,.0f}/month (40%)
   - Stable returns, lower volatility

2. **Mid Cap Fund**: ₹{(monthly_savings//2) * 0.3:,.0f}/month (30%)
   - Higher growth potential

3. **ELSS Fund**: ₹{(monthly_savings//2) * 0.3:,.0f}/month (30%)
   - Tax saving + equity exposure

🎯 **Expected Returns**: ₹{((monthly_savings//2) * 12 * 1.12):,.0f} in first year (at 12% return)
"""
        
        else:
            sip_recommendation = f"""
🎯 **Advanced SIP Strategy** (Based on ₹{monthly_savings:,.0f} monthly savings):

💪 **Recommended SIP Amount**: ₹{monthly_savings * 0.6:,.0f}/month

📈 **Aggressive Growth Portfolio**:
1. **Large Cap Fund**: ₹{(monthly_savings * 0.6) * 0.3:,.0f}/month (30%)
2. **Mid Cap Fund**: ₹{(monthly_savings * 0.6) * 0.25:,.0f}/month (25%)
3. **Small Cap Fund**: ₹{(monthly_savings * 0.6) * 0.15:,.0f}/month (15%)
4. **ELSS Fund**: ₹{(monthly_savings * 0.6) * 0.20:,.0f}/month (20%)
5. **International Fund**: ₹{(monthly_savings * 0.6) * 0.10:,.0f}/month (10%)

🚀 **Wealth Projection**: 
- 5 years: ₹{((monthly_savings * 0.6) * 12 * 5 * 1.15):,.0f}
- 10 years: ₹{((monthly_savings * 0.6) * 12 * 10 * 1.32):,.0f}
"""
        
        # Combine general advice with specific recommendations
        complete_response = f"""
{sip_advice}

{sip_recommendation}

🔥 **Action Steps**:
1. Choose 2-3 funds from different categories
2. Start SIP on 1st or 15th of each month
3. Review and rebalance every 6 months
4. Increase SIP amount annually by 10-15%

💡 **Pro Tip**: Set up auto-debit for consistent investing!

📞 **Need Help?**: Ask me "best mutual funds 2024" for current top performers!
"""
        
        return complete_response
        
    except Exception as e:
        return f"SIP advisory service is temporarily unavailable: {str(e)}"
    """
    Main handler function for investment queries
    This function should be called from complaint_handler.py
    """
    try:
        advisor = SmartInvestmentAdvisor()
        
        # Check if it's a general investment query or specific advice request
        investment_keywords = ['sip', 'investment', 'mutual fund', 'portfolio', 'invest', 'savings']
        
        if any(keyword in user_input.lower() for keyword in investment_keywords):
            if transactions:
                return advisor.get_investment_recommendations(transactions)
            else:
                return advisor.chat_with_advisor(user_input)
        else:
            return advisor.chat_with_advisor(user_input)
            
    except Exception as e:
        return f"Investment advisory service is temporarily unavailable: {str(e)}"

def generate_investment_summary(transactions: List[Dict]) -> str:
    """
    Generate investment summary for dashboard
    This function can be called from complaint_handler.py for dashboard queries
    """
    try:
        advisor = SmartInvestmentAdvisor()
        analysis = advisor.analyze_transactions_for_investment(transactions)
        
        summary = f"""
💰 Investment Summary:
- Monthly Savings Available: ₹{analysis['avg_monthly_savings']:,.0f}
- Total Savings This Year: ₹{analysis['total_savings']:,.0f}
- Current Investments: {len(analysis['investment_history'])} active

📈 Quick Recommendations:
- Start SIP with ₹{max(5000, analysis['avg_monthly_savings']//2):,.0f}/month
- Focus on diversified equity funds
- Maintain emergency fund
- Consider tax-saving investments (ELSS)

💡 Use 'investment advice' for detailed recommendations!
"""
        return summary
        
    except Exception as e:
        return f"Unable to generate investment summary: {str(e)}"

# For testing purposes
if __name__ == "__main__":
    # Test the investment advisor
    advisor = SmartInvestmentAdvisor()
    
    # Sample transaction data
    sample_transactions = [
        {"amount": "50000", "type": "received", "description": "Salary Credit", "created_at": "2024-01-15T10:00:00Z"},
        {"amount": "35000", "type": "expense", "description": "Monthly Expenses", "created_at": "2024-01-20T10:00:00Z"},
        {"amount": "5000", "type": "expense", "description": "SIP Investment", "created_at": "2024-01-25T10:00:00Z"}
    ]
    
    print("Testing Investment Advisor...")
    recommendations = advisor.get_investment_recommendations(sample_transactions)
    print(recommendations)