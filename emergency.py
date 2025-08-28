# emergency_detector.py
"""
Emergency Detection Module for Financial Assistant
Detects emergency situations and triggers voice calls
"""

import re
import time
from typing import Dict, List
from supabase import create_client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Supabase client
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Emergency detector: Supabase client initialized")
except Exception as e:
    print(f"❌ Emergency detector: Failed to initialize Supabase: {e}")
    supabase = None

# Emergency keywords and patterns
EMERGENCY_KEYWORDS = {
    'lost_card': [
        'lost my card', 'lost credit card', 'lost debit card', 'card is lost',
        'cannot find my card', 'misplaced my card', 'card missing', 'lost my atm card'
    ],
    'stolen_card': [
        'card stolen', 'someone stole my card', 'card theft', 'stolen credit card',
        'stolen debit card', 'my card was stolen', 'card got stolen'
    ],
    'fraud': [
        'fraud', 'fraudulent transaction', 'unauthorized transaction', 
        'someone used my card', 'fake transaction', 'scam', 'cheated',
        'unauthorized payment', 'money deducted without permission'
    ],
    'account_locked': [
        'account locked', 'cannot access account', 'blocked account',
        'account suspended', 'login blocked', 'account frozen'
    ],
    'suspicious_activity': [
        'suspicious transaction', 'weird transaction', 'unknown transaction',
        'strange payment', 'unfamiliar charge', 'unrecognized transaction'
    ],
    'phishing_attack': [
        'phishing', 'fake email', 'suspicious email', 'fake sms',
        'someone asking for otp', 'suspicious call', 'fake bank call'
    ]
}

def detect_emergency_situation(user_input: str) -> Dict:
    """
    Detect if user input contains emergency situations
    Returns emergency type and confidence level
    """
    user_input_lower = user_input.lower().strip()
    
    # Direct keyword matching
    for emergency_type, keywords in EMERGENCY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in user_input_lower:
                return {
                    'is_emergency': True,
                    'emergency_type': emergency_type,
                    'matched_keyword': keyword,
                    'confidence': 'high',
                    'severity': get_emergency_severity(emergency_type)
                }
    
    # Check for general emergency indicators with financial context
    emergency_indicators = ['emergency', 'urgent', 'help', 'immediately', 'asap', 'quick']
    financial_context = ['card', 'credit', 'debit', 'payment', 'transaction', 'account', 'money', 'bank']
    
    has_emergency = any(indicator in user_input_lower for indicator in emergency_indicators)
    has_financial_context = any(context in user_input_lower for context in financial_context)
    
    if has_emergency and has_financial_context:
        return {
            'is_emergency': True,
            'emergency_type': 'general_financial_emergency',
            'matched_keyword': 'general emergency with financial context',
            'confidence': 'medium',
            'severity': 'medium'
        }
    
    return {
        'is_emergency': False,
        'emergency_type': None,
        'matched_keyword': None,
        'confidence': 'none',
        'severity': 'none'
    }

def get_emergency_severity(emergency_type: str) -> str:
    """
    Determine the severity level of the emergency
    """
    severity_mapping = {
        'fraud': 'critical',
        'stolen_card': 'critical',
        'phishing_attack': 'critical',
        'lost_card': 'high',
        'suspicious_activity': 'high',
        'account_locked': 'medium',
        'general_financial_emergency': 'medium'
    }
    return severity_mapping.get(emergency_type, 'low')

def trigger_emergency_voice_call(user_id: str, emergency_type: str, description: str, severity: str = 'medium') -> Dict:
    """
    Trigger the emergency voice call by inserting into voice_call_requests table
    """
    if not supabase:
        return {
            'success': False,
            'error': 'Supabase client not initialized',
            'message': 'Database connection unavailable'
        }
    
    try:
        # Generate unique room name with timestamp
        timestamp = int(time.time() * 1000)
        room_name = f'emergency-{emergency_type}-{user_id}-{timestamp}'
        
        # Default agent ID - replace with your actual emergency agent ID
        agent_id = 'e274b907-7b9b-4046-9e0a-9fde747c9e9e'
        
        # Prepare call data
        call_data = {
            "user_id": user_id,
            "agent_id": agent_id,
            "room_name": room_name,
            "status": "pending",
            "call_duration": 0,
            "priority": severity,  # Add priority based on severity
            "emergency_type": emergency_type,
            "description": description[:500]  # Limit description length
        }
        
        # Insert into voice_call_requests table
        result = supabase.table("voice_call_requests").insert(call_data).execute()
        
        if result.data:
            print(f"✅ Emergency voice call created: {result.data[0]['id']}")
            return {
                'success': True,
                'call_id': result.data[0]['id'],
                'room_name': room_name,
                'agent_id': agent_id,
                'severity': severity,
                'message': 'Emergency voice call request created successfully'
            }
        else:
            return {
                'success': False,
                'error': 'No data returned from database',
                'message': 'Failed to create voice call request'
            }
        
    except Exception as e:
        print(f"❌ Error creating emergency voice call: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to create emergency voice call request'
        }

def get_emergency_instructions(emergency_type: str) -> str:
    """
    Get specific instructions based on emergency type
    """
    instructions = {
        'lost_card': """
**Immediate Actions:**
1. 🚫 Block your card immediately by calling your bank's 24/7 helpline
2. 📱 Use your mobile banking app to block the card if available
3. 👀 Monitor your account for any unauthorized transactions
4. 📝 Note down the last known location where you had your card

**Emergency Helplines:**
- SBI: 1800 1234 (Toll-free)
- HDFC: 1800 2600 (Toll-free)
- ICICI: 1860 120 7777
""",

        'stolen_card': """
**Immediate Actions:**
1. 🚫 Block your card IMMEDIATELY - every second counts
2. 🚨 File a police complaint if theft occurred with other items
3. 👀 Check your account for unauthorized transactions
4. 📝 Note the time and location of theft
5. 🛡️ Change your ATM PIN and online banking passwords

**Critical:** Do NOT delay blocking your card even by a few minutes!
""",

        'fraud': """
**Immediate Actions:**
1. 🛑 Do NOT share any OTP, PIN, or passwords with anyone
2. 📱 Block your card immediately
3. 📸 Take screenshots of fraudulent transactions
4. 📞 Call your bank's fraud helpline immediately
5. 📋 File a complaint with your bank's customer care

**Remember:** Banks will NEVER ask for your PIN or OTP over phone/SMS!
""",

        'account_locked': """
**Immediate Actions:**
1. 📞 Call your bank's customer service helpline
2. 🆔 Keep your account details and ID proof ready
3. 🔐 Try resetting your password through official channels
4. 🏦 Visit your nearest branch if phone support doesn't help

**Avoid:** Multiple login attempts as it may extend the lock period
""",

        'suspicious_activity': """
**Immediate Actions:**
1. 📱 Check your account statement immediately
2. 📝 Note down all suspicious transactions with details
3. 📞 Report to your bank's fraud department
4. 🛡️ Change all your banking passwords and PINs
5. 📧 Check for any suspicious emails or SMS

**Stay Alert:** Monitor your account daily for unusual activity
""",

        'phishing_attack': """
**Immediate Actions:**
1. 🚫 Do NOT click any suspicious links or attachments
2. 🔐 Change all your banking passwords immediately
3. 📱 Check your accounts for unauthorized access
4. 📧 Report the phishing attempt to your bank
5. 🗑️ Delete the suspicious email/SMS

**Remember:** Banks never ask for credentials via email/SMS!
"""
    }
    
    return instructions.get(emergency_type, """
**General Emergency Actions:**
1. 📞 Contact your bank immediately
2. 🛡️ Secure your accounts and change passwords
3. 📝 Document all relevant details
4. 👀 Monitor your accounts closely
""")

def handle_emergency_query(user_input: str, user_id: str) -> str:
    """
    Main function to handle emergency situations and trigger voice call
    """
    # Detect emergency situation
    emergency_info = detect_emergency_situation(user_input)
    
    if not emergency_info['is_emergency']:
        return None  # Not an emergency, continue with normal processing
    
    print(f"🚨 EMERGENCY DETECTED: {emergency_info['emergency_type']} (Severity: {emergency_info['severity']})")
    
    # Trigger emergency voice call
    call_result = trigger_emergency_voice_call(
        user_id=user_id,
        emergency_type=emergency_info['emergency_type'],
        description=user_input,
        severity=emergency_info['severity']
    )
    
    # Build response based on success/failure
    if call_result['success']:
        response = f"""🚨 **EMERGENCY DETECTED** 🚨

⚠️ **Emergency Type:** {emergency_info['emergency_type'].replace('_', ' ').title()}
🔴 **Severity Level:** {emergency_info['severity'].upper()}

✅ **Emergency Voice Call Initiated Successfully!**
📞 **Call ID:** {call_result.get('call_id', 'Pending')}
🏠 **Room:** {call_result.get('room_name', 'Assigned')}
⏱️ **Expected Call Time:** 2-3 minutes

{get_emergency_instructions(emergency_info['emergency_type'])}

🔒 **Your financial security is our top priority. Our emergency response team will contact you shortly!**

---
💡 **While you wait:**
- Keep your phone nearby and ready to answer
- Have your account details and ID ready
- Stay calm - we're here to help you resolve this quickly
"""
    else:
        response = f"""🚨 **EMERGENCY SITUATION DETECTED** 🚨

⚠️ **Emergency Type:** {emergency_info['emergency_type'].replace('_', ' ').title()}

❌ **Voice Call Issue:** {call_result.get('message', 'Unknown error occurred')}

{get_emergency_instructions(emergency_info['emergency_type'])}

**🆘 Alternative Actions:**
1. Call our emergency helpline: **1800-XXX-XXXX**
2. Try submitting your request again
3. Contact your bank directly using the numbers above

**We apologize for the technical difficulty during this emergency situation.**
"""
    
    return response

def test_emergency_detection() -> None:
    """
    Test function to verify emergency detection system
    """
    test_cases = [
        ("I lost my credit card", True),
        ("My debit card was stolen", True),
        ("There's a fraudulent transaction on my account", True),
        ("Someone used my card without permission", True),
        ("My account is locked and I need help urgently", True),
        ("I got a suspicious email asking for my PIN", True),
        ("What's my current balance?", False),
        ("Show me my transaction history", False),
        ("I need urgent help with my card", True),
        ("How do I transfer money?", False)
    ]
    
    print("🧪 Testing Emergency Detection System:")
    print("=" * 50)
    
    for test_input, expected_emergency in test_cases:
        result = detect_emergency_situation(test_input)
        is_emergency = result['is_emergency']
        status = "✅ PASS" if is_emergency == expected_emergency else "❌ FAIL"
        emergency_indicator = "🚨 EMERGENCY" if is_emergency else "✅ Normal"
        
        print(f"{status} | {emergency_indicator}")
        print(f"Input: '{test_input}'")
        if is_emergency:
            print(f"Type: {result['emergency_type']} | Severity: {result.get('severity', 'N/A')}")
        print("-" * 30)

# Helper function for integration
def is_emergency_query(user_input: str) -> bool:
    """
    Quick check if a query is emergency-related
    """
    return detect_emergency_situation(user_input)['is_emergency']

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_emergency_detection()