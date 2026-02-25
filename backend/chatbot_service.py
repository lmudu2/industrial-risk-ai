import os
import json
from groq import Groq
from sqlalchemy import text
from database import SessionLocal

import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Simple Prompt Engineering for Text-to-SQL
SYSTEM_PROMPT = """
You are an AI assistant for an Enterprise Asset Management (EAM) system.
Your job is to answer user questions by generating SQL queries for the SQLite database.

Database Schema:
- industries (id, name, description)
- assets (id, name, asset_type, location, status, manufacturer, purchase_cost, industry_id)
- sensors (id, asset_id, sensor_type, value, timestamp)
- maintenance_records (id, asset_id, maintenance_type, description, start_date, downtime_hours)
- work_orders (id, asset_id, title, description, status, priority, assigned_to, created_at)
- cost_records (id, work_order_id, labor_cost, parts_cost, total_cost)
- predictions (id, asset_id, risk_level, risk_score, confidence, predicted_failure, recommendation, predicted_cost, timestamp)

Enums:
- AssetStatus: operational, under_maintenance, failed, decommissioned
- MaintenanceType: preventive, corrective, predictive
- Priority: low, medium, high, critical
- PredictedFailure Values: 'Healthy', 'Preventive', 'General' (OK); 'Bearing', 'Cooling', 'Electrical', 'Seal' (Critical Failures)

Rules:
1. Return ONLY the SQL query. No markdown, no explanation.
2. Use standard SQLite syntax.
3. If the user asks a general question not about data (e.g., "What is predictive maintenance?"), answer politely in text (prefix with "TEXT:").
4. Today is {today}.
5. When asked for "failing assets", query `predictions` table where `predicted_failure` is NOT 'Healthy'.
6. STRATEGY: 
   - Check "CURRENT DASHBOARD DATA CONTEXT" (the visible stats on screen).
   - If the answer is purely mathematical or summary-based using ONLY those keys, use it (prefix with "TEXT:").
   - If the question mentions specific entities (Wind Turbines, Robotic Arms) or patterns (vibration anomalies) NOT fully detailed in the context BUT present in the schema/metadata, you MUST write a SQL query.
7. SCHEMA JOIN HINT: 
   - To filter by Industry Name, JOIN `assets` with `industries`.
   - To find "vibration failures" in logs, query `work_orders` where `description` LIKE '%vibration%' or `title` LIKE '%vibration%'.
   - "Emergency Interventions" are `work_orders` where `priority` = 'critical' or 'high'.
   - To find AI predicted patterns, query the `predictions` table.
8. EVERY TEXTUAL ANSWER (TEXT:) MUST follow the JSON format:
   {{
     "reasoning": "Explain WHY you used context vs SQL",
     "answer": "The concise answer",
     "citations": "Specific metadata or context keys used",
     "sql_logic": "N/A"
   }}
9. FINANCIAL DATA: Always format currency values professionally. Use '$M' for millions (e.g., $61.89M) and '$K' for thousands. Round all financial figures to 2 decimal places. 
10. DO NOT HALLUCINATE. If a SQL query is required but you can't form it, ask for clarification.
"""

def query_database(sql_query):
    db = SessionLocal()
    try:
        result = db.execute(text(sql_query)).fetchall()
        return result
    except Exception as e:
        raise e
    finally:
        db.close()

def format_ai_response(content):
    """
    Parses the raw JSON or 'TEXT:' string from the LLM and returns polished Markdown.
    Includes a 'fuzzy' JSON extractor to handle sloppy LLM output.
    """
    # Strip possible LLM prefix
    clean_content = content.replace("TEXT:", "").strip()
    
    # 1. Try to find a JSON block within the response
    import re
    json_match = re.search(r'(\{.*\})', clean_content, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = clean_content

    try:
        # Pre-process JSON: Handle literal newlines that LLMs sometimes inject into strings
        # This is a common source of json.loads failure
        sanitized_json = json_str.replace('\n', ' ').replace('\r', '')
        # Basic attempt to fix unescaped double quotes inside strings (common in SQL)
        # Note: This is heuristic, but helps with 90% of failures
        
        parsed = json.loads(sanitized_json)
        
        answer = parsed.get("answer", clean_content)
        reasoning = parsed.get("reasoning", "")
        citations = parsed.get("citations", "")
        
        # Professional Markdown output
        formatted = f"{answer}"
        
        if reasoning or citations:
            formatted += "\n\n---\n"
            if reasoning:
                formatted += f"**🧠 AI Reasoning:** {reasoning}\n"
            if citations:
                formatted += f"**📍 Sources:** {citations}"
                
        return formatted
        
    except Exception as e:
        # Final fallback: if json.loads fails, use regex to extract values directly
        # This prevents the "comma-separated list" weirdness
        print(f"Standard JSON parsing failed: {e}. Attempting regex extraction...")
        
        def extract_field(field_name, text):
            # Look for "field_name": "value" even with unescaped newlines/quotes
            pattern = rf'"{field_name}"\s*:\s*"(.*?)"(?=\s*[,}}])'
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else ""

        answer = extract_field("answer", clean_content)
        reasoning = extract_field("reasoning", clean_content)
        citations = extract_field("citations", clean_content)
        
        if not answer:
            # Absolute last resort: just scrub the keys and return the blob
            text_fallback = clean_content
            for key in ["\"answer\":", "\"reasoning\":", "\"citations\":", "\"sql_logic\":"]:
                text_fallback = text_fallback.replace(key, "")
            text_fallback = text_fallback.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
            answer = text_fallback.strip()

        # Format with the same structure as the successful case
        formatted = f"{answer}"
        if reasoning or citations:
            formatted += "\n\n---\n"
            if reasoning:
                formatted += f"**🧠 AI Reasoning:** {reasoning}\n"
            if citations:
                formatted += f"**📍 Sources:** {citations}"

        # PRIVACY SCRUBBING: Remove any line that looks like raw SQL
        clean_lines = []
        for line in formatted.split('\n'):
            lower_line = line.lower()
            if "select " in lower_line and "from " in lower_line:
                continue
            if "sql_logic" in lower_line:
                continue
            clean_lines.append(line)
        
        return "\n".join(clean_lines).strip()

def generate_response(user_query, context=None):
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return "⚠️ Groq API Key not found."

    client = Groq(api_key=api_key)

    try:
        # Dynamic Prompt with Today's Date and optional Dashboard Context
        current_date = datetime.date.today()
        final_system_prompt = SYSTEM_PROMPT.format(today=current_date)
        
        if context:
            final_system_prompt += f"\n\nCURRENT DASHBOARD DATA CONTEXT (Use this to answer questions about active issues):\n{context}"

        # Step 1: Get SQL from LLM
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("TEXT:"):
            return format_ai_response(content)
        
        # Clean SQL (remove markdown code blocks if presnet)
        sql_query = content.replace("```sql", "").replace("```", "").strip()
        
        # Step 2: SQL Execution (If LLM attempted one)
        print(f"Executing SQL: {sql_query}")
        try:
            data = query_database(sql_query)
            if not data:
                return "Based on the data, I found no results matching your query."
        except Exception as e:
            # New Fallback logic to intercept explicitly bad SQL commands that couldn't execute
            print(f"SQL Error intercepted: {e}")
            data = f"Error: Could not execute database command. Fallback request: The SQL query you attempted failed. Answer the user using ONLY the math and figures natively available in the DASHBOARD DATA CONTEXT instead. Apologize briefly."

        # Step 3: Summarize results
        # Clean the SQL query for injection into the prompt's JSON-like structure
        escaped_sql = sql_query.replace('"', "'").replace("\n", " ")
        
        summary_prompt = f"""
        User Query: {user_query}
        SQL Query Attempted: {escaped_sql}
        Data Results: {data}
        
        Rules:
        1. Answer the user query using ONLY the results and context provided. 
        2. MANDATORY SUMMARY FORMAT (Strict JSON):
           {{
             "reasoning": "Brief explanation of derivation (Do NOT mention SQL tables/syntax here)",
             "answer": "Final natural language answer",
             "citations": "Specific sources used",
             "sql_logic": "{escaped_sql if sql_query != 'TEXT:' else 'N/A'}"
           }}
        3. JSON SAFETY: Escape ALL special characters. Do NOT include raw newlines within JSON values. Use \\n for breaks.
        4. PRIVACY & SECURITY: NEVER include SQL queries, table names, or database syntax in the "answer" or "reasoning" fields. Database operations must remain completely invisible to the user.
        5. WORK ORDERS:
           - "Active Work Orders": Use {context.get('total_active_work_orders_count', 0)}
           - "Total Work Orders": Use {context.get('total_fleet_work_orders_count', 0)}
        4. No Hallucinations: If results are empty, state that in the "answer" field.
        5. PRIVACY: Do NOT include raw SQL or code in the "reasoning" or "answer" fields.
6. FORMATTING: Format the "answer" naturally. Convert large currency figures (millions) to '$M' format and round to 2 decimal places.
        """
        
        final_response = client.chat.completions.create(
            messages=[{"role": "user", "content": summary_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0
        )
        return format_ai_response(final_response.choices[0].message.content.strip())

    except Exception as e:
        return f"🤖 Error processing query: {str(e)}"

# ─────────────────────────────────────────────
# NEW: STRATEGIC ADVICE
# ─────────────────────────────────────────────

def get_maintenance_recommendation(data):
    """
    Get strategic advice (Repair vs Replace) from LLM based on simulation data.
    """
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return "⚠️ API Key missing."

    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are a Senior Reliability Engineer. Analyze this maintenance scenario for an industrial asset and provide an economically sound recommendation.
    
    Asset Context:
    - Type: {data.get('asset_type', 'Industrial Machinery')}
    - Current Age: {data.get('age_years', 0)} years
    - Estimated Repair Cost: ${data.get('predicted_cost', 0):,.2f}
    - Repair Urgency: {data.get('priority', 'Medium')}
    
    Exact Economic Calculations:
    - Straight-Line Estimated Residual Value: ${data.get('residual_value', 1000):,.2f}
    - Repair-to-Replace Ratio: {data.get('repair_ratio', 0):.1f}%
    
    Economic Rules for Recommendation:
    1. WARRANTY OVERRIDE: If the asset is under warranty (`{data.get('under_warranty', False)}`), ALWAYS recommend "Repair". Acknowledge that while the warranty covers major parts (reducing the cost by ~85%), the remaining ${data.get('predicted_cost', 0):,.2f} accounts for uncovered labor, consumables, or operational downtime. This overrides the 50% rule.
    2. THE 50% RULE: 
       - If Repair Ratio < 50%: Recommend "Repair".
       - If Repair Ratio >= 50%: Recommend "Replace" or "Monitor".
    3. URGENCY FACTOR: If Urgency is 'Critical' or 'High', prioritize "Repair" unless the asset is clearly beyond economical salvage (>80% ratio), to minimize downtime.
    
    Output Format:
    If NOT under warranty: "Recommend [Action]. Following the 50% rule, the repair cost is {data.get('repair_ratio', 0):.1f}% of the exact ${data.get('residual_value', 1000):,.2f} estimated residual value, making it [economically viable / uneconomical]."
    If UNDER warranty: "Recommend Repair. The asset is under active warranty. While major components are covered, the remaining ${data.get('predicted_cost', 0):,.2f} reflects estimated diagnostic labor, logistical downtime, or necessary consumables overhead."
    
    Keep it strictly under 50 words. No Markdown. Start immediately with the recommendation.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1, # Lowest temperature for logical consistency
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Advice unavailable."

def evaluate_dispatch_action(asset_name, risk_level, score, rul, cost):
    """Uses Groq to dynamically evaluate the best technician dispatch tier."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fallback to hardcoded heuristic if no API key
        if risk_level == 'Critical' and rul <= 7:
            return "Dispatch MASTER Technician (Immediate response required to prevent catastrophic failure in < 7 days)."
        elif cost > 50000:
            return "Dispatch SENIOR Technician (High liability repair requires experienced oversight)."
        return "Dispatch JUNIOR/INTERMEDIATE Technician (Standard repair, optimize for lower hourly rate)."
    
    client = Groq(api_key=api_key)
    
    try:
        prompt = f"""
        You are an AI Diagnostics Engine routing maintenance tickets.
        Given the following asset state, recommend a technician tier (MASTER, SENIOR, or JUNIOR/INTERMEDIATE) and provide a concise, one-sentence reasoning.
        
        Rules:
        - Critical risk with < 7 days RUL requires a MASTER.
        - High financial liability (>$50k) requires a SENIOR.
        - Standard repairs require JUNIOR/INTERMEDIATE to optimize cost.
        
        Asset: {asset_name}
        Risk: {risk_level} (Probability: {score}%)
        Estimated RUL: {rul} days
        Financial Liability: ${cost:,.0f}
        
        Output format exact: "Dispatch [TIER] Technician ([Brief Reasoning])."
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Updated from deprecated llama3-8b-8192
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=60
        )
        return response.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"Groq dynamic dispatch failed: {e}")
        return "Dispatch SENIOR Technician (Default routing due to AI service timeout)."

def get_signal_insights(signals_data, asset_metadata):
    """
    Get dynamic, technical explanations for sensor deviations from LLM.
    """
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return "⚠️ API Key missing."

    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are a Senior Reliability Engineer. Analyze these sensor signal deviations:
    
    Context:
    - Asset: {asset_metadata.get('asset_type', 'Unknown')}
    - Predicted Failure: {asset_metadata.get('predicted_failure', 'Unknown')}
    - Current Data: {signals_data}
    
    Mandatory Task:
    Explain how the current sensor data (use the numbers) specifically confirms the failure of the "{asset_metadata.get('predicted_failure', 'Unknown')}" component.
    
    STRICT RULES:
    1. The VERY FIRST SENTENCE MUST state: "The predicted failure of the {asset_metadata.get('predicted_failure', 'Unknown')} is confirmed by [specific numbers]..."
    2. Do NOT mention secondary consequences (like turbine blades) until later.
    3. Keep the total response under 45 words.
    4. Start immediately with the explanation.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=60
        )
        return response.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"Signal Insight Error: {e}")
        return "Technical insight currently unavailable."
        
def get_sensor_summary(sensors_data, asset_type):
    """
    Get a plain-English summary of what the raw sensor data means for the operator.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ API Key missing. Setup required for AI analysis."

    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an AI assistant helping a factory operator understand machine data at a glance.
    Look at the current sensor averages for this {asset_type}:
    {sensors_data}

    Briefly explain what these numbers mean in 2 short, simple sentences. Don't just list the numbers back.
    Instead, explain if the machine is running hot, vibrating normally, etc.
    Keep it extremely concise and easy to read.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=60
        )
        return response.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"Sensor Summary Error: {e}")
        return "Sensor telemetry is currently processing."

def get_executive_briefing(telemetry_data):
    """
    Generate an AI-driven "Morning Briefing" based strictly on injected data. 
    Prevents hallucination by forbidding the LLM from inventing facts outside the payload.
    """
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return "⚠️ Setup Incomplete: Missing GROQ_API_KEY environment variable. The AI Command Center requires an active Groq API key."

    client = Groq(api_key=api_key)
    
    alert_label = telemetry_data.get('alert_type_name', 'Critical AI Alerts')
    spend = telemetry_data.get('total_maintenance_spend', 0)
    liability = telemetry_data.get('predicted_liability', 0)
    
    # Smart formatting for the prompt
    spend_fmt = f"${spend/1000000:.2f}M" if spend >= 100000 else f"${spend/1000:.1f}K"
    liability_fmt = f"${liability/1000000:.2f}M" if liability >= 100000 else f"${liability/1000:.1f}K"

    prompt = f"""
    You are an AI Executive Assistant for the Enterprise Asset Management (EAM) platform.
    Your task is to write a punchy, 2-3 sentence "Morning Briefing" for the plant manager.
    
    CRITICAL INSTRUCTION: You MUST strictly use ONLY the data provided below. Do NOT invent, guess, or hallucinate any numbers, names, or statuses.
    
    DATA PAYLOAD:
    - Total Assets Monitored: {telemetry_data.get('total_assets', 0)}
    - Active Maintenance Tickets (Work Orders): {telemetry_data.get('active_work_orders', 0)}
    - Total Maintenance Spend: {spend_fmt}
    - {alert_label}: {telemetry_data.get('critical_alerts', 0)}
    - Total Predicted Liability: {liability_fmt}
    
    Top Imminent Failures (if any):
    {telemetry_data.get('top_failures', 'None detected.')}
    
    Formatting Rules:
    - Keep it under 50 words.
    - Be professional but urgent if there are high alert counts.
    - If there are alerts, explicitly mention the liability.
    - If there are NO alerts, state that the system is nominal.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0, # Zero temperature is critical for avoiding hallucinations with data injection
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Briefing temporarily unavailable. System Error: {str(e)}"
