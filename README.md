# Asset Risk Analysis for Industrial Machinery 🏭🛰️📡

An Enterprise Asset Management (EAM) Predictive Maintenance Platform that leverages Machine Learning and Generative AI to monitor, analyze, and mitigate industrial equipment risks.

## 🚀 Overview
This platform provides a comprehensive suite for cross-industry asset management, focusing on:
- **Predictive Analytics**: Machine Learning models that categorize asset health (Healthy, Warning, High Risk, Critical) based on live sensor telemetry.
- **Financial Intelligence**: Exponential regression models to estimate repair costs, downtime liability, and "Repair vs. Replace" ratios.
- **AI-Driven Reasoning**: Integration with Llama-3 (via Groq) to provide natural language insights, strategic maintenance advice, and an interactive fleet assistant.
- **Autonomous Dispatch**: Tiered automation that triggers alerts and work orders instantly for high-confidence critical failures.

## 🌐 Deployment Options
- **Streamlit Cloud (Live)**: [Industrial Risk AI Dashboard](https://chatbotservicepy-nuclcdfs5huczcuadti3as.streamlit.app)
- **Google Colab (Interactive Demo)**: Open the provided [Predictive_Maintenance_Platform.ipynb](https://colab.research.google.com/github/lmudu2/industrial-risk-ai/blob/main/Predictive_Maintenance_Platform.ipynb) directly in Google Colab to run the full dashboard & backend API in the cloud without installing anything locally.

## 🏗️ Architecture
- **Frontend**: Streamlit-based AI-first dashboard with glassmorphism aesthetics and interactive visualizations.
- **Backend**: FastAPI powered by SQLAlchemy (SQLite/PostgreSQL) and Pydantic for robust data validation.
- **ML Engine**: Scikit-learn (Random Forest & Regression) for real-time risk classification and cost estimation.
- **AI Assistant**: Groq LLM integration with specialized "Fuzzy JSON" recovery logic for rock-solid conversational reporting.

## 🛠️ Key Features
- **Fleet-Wide Risk Map**: Real-time drill-down into specific machines (Robotic Arms, Wind Turbines, CNC Machines, etc.).
- **Smart Action Center**: Prioritized incident queue with recommended technician dispatch tiers (Master, Senior, Junior).
- **Cost Predictor Simulator**: A "what-if" tool for finance-aligned maintenance strategy.
- **Resilient Chatbot**: An enterprise-grade assistant with SQL privacy hardening and deep semantic awareness of dashboard data.

## 🏁 Getting Started
1. **Clone the repository**:
   ```bash
   git clone <remote-url>
   cd asset-risk-analysis-industrial-machinery
   ```
2. **Setup environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure API Keys**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_key_here
   SENDGRID_API_KEY=your_key_here
   ```
4. **Launch Platform**:
   ```bash
   chmod +x start_app.sh
   ./start_app.sh
   ```

## 🧠 Core Engineering Principles
This POC was built with a focus on **Graceful Degradation**, **State Synchronization**, and **Real-World Business Nuance** (e.g., fractional warranty liability modeling).

---
*Developed as a Strategic AI Proof of Concept for Enterprise Asset Management.*
