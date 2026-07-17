# Agentic AI Policy Compliance Agent

An autonomous GRC (Governance, Risk, and Compliance) agent that continuously audits system configurations against written organizational policies, calculates compliance scores, provides XAI (Explainable AI) feature importance, and spins up an interactive monitoring dashboard.

##  Features

- **Policy Ingestion**: Parses raw text security policies into structured compliance targets using an LLM.
- **Config Audit Loop**: Automatically maps infrastructure states against rule matrices.
- **Explainable AI (XAI)**: Generates feature-level risk metrics using surrogate SHAP/LIME-style scoring.
- **Streamlit Dashboard**: A high-utility interface displaying compliance health, active violations, and automated remediation scripts.

## Architecture Overview

1. **Ingestion Engine**: Text Cleaning -> NLP Extraction -> JSON Ruleset Mapping.
2. **Agentic Reasoning Core**: Iterative evaluation of system settings against extracted rulesets.
3. **Analytics & XAI Suite**: Computes risk weight vectors and feature impact.
4. **Presentation Layer**: Streamlit web application with interactive metrics.

##  Quick Start

### 1. Prerequisites

- Python 3.10+
- OpenAI API Key (or a local Ollama instance running `llama3`)

### 2. Installation

```bash
pip install openai streamlit pandas numpy scikit-learn matplotlib
```

### 3. Execution

Save the monolithic pipeline to `app.py` and run:

```bash
streamlit run app.py
```
