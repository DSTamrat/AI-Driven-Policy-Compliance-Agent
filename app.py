import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import os

# Page layout configurations
st.set_page_config(page_title="AI Policy Compliance Agent", page_icon="", layout="wide")

# Default values for smooth fallback
DEFAULT_POLICY = """COMPANY SECURITY POLICY RULES v4.2:
1. All internet-facing firewalls must completely block port 22 (SSH) and port 3389 (RDP) from public access.
2. Production data storage buckets must have encryption-at-rest enabled at all times.
3. Multi-Factor Authentication (MFA) must be strictly enforced for all system administrative users.
4. Automated system security patches must be executed at minimum every 7 days."""

DEFAULT_CONFIG = {
    "server_name": "prod-core-db-01",
    "environment": "Production",
    "firewall_rules": {"allow_public_port_80": True, "allow_public_port_443": True, "allow_public_port_22": True, "allow_public_port_3389": False},
    "storage_settings": {"s3_bucket_encryption": False, "backup_retention_days": 30},
    "identity_access": {"mfa_enforced_admins": True, "password_rotation_days": 90},
    "patching_lifecycle": {"auto_patch_interval_days": 14}
}

# -----------------------------------------------------------------------------
# DETMINISTIC AGENT ENGINE (Offline Rule Matching Pipeline)
# -----------------------------------------------------------------------------
class RuleEngineAgent:
    """
    Parses unstructured text files locally and dynamically executes rulesets 
    against user-provided configuration objects.
    """
    @staticmethod
    def evaluate(policy_text: str, config_dict: dict) -> pd.DataFrame:
        results = []
        
        # Rule 1: Firewall Check
        firewall = config_dict.get("firewall_rules", {})
        p22 = firewall.get("allow_public_port_22", False)
        p3389 = firewall.get("allow_public_port_3389", False)
        if "22" in policy_text or "3389" in policy_text:
            status = "Violated" if (p22 or p3389) else "Compliant"
            val_text = f"Port 22 Open: {p22}, Port 3389 Open: {p3389}"
            results.append({"Rule ID": "R1", "Component": "Firewall", "Description": "Block public port 22/3389", "Status": status, "Value": val_text, "Weight": 0.4})
            
        # Rule 2: Encryption Check
        storage = config_dict.get("storage_settings", {})
        enc = storage.get("s3_bucket_encryption", True)
        if "encryption" in policy_text.lower():
            status = "Compliant" if enc else "Violated"
            val_text = "Enabled" if enc else "Disabled"
            results.append({"Rule ID": "R2", "Component": "Storage", "Description": "Enable storage encryption", "Status": status, "Value": val_text, "Weight": 0.3})
            
        # Rule 3: IAM Check
        iam = config_dict.get("identity_access", {})
        mfa = iam.get("mfa_enforced_admins", False)
        if "mfa" in policy_text.lower() or "multi-factor" in policy_text.lower():
            status = "Compliant" if mfa else "Violated"
            val_text = "Enforced" if mfa else "Missing"
            results.append({"Rule ID": "R3", "Component": "IAM", "Description": "Enforce Admin MFA", "Status": status, "Value": val_text, "Weight": 0.2})
            
        # Rule 4: Patching Lifecycle Check
        patch = config_dict.get("patching_lifecycle", {})
        days = patch.get("auto_patch_interval_days", 0)
        if "patch" in policy_text.lower():
            status = "Violated" if days > 7 else "Compliant"
            val_text = f"{days} Days"
            results.append({"Rule ID": "R4", "Component": "Patching", "Description": "Patch Cycle <= 7 Days", "Status": status, "Value": val_text, "Weight": 0.1})
            
        return pd.DataFrame(results) if results else pd.DataFrame(columns=["Rule ID", "Component", "Description", "Status", "Value", "Weight"])

class ComplianceXAI:
    @staticmethod
    def calculate_shap_contributions(df: pd.DataFrame):
        if df.empty:
            return df
        X = np.array([[1 if s == "Compliant" else 0 for s in df["Status"]]])
        base_score = 100
        deductions = sum([row["Weight"] * 100 for idx, row in df.iterrows() if row["Status"] == "Violated"])
        y = np.array([base_score - deductions])
        
        X_perturbed = np.random.binomial(1, 0.7, size=(50, len(df)))
        weights = df["Weight"].values * 100
        y_perturbed = base_score - np.dot(X_perturbed, weights)
        
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(X_perturbed, y_perturbed)
        
        coefs = np.atleast_1d(surrogate.coef_).flatten()
        df["Risk Contribution Score"] = np.round(coefs, 2) * -1
        return df

# -----------------------------------------------------------------------------
# GRAPHICAL DISPLAY PRESENTATION LAYER
# -----------------------------------------------------------------------------
st.title(" Enterprise AI Policy Compliance Agent")
st.markdown("Automate governance auditing by uploading production text parameters and live configuration metrics.")
st.markdown("---")

# Sidebar Configuration Control Center for Uploads
st.sidebar.header(" Ingestion Control Center")
uploaded_policy = st.sidebar.file_uploader("Upload Corporate Policy (.txt)", type=["txt"])
uploaded_config = st.sidebar.file_uploader("Upload System Configuration (.json)", type=["json"])

# Parse Ingested Inputs
policy_content = uploaded_policy.read().decode("utf-8") if uploaded_policy else DEFAULT_POLICY
if uploaded_config:
    try:
        config_content = json.load(uploaded_config)
    except:
        st.sidebar.error("Invalid JSON configuration format detected.")
        config_content = DEFAULT_CONFIG
else:
    config_content = DEFAULT_CONFIG

# Execute Audit Engine Pipeline
audit_df = RuleEngineAgent.evaluate(policy_content, config_content)
processed_df = ComplianceXAI.calculate_shap_contributions(audit_df)

# Global Performance Metrics
total_rules = len(processed_df)
violations_count = len(processed_df[processed_df["Status"] == "Violated"]) if not processed_df.empty else 0
compliance_score = int(100 - processed_df.loc[processed_df["Status"] == "Violated", "Weight"].sum() * 100) if not processed_df.empty else 100
target_asset = config_content.get("server_name", "Unknown Node")

# Row 1: KPI Score Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Global Compliance Score", value=f"{compliance_score}%", delta=f"{compliance_score - 100}% From Max")
col2.metric(label="Total Rules Extracted", value=total_rules)
col3.metric(label="Active System Violations", value=violations_count, delta="Action Required" if violations_count > 0 else "System Clear", delta_color="inverse" if violations_count > 0 else "normal")
col4.metric(label="Target System Asset", value=target_asset)

# Dynamic Table View
st.markdown("###  Live Configuration Audit Matrix")
if not processed_df.empty:
    st.dataframe(processed_df[["Rule ID", "Component", "Description", "Status", "Value"]], use_container_width=True)
else:
    st.info("No matching rules could be verified between the policy text and system config file columns.")

st.markdown("---")
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("###  Explainable AI (XAI) Impact Analysis")
    if not processed_df.empty and violations_count > 0:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        violation_df = processed_df[processed_df["Status"] == "Violated"]
        colors = ["#ff4b4b" if val > 0 else "#28a745" for val in violation_df["Risk Contribution Score"]]
        ax.barh(violation_df["Component"], violation_df["Risk Contribution Score"], color=colors)
        ax.set_xlabel("Compliance Score Deduction Points")
        ax.set_title("Local Risk Weight Isolation Breakdown")
        st.pyplot(fig)
    else:
        st.success(" Zero risk deductions detected. System parameters are running fully optimal.")

with right_col:
    st.markdown("###  Agentic Playbook Remediation Script")
    if not processed_df.empty and violations_count > 0:
        violated_components = violation_df["Component"].tolist()
        selected_component = st.selectbox("Select Component to Remediate:", violated_components)
        
        if selected_component == "Firewall":
            st.code("# Secure Shell public mitigation rule\niptables -A INPUT -p tcp --dport 22 -j DROP\nservice iptables save", language="bash")
        elif selected_component == "Storage":
            st.code("# Enforce server-side KMS encryption standard\naws s3api put-bucket-encryption \\\n  --bucket prod-core-db-01-data \\\n  --server-side-encryption-configuration '{\"Rules\": [{\"ApplyServerSideEncryptionByDefault\": {\"SSEAlgorithm\": \"aws:kms\"}}]}'", language="bash")
        elif selected_component == "Patching":
            st.code("# Adjust system-d configuration timer lifecycle\nsed -i 's/Unattended-Upgrade::Automatic-Reboot-Interval \"14\";/Unattended-Upgrade::Automatic-Reboot-Interval \"7\";/g' /etc/apt/apt.conf.d/50unattended-upgrades", language="bash")
    else:
        st.success("Infrastructure matches policy constraints. No remediation actions required.")
