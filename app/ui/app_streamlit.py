import streamlit as st
import requests
import json
import os, re
import time
from datetime import datetime
import pandas as pd

FASTAPI_PORT = os.getenv("FASTAPI_PORT", "8000")
FASTAPI_URL = f"http://localhost:{FASTAPI_PORT}"

st.set_page_config(page_title="Policy-Aware RAG", page_icon="🧠", layout="wide")
st.title("RA3G")
st.text("🧠 Policy-Aware RAG System with Governance Control")

# Tabs: Chat | Status | Logs | Configuration
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "📊 Status", "📋 Logs", "⚙️ Configuration"])

# ---------------------------------------------------
# TAB 1 — CHAT INTERFACE
# ---------------------------------------------------
with tab1:
    # Session state initialization
    if "session_id" not in st.session_state:
        st.session_state.session_id = "demo_session"
    if "history" not in st.session_state:
        st.session_state.history = []

    st.sidebar.title("⚙️ Controls")

    if st.sidebar.button("Check System Health", key="check_health_btn_tab1"):
        resp = requests.get(f"{FASTAPI_URL}/health")
        if resp.status_code == 200:
            st.sidebar.success("✅ System is healthy")
            st.sidebar.json(resp.json())
        else:
            st.sidebar.error("❌ Health check failed")

    if st.sidebar.button("Clear Memory", key="clear_memory_btn_tab1"):
        resp = requests.delete(
            f"{FASTAPI_URL}/memory/clear", headers={"session_id": st.session_state.session_id}
        )
        if resp.status_code == 200:
            st.session_state.history = []
            st.sidebar.success("Memory cleared successfully.")
        else:
            st.sidebar.error("No memory found for this session.")

    # Main chat interface
    query = st.text_area("💬 Enter your question:", placeholder="Ask something...", key="query_input_tab1")
    top_k = st.slider("Top K retrieved passages", 1, 10, 5, key="topk_slider_tab1")

    if st.button("Submit Query", key="submit_btn_tab1"):
        if query.strip():
            payload = {"query": query, "top_k": top_k}
            headers = {"session_id": st.session_state.session_id}
            with st.spinner("🔍 Processing your query..."):
                resp = requests.post(f"{FASTAPI_URL}/query", json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.history.append(data)
                st.success("✅ Query processed successfully!")
                st.write("### Answer:")
                st.markdown(f"**{data['answer']}**")

                confidence = data.get("confidence", 0)
                st.progress(confidence)
                st.write(f"**Confidence:** {confidence*100:.1f}%")

                with st.expander("Governance Details"):
                    st.json(data["governance"])
                with st.expander("Retrieved Passages"):
                    st.json(data["retrieved"])
                with st.expander("Trace / Reasoning Steps"):
                    st.json(data["trace"])
            else:
                st.error("Error processing query.")
        else:
            st.warning("Please enter a query before submitting.")

    if st.session_state.history:
        st.markdown("---")
        st.write("### 🧠 Conversation History")
        for turn in reversed(st.session_state.history):
            st.markdown(f"**Q:** {turn['query']}")
            st.markdown(f"**A:** {turn['answer']}")

# ---------------------------------------------------
# TAB 2 — STATUS MONITOR
# ---------------------------------------------------
with tab2:
    st.subheader("📊 Agent Status Monitor")
    
    # Manual refresh button
    col_refresh, col_auto = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh Now", key="manual_refresh_btn"):
            fetch_health_data.clear()
            st.rerun()
    
    with col_auto:
        auto_refresh = st.checkbox("🔄 Auto-refresh (10s)", value=False, key="auto_refresh_status")
        refresh_interval = 10
    
    # Initialize refresh timer in session state
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Cache health data for 10 seconds
    @st.cache_data(ttl=refresh_interval)
    def fetch_health_data():
        """Fetch health data from API with caching."""
        try:
            resp = requests.get(f"{FASTAPI_URL}/health", timeout=2)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            return {"error": str(e)}
    
    # Status indicator mapping
    def get_status_emoji(status: str) -> str:
        """Get emoji for status."""
        status_map = {
            "healthy": "🟢",
            "slow": "🟡",
            "down": "🔴"
        }
        return status_map.get(status, "⚪")
    
    # Fetch and display status
    health_data = fetch_health_data()
    
    if health_data and "error" not in health_data:
        agents = health_data.get("agents", {})
        overall_status = health_data.get("status", "unknown")
        
        # Overall status
        st.markdown(f"**Overall System Status:** {get_status_emoji('healthy' if overall_status == 'ok' else 'down')} {overall_status.upper()}")
        st.markdown("---")
        
        # Agent status cards
        col1, col2 = st.columns(2)
        
        agent_names = ["gateway", "retriever", "reasoner", "governance"]
        agent_display = {
            "gateway": "🌐 Gateway",
            "retriever": "🔍 Retriever",
            "reasoner": "🧠 Reasoning",
            "governance": "🛡️ Governance"
        }
        
        for idx, agent_name in enumerate(agent_names):
            col = col1 if idx % 2 == 0 else col2
            
            with col:
                if agent_name in agents:
                    agent_status = agents[agent_name]
                    status = agent_status.get("status", "unknown")
                    uptime = agent_status.get("uptime_seconds", 0)
                    latency = agent_status.get("latency_ms", 0)
                    errors = agent_status.get("error_count", 0)
                    requests = agent_status.get("request_count", 0)
                    last_activity = agent_status.get("last_activity", "N/A")
                    
                    # Status card
                    with st.expander(f"{get_status_emoji(status)} {agent_display.get(agent_name, agent_name)}", expanded=False):
                        st.metric("Status", status.upper())
                        st.metric("Uptime", f"{uptime:.1f}s")
                        st.metric("Avg Latency", f"{latency:.1f}ms")
                        st.metric("Errors", errors)
                        st.metric("Requests", requests)
                        st.caption(f"Last activity: {last_activity}")
                        
                        # View detailed status
                        if st.button(f"📋 View Details", key=f"details_{agent_name}"):
                            try:
                                detail_resp = requests.get(f"{FASTAPI_URL}/health/{agent_name}", timeout=2)
                                if detail_resp.status_code == 200:
                                    st.json(detail_resp.json())
                            except Exception as e:
                                st.error(f"Failed to fetch details: {e}")
                else:
                    st.info(f"⚠️ {agent_display.get(agent_name, agent_name)}: No data available")
        
        # Auto-refresh logic with placeholder
        refresh_placeholder = st.empty()
        if auto_refresh:
            current_time = time.time()
            elapsed = current_time - st.session_state.last_refresh
            if elapsed >= refresh_interval:
                st.session_state.last_refresh = current_time
                # Clear cache to force refresh
                fetch_health_data.clear()
                st.rerun()
            else:
                remaining = refresh_interval - elapsed
                refresh_placeholder.caption(f"⏱️ Next refresh in {remaining:.1f}s")
        else:
            refresh_placeholder.empty()
    else:
        error_msg = health_data.get("error", "Unknown error") if health_data else "Failed to fetch health data"
        st.error(f"❌ Failed to fetch agent status: {error_msg}")
        st.info("Make sure the FastAPI backend is running.")

# ---------------------------------------------------
# TAB 3 — LOG VIEWER
# ---------------------------------------------------
with tab3:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    LOG_FILES = {
        "Gateway": os.path.join(LOG_DIR, "gateway.log"),
        "Retriever": os.path.join(LOG_DIR, "retriever.log"),
        "Reasoning": os.path.join(LOG_DIR, "reasoning.log"),
        "Governance": os.path.join(LOG_DIR, "governance.log"),
    }

    st.subheader("Logs")

    # --- Button to clear all logs ---
    if st.button("Clear All Logs", key="clear_all_logs_btn"):
        deleted_files = []
        for name, path in LOG_FILES.items():
            if os.path.exists(path):
                open(path, "w").close()  # truncate file content
                deleted_files.append(name)
        if deleted_files:
            st.success(f"✅ Cleared logs for: {', '.join(deleted_files)}")
        else:
            st.warning("No log files found to clear.")

    # --- Log selection and display ---
    log_choice = st.selectbox("Select a log file:", list(LOG_FILES.keys()), key="log_choice_tab2")
    log_path = LOG_FILES[log_choice]

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.readlines()
    except FileNotFoundError:
        st.error(f"❌ Log file not found: `{log_path}`")
        st.stop()

    search_term = st.text_input("Search keyword:", key="log_search_term_tab2")
    show_errors_only = st.checkbox("Show only errors", key="errors_only_checkbox_tab2")
    limit_lines = st.slider("Limit number of lines", 50, 1000, 300, key="limit_lines_slider_tab2")

    def parse_log_line(line):
        match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?\d*\s*-\s*(\w+)\s*-\s*(.*)", line)
        if match:
            time_str, level, msg = match.groups()
            try:
                timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                timestamp = None
            return {"time": timestamp, "level": level, "message": msg.strip()}
        return {"time": None, "level": "UNKNOWN", "message": line.strip()}

    data = [parse_log_line(line) for line in log_content[-limit_lines:]]
    df = pd.DataFrame(data)

    if search_term:
        df = df[df["message"].str.contains(search_term, case=False, na=False)]
    if show_errors_only:
        df = df[df["level"].isin(["ERROR", "CRITICAL"])]

    st.caption(f"Showing last {len(df)} lines (filtered)")

    if df.empty:
        st.warning("No log entries match your filters.")
    else:
        st.dataframe(df, width='stretch', hide_index=True)


# ---------------------------------------------------
# TAB 4 — CONFIGURATION EDITOR
# ---------------------------------------------------
with tab4:
    st.subheader("⚙️ Configuration Settings")

    CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config.yml"))

    if not os.path.exists(CONFIG_PATH):
        st.error(f"Configuration file not found: {CONFIG_PATH}")
        st.stop()

    import yaml

    # --- Load current configuration
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    st.caption("Modify the configuration values below and click **Save Changes**.")

    # Editable YAML fields
    editable_config = {}
    for key, value in config_data.items():
        if isinstance(value, bool):
            editable_config[key] = st.checkbox(key, value=value)
        elif isinstance(value, (int, float)):
            editable_config[key] = st.number_input(key, value=value)
        elif isinstance(value, list):
            editable_config[key] = st.text_area(
                key, value=", ".join(map(str, value)), help="Comma-separated list"
            )
        else:
            editable_config[key] = st.text_input(key, value=value)

    if st.button("💾 Save Changes", key="save_config_btn"):
        # Convert comma-separated strings back to lists
        for key, value in editable_config.items():
            if isinstance(config_data.get(key), list):
                editable_config[key] = [v.strip() for v in value.split(",") if v.strip()]

        # Save updated YAML
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(editable_config, f, sort_keys=False, allow_unicode=True)

        st.success("✅ Configuration saved successfully!")
        st.info("Please restart the service for changes to take effect.")
        st.json(editable_config)

    if st.button("🔄 Reload from File", key="reload_config_btn"):
        st.rerun()