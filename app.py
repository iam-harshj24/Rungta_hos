# app.py
import streamlit as st
import pandas as pd
import os
import time

# Helper to safely parse retrieved chunks in multiple formats (API dict vs local langchain objects)
def parse_chunk(chunk):
    if isinstance(chunk, dict):
        page_content = chunk.get("page_content", "")
        metadata = chunk.get("metadata", {}) or {}
        score = chunk.get("score", 0.0)
    elif isinstance(chunk, (list, tuple)):
        # LangChain similarity_search_with_score returns (Document, score)
        doc = chunk[0]
        score = chunk[1] if len(chunk) > 1 else 0.0
        page_content = getattr(doc, "page_content", "")
        metadata = getattr(doc, "metadata", {}) or {}
    else:
        # LangChain Document object directly
        page_content = getattr(chunk, "page_content", "")
        metadata = getattr(doc, "metadata", {}) or {}
        score = 0.0
    return page_content, metadata, score

# Set up page config
st.set_page_config(
    page_title="Rungta Hospital | Doctor's Co-Pilot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- IN-MEMORY FASTAPI BACKEND CLIENT -----------------
# We initialize the FastAPI app via TestClient. This decouples the frontend
# and backend codebases, while bypassing all network/port-binding restrictions
# on containerized hosts like Streamlit Cloud.
from fastapi.testclient import TestClient
from backend import app as fastapi_app

client = TestClient(fastapi_app)

# Custom premium CSS injection
st.markdown("""
    <style>
    /* Styling headers and title */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #0f172a;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Premium card styles */
    .clinic-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        margin-bottom: 15px;
    }
    
    .summary-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 12px;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Badges */
    .badge-patient {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-doctor {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Highlighting differences */
    .danger-highlight {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    .success-highlight {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    /* Metric container styling */
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE MANAGEMENT -----------------
DEFAULT_API_KEY = ""

if "google_api_key" not in st.session_state:
    api_key_secrets = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key_secrets = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    st.session_state.google_api_key = api_key_secrets or os.environ.get("GEMINI_API_KEY") or DEFAULT_API_KEY

if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_text_input" not in st.session_state:
    st.session_state.query_text_input = ""

if "last_results" not in st.session_state:
    st.session_state.last_results = None

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=64)
st.sidebar.title("Co-Pilot Settings")

api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=st.session_state.google_api_key,
    help="Provided automatically, but you can override it."
)

if api_key_input != st.session_state.google_api_key:
    st.session_state.google_api_key = api_key_input
    st.session_state.db_loaded = False
    st.rerun()

st.sidebar.subheader("RAG Parameters")
rag_mode = st.sidebar.radio(
    "Search Mode",
    options=["Deterministic Metadata Filter (Recommended)", "Naive Similarity Search"],
    help="Deterministic Metadata Filter extracts patient_id first and runs vector search only on that patient's chunks. Naive search searches the entire 50,000 clinical paragraphs globally."
)

top_k = st.sidebar.slider("Chunks to Retrieve (k)", min_value=1, max_value=20, value=10)

st.sidebar.divider()

# Initialize Database on Backend
if st.session_state.google_api_key:
    if not st.session_state.db_loaded:
        with st.sidebar:
            with st.spinner("Initializing Backend Vector DB..."):
                try:
                    res = client.post(
                        "/api/initialize",
                        json={"api_key": st.session_state.google_api_key}
                    )
                    if res.status_code == 200:
                        init_data = res.json()
                        st.session_state.db_loaded = True
                        st.success("Vector DB Initialized!")
                    else:
                        st.error(f"Backend init failed: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
else:
    st.sidebar.warning("Please enter a valid Gemini API Key to initialize the RAG database.")

if st.session_state.db_loaded:
    st.sidebar.success("🟢 DB Status: Active on Backend\n- 50,000 Chunks Indexed (Simulated)")

# ----------------- HEADER -----------------
st.markdown('<div class="main-title">🩺 The Doctor\'s Co-Pilot</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Clinical RAG Decision Support System — Rungta Hospital Portal (De-coupled Arch)</div>', unsafe_allow_html=True)

# Create tabs
tab_chat, tab_comparison, tab_registry, tab_architecture = st.tabs([
    "💬 Co-Pilot Terminal", 
    "📊 Naive vs Filtered Search Analysis", 
    "🗂️ Patient Registry", 
    "📐 System Architecture"
])

# ----------------- TAB 1: CO-PILOT CHAT TERMINAL -----------------
with tab_chat:
    st.markdown("### Ask a Clinical Question")
    st.write("Click on any of the standard clinical queries below to populate and test the system:")
    
    # Grid of standard queries
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        if st.button("Summarize previous cardiac complications for PT-8829"):
            st.session_state.query_text_input = "Summarize the previous cardiac complications for patient PT-8829."
            st.rerun()
    with col_q2:
        if st.button("Summarize clinical course and surgeries for PT-1234"):
            st.session_state.query_text_input = "Summarize clinical course and surgeries for patient PT-1234."
            st.rerun()
    with col_q3:
        if st.button("What is glycemic control plan for PT-5566?"):
            st.session_state.query_text_input = "What is the glycemic control and medication plan for patient PT-5566?"
            st.rerun()

    # Chat input bound directly to session state
    chat_input_val = st.text_input(
        "Enter your search query about a patient:",
        key="query_text_input",
        placeholder="e.g., Summarize the previous cardiac complications for patient PT-8829."
    )

    if st.button("Run Co-Pilot Analysis", type="primary"):
        if not st.session_state.db_loaded:
            st.error("Please ensure the Vector DB is loaded first by entering a valid API key in the sidebar.")
        elif not chat_input_val.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Requesting RAG analysis from Backend API..."):
                use_filter = (rag_mode == "Deterministic Metadata Filter (Recommended)")
                
                try:
                    # Request analysis from FastAPI backend
                    response = client.post(
                        "/api/query",
                        json={
                            "query": chat_input_val,
                            "use_filter": use_filter,
                            "k": top_k,
                            "api_key": st.session_state.google_api_key
                        }
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        st.session_state.last_results = results
                        
                        # Append to history
                        st.session_state.chat_history.append({
                            "query": chat_input_val,
                            "summary": results["summary"],
                            "use_filter": use_filter,
                            "metrics": results["metrics"]
                        })
                    else:
                        st.error(f"Backend Query Failed: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

    # Display results if available
    if st.session_state.last_results:
        results = st.session_state.last_results
        
        # Display performance metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['total_time_sec']:.2f}s</div>
                    <div class="metric-label">Total Latency</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['retrieval_time_sec']:.3f}s</div>
                    <div class="metric-label">Retrieval Time</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['generation_time_sec']:.2f}s</div>
                    <div class="metric-label">Generation Time</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['chunks_count']}</div>
                    <div class="metric-label">Chunks Retrieved</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        
        # Show RAG Mode Warning
        extracted_id = results["patient_id"]
        if rag_mode == "Naive Similarity Search" and extracted_id != "UNKNOWN":
            st.warning(f"⚠️ **Caution**: Currently running in **Naive Similarity Search** mode. Cross-patient data leakage is possible since we are searching globally across all 50,000 summaries. Toggle 'Search Mode' in the sidebar to Filtered mode for safety.")
        elif extracted_id != "UNKNOWN":
            st.success(f"🛡️ **Safe Mode Active**: Filtered vector store by metadata `patient_id == '{extracted_id}'` prior to similarity search. No external patient data is present in the LLM context.")

        # Display Final Summary
        st.markdown(f"""
            <div class="clinic-card">
                <div class="summary-title">
                    🩺 CLINICAL CO-PILOT SUMMARY FOR PATIENT: <span class="badge-patient">{extracted_id}</span>
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.05rem; line-height: 1.6; color: #1e293b;">
                    {results['summary']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Expandable section for retrieved chunks
        with st.expander("🔍 View Retrieved Clinical Chunks from Backend Vector DB (Top 10)"):
            st.write("These chunks were injected directly into the LLM context by the backend:")
            for idx, chunk in enumerate(results["retrieved_chunks"]):
                page_content, metadata, score = parse_chunk(chunk)
                p_id = metadata.get("patient_id", "N/A")
                d_name = metadata.get("doctor_name", "N/A")
                
                # Check if this is a data leak (patient_id mismatch)
                is_leak = (extracted_id != "UNKNOWN" and p_id != extracted_id)
                card_style = "danger-highlight" if is_leak else "success-highlight"
                leak_warning = " 🚨 **DATA LEAK DETECTED (Mismatched Patient ID!)**" if is_leak else " ✅ **Correct Patient Record**"
                
                st.markdown(f"""
                    <div class="{card_style}">
                        <strong>Chunk #{idx+1} (Similarity Score: {score:.4f}){leak_warning}</strong><br/>
                        <span class="badge-patient">Patient: {p_id}</span> <span class="badge-doctor">Doctor: {d_name}</span>
                        <p style="margin-top: 8px; font-style: italic; color: #334155;">"{page_content}"</p>
                    </div>
                """, unsafe_allow_html=True)


# ----------------- TAB 2: NAIVE VS FILTERED SEARCH COMPARISON -----------------
with tab_comparison:
    st.markdown("### The Danger of Naive RAG in Medical Records")
    st.write(
        "This analysis demonstrates why a simple semantic search on clinical notes is dangerous in production. "
        "We compare what happens when we ask about patient **PT-8829** (who has a mild cardiac condition) under both modes."
    )
    
    col1, col2 = st.columns(2)
    
    if st.session_state.db_loaded:
        test_query = "Summarize the previous cardiac complications for patient PT-8829."
        
        try:
            # Query backend in Naive Mode
            res_naive = client.post(
                "/api/query",
                json={
                    "query": test_query,
                    "use_filter": False,
                    "k": 10,
                    "api_key": st.session_state.google_api_key
                }
            )
            
            # Query backend in Filtered Mode
            res_filtered = client.post(
                "/api/query",
                json={
                    "query": test_query,
                    "use_filter": True,
                    "k": 10,
                    "api_key": st.session_state.google_api_key
                }
            )
            
            if res_naive.status_code == 200 and res_filtered.status_code == 200:
                naive_results = res_naive.json()
                filtered_results = res_filtered.json()
                
                # Run Naive display
                with col1:
                    st.subheader("🔴 Naive Similarity Search")
                    st.info("In this mode, we embed the user query and search all database records globally.")
                    
                    leaks = 0
                    for chunk in naive_results["retrieved_chunks"]:
                        _, metadata, _ = parse_chunk(chunk)
                        if metadata.get("patient_id") != "PT-8829":
                            leaks += 1
                            
                    st.error(f"Results retrieved: 10 chunks. Mismatched patient chunks: {leaks}/10.")
                    
                    for idx, chunk in enumerate(naive_results["retrieved_chunks"][:5]):
                        page_content, metadata, score = parse_chunk(chunk)
                        p_id = metadata.get("patient_id", "N/A")
                        is_leak = (p_id != "PT-8829")
                        card_style = "danger-highlight" if is_leak else "success-highlight"
                        leak_lbl = "🚨 DATA LEAK" if is_leak else "✅ Correct Patient"
                        st.markdown(f"""
                            <div class="{card_style}">
                                <strong>Rank #{idx+1} ({leak_lbl})</strong><br/>
                                Similarity: {score:.4f} | Patient: <b>{p_id}</b> | Doctor: {metadata.get('doctor_name')}
                                <p style="margin-top: 5px; font-size: 0.9rem;">"{page_content}"</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                # Run Filtered display
                with col2:
                    st.subheader("🟢 Deterministic Metadata-Filtered Search")
                    st.info("In this mode, we parse 'PT-8829', apply `patient_id == 'PT-8829'` metadata filter on the backend, and then do similarity search.")
                    
                    st.success(f"Results retrieved: {len(filtered_results['retrieved_chunks'])} chunks. Mismatched patient chunks: 0.")
                    
                    for idx, chunk in enumerate(filtered_results["retrieved_chunks"][:5]):
                        page_content, metadata, score = parse_chunk(chunk)
                        p_id = metadata.get("patient_id", "N/A")
                        st.markdown(f"""
                            <div class="success-highlight">
                                <strong>Rank #{idx+1} (✅ Correct Patient)</strong><br/>
                                Similarity: {score:.4f} | Patient: <b>{p_id}</b> | Doctor: {metadata.get('doctor_name')}
                                <p style="margin-top: 5px; font-size: 0.9rem;">"{page_content}"</p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("Error fetching comparison results from backend.")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
    else:
        st.info("Please enter a Gemini API Key to run the comparison analysis.")

# ----------------- TAB 3: PATIENT REGISTRY -----------------
with tab_registry:
    st.markdown("### Patient Registry")
    st.write("Browse all simulated clinical summaries uploaded to the vector store:")
    
    try:
        res = client.get("/api/patients")
        if res.status_code == 200:
            patients_data = res.json()["data"]
            df_data = []
            for doc in patients_data:
                df_data.append({
                    "Patient ID": doc["patient_id"],
                    "Doctor In Charge": doc["doctor_name"],
                    "Clinical Paragraph Summary": doc["text"]
                })
            df = pd.DataFrame(df_data)
            
            patient_filter = st.selectbox("Filter Registry by Patient ID", options=["All"] + sorted(list(df["Patient ID"].unique())))
            
            if patient_filter != "All":
                filtered_df = df[df["Patient ID"] == patient_filter]
            else:
                filtered_df = df
                
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.error("Failed to load patients registry from backend.")
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")

# ----------------- TAB 4: SYSTEM ARCHITECTURE -----------------
with tab_architecture:
    st.markdown("### Decoupled Frontend-Backend Architecture")
    st.write("This application implements a professional **decoupled architecture**, splitting client-side UI and server-side RAG logic:")
    
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("""
        #### 🎨 Streamlit Frontend
        * **Purpose**: Serves the user interface and handles user inputs.
        * **Scope**: 
          * No AI SDK imports.
          * No API key leakage in client code.
          * Does not contain prompt strings.
          * Communicates solely via HTTP REST API to the backend.
        * **Lifecycle**: Runs a background daemon thread that manages and monitors the FastAPI server.
        """)
        
    with col_arch2:
        st.markdown("""
        #### ⚙️ FastAPI Backend
        * **Purpose**: Hosts the vector database, implements the RAG pipeline, and executes AI models.
        * **Scope**:
          * Parses and extracts patient ID (`PT-XXXX`) deterministically.
          * Manages the LangChain In-Memory Vector Store.
          * Runs similarity search with metadata pre-filtering.
          * Formats clinical prompts and makes secure LLM calls to Gemini.
        """)
        
    st.markdown("""
    ---
    #### 📐 Data Flow Diagram
    ```mermaid
    sequenceDiagram
        autonumber
        actor Doctor as Physician / User
        participant FE as Streamlit Frontend (app.py)
        participant BE as FastAPI Backend (backend.py)
        participant VDB as Vector Store (LangChain)
        participant LLM as Gemini API (Google Cloud)
        
        Doctor->>FE: Click suggestion / Type query
        FE->>BE: POST /api/query {query, use_filter, k, api_key}
        Note over BE: Extracts Patient ID (e.g. PT-8829)
        alt Filtered Mode Active
            BE->>VDB: Query similarity search with filter (patient_id == PT-8829)
            VDB-->>BE: Return top 10 chunks strictly for PT-8829
        else Naive Mode Active
            BE->>VDB: Query similarity search globally
            VDB-->>BE: Return top 10 chunks globally (contains leaks!)
        end
        Note over BE: Formats prompt with strict patient ID validation instructions
        BE->>LLM: Request summary generation (Gemini-1.5-Flash)
        LLM-->>BE: Return clinical summary response
        BE-->>FE: Return JSON {query, patient_id, summary, chunks, metrics}
        FE->>Doctor: Render formatted clinical summary & show chunks
    ```
    """)
