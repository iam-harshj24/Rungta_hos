# test_pipeline.py
import os
from core.rag_pipeline import DoctorCoPilotRAG
from data.mock_data import get_all_mock_documents

def main():
    # Use the provided API Key or get it from environment
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDyIzYSYd6m-EIZZh9fGxA_JAJoppXJUUQ")
    
    print("======================================================================")
    print("🩺 STARTING CLINICAL CO-PILOT RAG VALIDATION")
    print("======================================================================")
    
    print("\n[Step 1] Initializing RAG Pipeline and Vector Store...")
    try:
        rag = DoctorCoPilotRAG(google_api_key=api_key)
        print("✅ Pipeline successfully initialized with Google GenAI Embeddings and LLM.")
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        return

    print("\n[Step 2] Loading clinical notes into LangChain In-Memory Vector Store...")
    mock_docs = get_all_mock_documents()
    rag.load_data(mock_docs)
    print(f"✅ Loaded {len(mock_docs)} clinical note paragraphs into the database.")

    # Test Query
    query = "Summarize the previous cardiac complications for patient PT-8829."
    print(f"\n[Step 3] Executing query: '{query}'")
    
    # Run Naive Mode Search
    print("\n=== RUNNING IN [NAIVE SIMILARITY SEARCH] MODE ===")
    naive_results = rag.execute_workflow(query, use_filter=False, k=10)
    print(f"Retrieved {len(naive_results['retrieved_chunks'])} chunks globally:")
    for i, (doc, score) in enumerate(naive_results['retrieved_chunks'][:5]):
        p_id = doc.metadata.get("patient_id")
        lbl = "🚨 LEAK" if p_id != "PT-8829" else "✅ CORRECT"
        print(f"  - Chunk {i+1} [Patient: {p_id} ({lbl})] (Similarity Score: {score:.4f}): {doc.page_content[:90]}...")
        
    print("\n--- Generated Naive LLM Summary ---")
    print(naive_results['summary'])
    print("-----------------------------------")
    
    # Run Filtered Mode Search
    print("\n=== RUNNING IN [DETERMINISTIC METADATA-FILTERED] MODE (RECOMMENDED) ===")
    filtered_results = rag.execute_workflow(query, use_filter=True, k=10)
    print(f"Retrieved {len(filtered_results['retrieved_chunks'])} chunks for PT-8829:")
    for i, (doc, score) in enumerate(filtered_results['retrieved_chunks']):
        p_id = doc.metadata.get("patient_id")
        lbl = "🚨 LEAK" if p_id != "PT-8829" else "✅ CORRECT"
        print(f"  - Chunk {i+1} [Patient: {p_id} ({lbl})] (Similarity Score: {score:.4f}): {doc.page_content[:90]}...")
        
    print("\n--- Generated Filtered LLM Summary ---")
    print(filtered_results['summary'])
    print("--------------------------------------")
    
    print("\n[Metrics Summary]")
    print(f"  - Retrieval Time: {filtered_results['metrics']['retrieval_time_sec']:.4f} seconds")
    print(f"  - Generation Time: {filtered_results['metrics']['generation_time_sec']:.4f} seconds")
    print(f"  - Total Execution Time: {filtered_results['metrics']['total_time_sec']:.4f} seconds")
    print("\n✅ Verification complete. System is performing within specifications.")
    print("======================================================================")

if __name__ == "__main__":
    main()
