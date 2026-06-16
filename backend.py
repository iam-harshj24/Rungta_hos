# backend.py
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from core.rag_pipeline import DoctorCoPilotRAG
from data.mock_data import get_all_mock_documents

app = FastAPI(
    title="Rungta Hospital Doctor's Co-Pilot Backend",
    description="FastAPI backend for Clinical RAG Decision Support System",
    version="1.0.0"
)

# Global instance of RAG system
rag_instance: Optional[DoctorCoPilotRAG] = None

class InitializeRequest(BaseModel):
    api_key: str = Field(..., description="Gemini API Key")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Raw text clinical query")
    use_filter: bool = Field(True, description="Whether to apply deterministic metadata pre-filtering")
    k: int = Field(10, description="Number of chunks to retrieve")
    api_key: str = Field(..., description="Gemini API Key")

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "database_loaded": rag_instance is not None}

@app.post("/api/initialize")
def initialize_database(req: InitializeRequest):
    global rag_instance
    try:
        # Re-initialize only if key changes or not initialized yet
        if rag_instance is None or rag_instance.api_key != req.api_key:
            rag_instance = DoctorCoPilotRAG(google_api_key=req.api_key)
            mock_docs = get_all_mock_documents()
            rag_instance.load_data(mock_docs)
            
        status_msg = "Database initialized successfully"
        if rag_instance.is_fallback:
            status_msg += " (Local Fallback Mode active due to API key permissions/restrictions)"
            
        return {
            "status": "success",
            "message": status_msg,
            "chunks_count": len(rag_instance.raw_docs),
            "is_fallback": rag_instance.is_fallback
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

@app.post("/api/query")
def execute_query(req: QueryRequest):
    global rag_instance
    try:
        # Ensure database is initialized
        if rag_instance is None or rag_instance.api_key != req.api_key:
            rag_instance = DoctorCoPilotRAG(google_api_key=req.api_key)
            mock_docs = get_all_mock_documents()
            rag_instance.load_data(mock_docs)
            
        results = rag_instance.execute_workflow(
            query=req.query,
            use_filter=req.use_filter,
            k=req.k
        )
        
        # Serialize the Document objects to dict for JSON transmission
        serialized_chunks = []
        for doc, score in results["retrieved_chunks"]:
            serialized_chunks.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
            
        return {
            "query": results["query"],
            "patient_id": results["patient_id"],
            "summary": results["summary"],
            "retrieved_chunks": serialized_chunks,
            "metrics": results["metrics"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")

@app.get("/api/patients")
def get_patients():
    try:
        mock_docs = get_all_mock_documents()
        return {"status": "success", "data": mock_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
