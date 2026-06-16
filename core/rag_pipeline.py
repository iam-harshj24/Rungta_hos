# core/rag_pipeline.py
import re
import time
import math
from collections import Counter
from typing import List, Dict, Tuple, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

CLINICAL_PROMPT_TEMPLATE = """You are "The Doctor's Co-Pilot", a clinical assistant at Rungta Hospital.
You are helping a physician summarize historical patient records.

Requested Patient ID: {patient_id}
Doctor's Query: {query}

Here are the retrieved clinical note chunks from the Vector Database:
=========================================
{context_text}
=========================================

Instructions:
1. Provide a concise, medically-accurate clinical summary that directly answers the doctor's query.
2. Rely ONLY on the provided clinical note chunks. Do not make up any medical history or use external facts.
3. Critical Safety Warning: Verify that the patient ID in the retrieved chunks matches "{patient_id}". If you notice that chunks for a different patient ID (e.g., PT-1234 or others) have been retrieved due to database similarity matching (cross-patient leakage), EXPLICITLY highlight this discrepancy to warn the physician, ignore that mismatched content, and summarize only the data for "{patient_id}".
4. If the retrieved chunks contain no information relevant to the query, state clearly that no records for the query were found for this patient.
5. Format your response professionally. Use clean bullet points, highlighting key medical events, medications, and clinical interventions.

Clinical Summary:"""


class LocalTFIDFEmbeddings:
    """
    A lightweight, pure-Python fallback embedding generator using TF-IDF.
    Ensures the application works even if the Gemini API key has 403 or network errors.
    """
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.doc_count = 0
        
    def fit(self, texts: List[str]):
        self.doc_count = len(texts)
        df = Counter()
        for text in texts:
            words = set(self._tokenize(text))
            for word in words:
                df[word] += 1
                
        self.vocab = {word: idx for idx, word in enumerate(sorted(list(df.keys())))}
        for word, count in df.items():
            self.idf[word] = math.log((1 + self.doc_count) / (1 + count)) + 1
            
    def _tokenize(self, text: str) -> List[str]:
        # Simple lowercase word tokenizer
        return re.findall(r'\b[a-zA-Z0-9-]+\b', text.lower())
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.vocab:
            self.fit(texts)
            
        vectors = []
        for text in texts:
            vector = [0.0] * len(self.vocab)
            words = self._tokenize(text)
            tf = Counter(words)
            for word, count in tf.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    vector[idx] = count * self.idf[word]
            # L2 Normalization
            norm = math.sqrt(sum(v*v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            vectors.append(vector)
        return vectors
        
    def embed_query(self, text: str) -> List[float]:
        vector = [0.0] * len(self.vocab)
        words = self._tokenize(text)
        tf = Counter(words)
        for word, count in tf.items():
            if word in self.vocab:
                idx = self.vocab[word]
                vector[idx] = count * self.idf[word]
        # L2 Normalization
        norm = math.sqrt(sum(v*v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


def local_heuristic_summary(query: str, patient_id: str, retrieved_docs: List[Document]) -> str:
    """
    Generates a structured clinical summary from retrieved documents without calling the LLM API.
    Used as a fallback when the Gemini API key fails.
    """
    if not retrieved_docs:
        return f"No clinical notes found in the database for Patient {patient_id}."
        
    summary = f"### Clinical Summary for Patient {patient_id} (Local Fallback Mode)\n"
    summary += "*Notice: This summary was generated using local rule-based extraction because the Gemini API key provided returned a 403 Permission Denied error or is not configured.*\n\n"
    
    diagnoses = []
    procedures = []
    meds = []
    observations = []
    
    # Process text using simple pattern matching
    for doc in retrieved_docs:
        text = doc.page_content
        chunk_pid = doc.metadata.get("patient_id", "Unknown")
        if chunk_pid != patient_id:
            # Skip or flag data leak
            continue
            
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            lower_s = sentence.lower()
            if any(w in lower_s for w in ["admitted", "diagnosed", "history of", "stenosis", "ischemia", "diabetes", "copd", "nephropathy"]):
                diagnoses.append(sentence)
            elif any(w in lower_s for w in ["stent", "procedure", "catheterization", "cabg", "surgery", "debridement", "placed", "implanted", "defibrillator"]):
                procedures.append(sentence)
            elif any(w in lower_s for w in ["medication", "prescribed", "aspirin", "clopidogrel", "atorvastatin", "ramipril", "insulin", "prednisone"]):
                meds.append(sentence)
            else:
                observations.append(sentence)
                
    if diagnoses:
        summary += "**Key Diagnoses & Clinical Findings:**\n"
        for d in list(dict.fromkeys(diagnoses))[:4]:
            summary += f"- {d}.\n"
        summary += "\n"
        
    if procedures:
        summary += "**Procedures & Interventions:**\n"
        for p in list(dict.fromkeys(procedures))[:4]:
            summary += f"- {p}.\n"
        summary += "\n"
        
    if meds:
        summary += "**Medication Management:**\n"
        for m in list(dict.fromkeys(meds))[:4]:
            summary += f"- {m}.\n"
        summary += "\n"
        
    if not diagnoses and not procedures and not meds:
        summary += "**General Progress Notes:**\n"
        for o in list(dict.fromkeys(observations))[:5]:
            summary += f"- {o}.\n"
            
    return summary


class DoctorCoPilotRAG:
    def __init__(self, google_api_key: str):
        self.api_key = google_api_key
        self.is_fallback = False
        self.fallback_embeddings = None
        self.local_embeddings_cache = []
        self.raw_docs = []
        
        # Try to initialize Gemini API, but fail gracefully
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001", 
                google_api_key=google_api_key
            )
            self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=google_api_key,
                temperature=0.0
            )
        except Exception as e:
            print(f"[Warning] Gemini API Initialization failed: {e}. Running in local fallback mode.")
            self.is_fallback = True
            self.embeddings = None
            self.vector_store = None
            self.llm = None
            
        self.prompt_template = PromptTemplate(
            input_variables=["patient_id", "query", "context_text"],
            template=CLINICAL_PROMPT_TEMPLATE
        )
        
    def load_data(self, documents_data: List[Dict[str, str]]):
        """
        Loads clinical note paragraphs into the vector store.
        Each doc should have keys: 'text', 'patient_id', 'doctor_name'
        """
        self.raw_docs = [
            Document(
                page_content=doc["text"],
                metadata={
                    "patient_id": doc["patient_id"],
                    "doctor_name": doc["doctor_name"]
                }
            )
            for doc in documents_data
        ]
        
        if not self.is_fallback:
            try:
                self.vector_store.add_documents(self.raw_docs)
            except Exception as e:
                print(f"[Warning] Failed to load into Gemini Vector Store: {e}. Falling back to Local TF-IDF.")
                self.is_fallback = True
                
        if self.is_fallback:
            # Initialize local TF-IDF embeddings
            self.fallback_embeddings = LocalTFIDFEmbeddings()
            texts = [doc.page_content for doc in self.raw_docs]
            self.fallback_embeddings.fit(texts)
            # Embed all raw documents locally
            self.local_embeddings_cache = self.fallback_embeddings.embed_documents(texts)
        
    def extract_patient_id(self, query: str) -> str:
        """
        Extracts patient ID (format PT-XXXX) from the raw text query.
        Returns the patient ID (e.g., 'PT-8829') or None if not found.
        """
        match = re.search(r'\bPT-\d{4,5}\b', query, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        return "UNKNOWN"
        
    def _local_similarity_search(self, query: str, k: int = 10, filter_func = None) -> List[Tuple[Document, float]]:
        """
        Custom cosine similarity search using local TF-IDF embeddings.
        """
        query_vector = self.fallback_embeddings.embed_query(query)
        
        scored_docs = []
        for idx, doc in enumerate(self.raw_docs):
            if filter_func and not filter_func(doc):
                continue
                
            doc_vector = self.local_embeddings_cache[idx]
            # Calculate cosine similarity
            dot_prod = sum(q * d for q, d in zip(query_vector, doc_vector))
            norm_q = math.sqrt(sum(q*q for q in query_vector))
            norm_d = math.sqrt(sum(d*d for d in doc_vector))
            similarity = dot_prod / (norm_q * norm_d) if (norm_q * norm_d) > 0 else 0.0
            scored_docs.append((doc, similarity))
            
        # Sort by similarity score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]

    def retrieve_chunks(self, query: str, use_filter: bool = True, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Retrieves the top k chunks along with their similarity scores.
        If use_filter is True, it applies a metadata filter on the extracted patient_id.
        """
        patient_id = self.extract_patient_id(query)
        
        filter_func = None
        if use_filter and patient_id != "UNKNOWN":
            filter_func = lambda doc: doc.metadata.get("patient_id") == patient_id
            
        if self.is_fallback:
            return self._local_similarity_search(query, k=k, filter_func=filter_func)
            
        try:
            if filter_func:
                return self.vector_store.similarity_search_with_score(query, k=k, filter=filter_func)
            else:
                return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"[Warning] Gemini Embedding API call failed: {e}. Falling back to Local TF-IDF.")
            self.is_fallback = True
            if not self.fallback_embeddings:
                self.fallback_embeddings = LocalTFIDFEmbeddings()
                texts = [doc.page_content for doc in self.raw_docs]
                self.fallback_embeddings.fit(texts)
                self.local_embeddings_cache = self.fallback_embeddings.embed_documents(texts)
            return self._local_similarity_search(query, k=k, filter_func=filter_func)
            
    def generate_summary(self, query: str, retrieved_docs: List[Document]) -> str:
        """
        Builds the prompt using retrieved documents and executes the LLM call.
        """
        patient_id = self.extract_patient_id(query)
        
        if self.is_fallback:
            return local_heuristic_summary(query, patient_id, retrieved_docs)
            
        # Format the context text from the retrieved chunks
        context_chunks = []
        for i, doc in enumerate(retrieved_docs):
            chunk_info = (
                f"Chunk {i+1} [Patient: {doc.metadata.get('patient_id', 'N/A')}, "
                f"Doctor: {doc.metadata.get('doctor_name', 'N/A')}]:\n"
                f"{doc.page_content}"
            )
            context_chunks.append(chunk_info)
            
        context_text = "\n\n".join(context_chunks) if context_chunks else "No clinical notes found in Vector Database matching search criteria."
        
        # Create prompt
        prompt = self.prompt_template.format(
            patient_id=patient_id,
            query=query,
            context_text=context_text
        )
        
        try:
            # Invoke LLM
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"[Warning] Gemini LLM API call failed: {e}. Falling back to Local Heuristic Summarizer.")
            return local_heuristic_summary(query, patient_id, retrieved_docs)

    def execute_workflow(self, query: str, use_filter: bool = True, k: int = 10) -> Dict[str, Any]:
        """
        Runs the complete RAG workflow:
        1. Similarity search (naive or filtered)
        2. Format prompt and pass to LLM
        3. Returns generated summary and metadata metrics
        """
        start_time = time.time()
        
        # 1. Similarity Search
        scored_docs = self.retrieve_chunks(query, use_filter=use_filter, k=k)
        retrieval_time = time.time() - start_time
        
        # Extract documents
        docs = [doc for doc, _ in scored_docs]
        
        # 2. Summary Generation
        llm_start_time = time.time()
        summary = self.generate_summary(query, docs)
        llm_time = time.time() - llm_start_time
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "patient_id": self.extract_patient_id(query),
            "summary": summary,
            "retrieved_chunks": scored_docs,
            "metrics": {
                "retrieval_time_sec": retrieval_time,
                "generation_time_sec": llm_time,
                "total_time_sec": total_time,
                "chunks_count": len(docs),
                "is_fallback": self.is_fallback
            }
        }
