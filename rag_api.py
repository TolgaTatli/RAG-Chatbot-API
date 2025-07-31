from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from rag_system import RAGRetriever
from ollama_rag import OllamaRAGQA

# Pydantic modelleri
class GenerateRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class GenerateResponse(BaseModel):
    answer: str
    confidence: float
    method: str
    sources: List[Dict[str, Any]]

# FastAPI uygulaması
app = FastAPI(
    title="RAG API",
    description="Ollama tabanlı RAG (Retrieval-Augmented Generation) API",
    version="1.0.0"
)

# CORS yapılandırması
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm originlere izin ver (geliştirme için)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
retriever = None
qa_system = None

@app.on_event("startup")
async def startup_event():
    """Uygulama başladığında çalıştırılır"""
    global retriever, qa_system

    print("🦙 RAG API başlatılıyor...")

    try:
        # Retriever'ı başlat
        retriever = RAGRetriever()

        # Önceden oluşturulmuş index'i yükle
        retriever.load_from_files("faiss_index.bin", "documents.pkl")

        # Ollama QA sistemi oluştur
        qa_system = OllamaRAGQA(retriever, model_name="gemma3")

        print("✅ RAG sistemi başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ RAG sistemi yüklenirken hata oluştu: {e}")

@app.get("/")
async def root():
    """API ana sayfası"""
    return {
        "status": "active",
        "message": "RAG API çalışıyor! POST /generate endpoint'ini kullanın."
    }

@app.get("/status")
async def check_status():
    """Sistem durumunu kontrol et"""
    if not qa_system or not retriever:
        raise HTTPException(status_code=500, detail="RAG sistemi henüz başlatılmadı")

    ollama_status = qa_system.check_ollama_status()

    return {
        "rag_system": "ready" if retriever else "not_loaded",
        "ollama_status": "online" if ollama_status else "offline",
        "model": qa_system.model_name,
        "index_status": "loaded" if retriever.index is not None else "not_loaded",
        "document_count": len(retriever.documents) if retriever.documents else 0
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Sorguya yanıt oluştur

    - **question**: Kullanıcı sorusu
    - **top_k**: Kullanılacak döküman sayısı [isteğe bağlı]
    """
    if not qa_system or not retriever:
        raise HTTPException(status_code=500, detail="RAG sistemi henüz başlatılmadı")

    try:

        result = qa_system.answer_question(
            question=request.question,
            top_k=request.top_k
        )

        # Yanıtı döndür
        return {
            "answer": result["answer"],
            "confidence": result.get("confidence", 0),
            "method": result.get("method", "unknown"),
            "sources": result.get("sources", [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata oluştu: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
