from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware

from rag_system import RAGRetriever
from ollama_rag import OllamaRAGQA

class GenerateRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class GenerateResponse(BaseModel):
    answer: str
    confidence: float
    method: str
    sources: List[Dict[str, Any]]

app = FastAPI(
    title="RAG API",
    description="Ollama RAG A",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
qa_system = None

@app.on_event("startup")
async def startup_event():
    """Uygulama başladığında çalıştırılır"""
    global retriever, qa_system

    print("🦙 RAG API başlatılıyor...")

    try:
        retriever = RAGRetriever()
        retriever.load_from_files("faiss_index.bin", "documents.pkl")
        qa_system = OllamaRAGQA(retriever, model_name="gemma3")

        print("RAG sistemi başarıyla yüklendi.")
    except Exception as e:
        print(f"RAG sistemi yüklenirken hata oluştu: {e}")

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

        return {
            "answer": result["answer"],
            "confidence": result.get("confidence", 0),
            "method": result.get("method", "unknown"),
            "sources": result.get("sources", [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata oluştu: {str(e)}")

@app.post("/generate-stream")
async def generate_stream(request: GenerateRequest):
    """
    Sorguya yanıt oluştur ve sonucu gerçek streaming olarak döndür

    - **question**: Kullanıcı sorusu
    - **top_k**: Kullanılacak döküman sayısı [isteğe bağlı]
    """
    if not qa_system or not retriever:
        raise HTTPException(status_code=500, detail="RAG sistemi henüz başlatılmadı")

    try:
        search_results = retriever.search(request.question, request.top_k)
        context = retriever.get_context_for_query(request.question, request.top_k)

        async def event_generator():
            try:
                yield "data: " + json.dumps({
                    "type": "start",
                    "message": "Yanıt oluşturuluyor...",
                    "sources_count": len(search_results)
                }) + "\n\n"

                for chunk in qa_system.generate_answer_stream(request.question, context):
                    if chunk:
                        yield "data: " + json.dumps({
                            "type": "chunk",
                            "text": chunk
                        }) + "\n\n"

                top_confidence = search_results[0]['score'] if search_results else 0
                yield "data: " + json.dumps({
                    "type": "end",
                    "sources": search_results[:4],
                    "confidence": top_confidence,
                    "method": "ollama_with_rag" if context else "ollama_general"
                }) + "\n\n"

            except Exception as e:
                yield "data: " + json.dumps({
                    "type": "error",
                    "message": f"Hata oluştu: {str(e)}"
                }) + "\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata oluştu: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
