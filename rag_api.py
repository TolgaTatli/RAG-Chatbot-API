import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware

from rag_system import RAGRetriever
from ollama_rag import OllamaRAGQA
from supabase_client import SupabaseLogger


# ====== REQUEST/RESPONSE MODELS ======

class GenerateRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    user_id: Optional[str] = None


class GenerateResponse(BaseModel):
    answer: str
    confidence: float
    method: str
    sources: List[Dict[str, Any]]


class AuthRequest(BaseModel):
    email: str
    password: str


class SignUpRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


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
supabase_logger = None
security = HTTPBearer(auto_error=False)


# ====== AUTH DEPENDENCY ======

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """JWT token'dan kullanıcı bilgilerini çıkar (opsiyonel)"""
    if not credentials:
        print("🔍 get_current_user: Token bulunamadı")
        return None
    
    if not supabase_logger:
        print("🔍 get_current_user: Supabase logger yok")
        return None
        
    try:
        print(f"🔍 get_current_user: Token alındı - {credentials.credentials[:20]}...")
        user_info = supabase_logger.get_user(credentials.credentials)
        if user_info["success"]:
            print(f"✅ get_current_user: Kullanıcı doğrulandı - {user_info['user'].email}")
            return user_info["user"]
        else:
            print(f"❌ get_current_user: Token geçersiz - {user_info.get('message', 'Bilinmeyen hata')}")
        return None
    except Exception as e:
        print(f"❌ get_current_user exception: {e}")
        return None


async def require_auth(user = Depends(get_current_user)):
    """Auth zorunlu endpoint'ler için"""
    if not user:
        raise HTTPException(
            status_code=401, 
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user


@app.on_event("startup")
async def startup_event():
    global retriever, qa_system, supabase_logger

    print("🦙 RAG API başlatılıyor...")

    try:
        retriever = RAGRetriever()
        retriever.load_from_files("faiss_index.bin", "documents.pkl")
        qa_system = OllamaRAGQA(retriever, model_name="gemma3")
        supabase_logger = SupabaseLogger()

        print("RAG sistemi başarıyla yüklendi.")
        print("Supabase başarıyla bağlandı ve çalışıyor!")
    except Exception as e:
        print(f"RAG sistemi yüklenirken hata oluştu: {e}")
        supabase_logger = None


@app.get("/")
async def root():
    return {
        "status": "active",
        "message": "RAG API çalışıyor! POST /generate endpoint'ini kullanın."
    }


@app.get("/status")
async def check_status():
    if not qa_system or not retriever:
        raise HTTPException(status_code=500, detail="RAG sistemi henüz başlatılmadı")

    ollama_status = qa_system.check_ollama_status()

    return {
        "rag_system": "ready" if retriever else "not_loaded",
        "ollama_status": "online" if ollama_status else "offline",
        "model": qa_system.model_name,
        "index_status": "loaded" if retriever.index is not None else "not_loaded",
        "document_count": len(retriever.documents) if retriever.documents else 0,
        "supabase_auth": "available" if supabase_logger else "unavailable"
    }


# ====== AUTHENTICATION ENDPOINTS ======

@app.post("/auth/signup")
async def signup(request: SignUpRequest):
    """Yeni kullanıcı kaydı"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Auth servisi mevcut değil")
    
    metadata = {"full_name": request.full_name} if request.full_name else None
    result = supabase_logger.sign_up(request.email, request.password, metadata)
    
    if result["success"]:
        return {
            "message": result["message"],
            "user_id": result["user"].id if result["user"] else None
        }
    else:
        raise HTTPException(status_code=400, detail=result["message"])


@app.post("/auth/signin")
async def signin(request: AuthRequest):
    """Kullanıcı girişi"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Auth servisi mevcut değil")
    
    result = supabase_logger.sign_in(request.email, request.password)
    
    if result["success"]:
        return {
            "message": result["message"],
            "access_token": result["access_token"],
            "clear_chat": True,
            "user": {
                "id": result["user"].id,
                "email": result["user"].email,
                "created_at": result["user"].created_at
            }
        }
    else:
        raise HTTPException(status_code=401, detail=result["message"])


@app.post("/auth/signout")
async def signout():
    """Kullanıcı çıkışı"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Auth servisi mevcut değil")
    
    result = supabase_logger.sign_out()
    return {"message": result["message"]}


@app.get("/auth/me")
async def get_me(user = Depends(require_auth)):
    """Mevcut kullanıcı bilgilerini getir"""
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "created_at": user.created_at,
            "user_metadata": user.user_metadata
        }
    }


@app.get("/auth/stats")
async def get_user_stats(user = Depends(require_auth)):
    """Kullanıcının conversation istatistiklerini getir"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Supabase bağlantısı yok")
    
    stats = supabase_logger.get_user_stats(user.id)
    if stats["success"]:
        return stats["stats"]
    else:
        raise HTTPException(status_code=500, detail=stats["message"])


@app.get("/auth/models")
async def get_user_models(user = Depends(require_auth)):
    """Kullanıcının kullandığı modelleri getir"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Supabase bağlantısı yok")
    
    models = supabase_logger.get_user_models(user.id)
    return {"models": models}


# ====== RAG ENDPOINTS ======


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, current_user = Depends(get_current_user)):
    if not qa_system or not retriever:
        raise HTTPException(status_code=500, detail="RAG sistemi henüz başlatılmadı")

    # Authenticated user'ın ID'sini kullan, yoksa request'teki user_id'yi kullan
    user_id = current_user.id if current_user else request.user_id

    start_time = time.time()
    conversation_saved = False
    conversation_id = None

    try:
        result = qa_system.answer_question(
            question=request.question,
            top_k=request.top_k
        )

        response_time = time.time() - start_time

        # Chat kaydetme işlemini güçlendir ve SENKRON yap
        if supabase_logger and result.get("answer"):
            try:
                conversation_result = supabase_logger.log_conversation_with_id(
                    question=request.question,
                    answer=result["answer"],
                    model_name=qa_system.model_name,
                    confidence=result.get("confidence", 0),
                    sources=result.get("sources", []),
                    response_time=response_time,
                    user_id=user_id
                )
                if conversation_result["success"]:
                    conversation_saved = True
                    conversation_id = conversation_result["conversation_id"]
                    print(f"✓ Conversation kaydedildi (ID: {conversation_id}, User: {user_id or 'Anonymous'}, Response: {len(result['answer'])} chars)")
                else:
                    print(f"✗ Conversation kaydetme başarısız (User: {user_id or 'Anonymous'})")
            except Exception as save_error:
                print(f"❌ Conversation kaydetme hatası: {save_error}")
        else:
            if not supabase_logger:
                print("⚠️ Supabase logger mevcut değil - conversation kaydedilmedi!")
            elif not result.get("answer"):
                print("⚠️ Boş answer - conversation kaydedilmedi!")

        return {
            "answer": result["answer"],
            "confidence": result.get("confidence", 0),
            "method": result.get("method", "unknown"),
            "sources": result.get("sources", []),
            "conversation_saved": conversation_saved,
            "conversation_id": conversation_id,
            "user_authenticated": current_user is not None
        }

    except Exception as e:
        print(f"❌ Generate endpoint hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata oluştu: {str(e)}")


@app.post("/generate-stream")
async def generate_stream(request: GenerateRequest, current_user = Depends(get_current_user)):
    if not qa_system or not retriever:
        raise HTTPException(status_code=500, detail="RAG sistemi henüz başlatılmadı")

    # Authenticated user'ın ID'sini kullan, yoksa request'teki user_id'yi kullan
    user_id = current_user.id if current_user else request.user_id

    start_time = time.time()
    full_answer = ""

    try:
        search_results = retriever.search(request.question, request.top_k)
        context = retriever.get_context_for_query(request.question, request.top_k)

        async def event_generator():
            nonlocal full_answer
            try:
                yield "data: " + json.dumps({
                    "type": "start",
                    "message": "Yanıt oluşturuluyor...",
                    "sources_count": len(search_results),
                    "authenticated": current_user is not None
                }) + "\n\n"

                for chunk in qa_system.generate_answer_stream(request.question, context):
                    if chunk:
                        full_answer += chunk  # Chunk'ları biriktir
                        yield "data: " + json.dumps({
                            "type": "chunk",
                            "text": chunk
                        }) + "\n\n"

                response_time = time.time() - start_time
                top_confidence = search_results[0]['score'] if search_results else 0
                conversation_saved = False
                conversation_id = None

                # Chat kaydetme işlemini güçlendir ve senkron yap
                if supabase_logger and full_answer.strip():
                    try:
                        conversation_result = supabase_logger.log_conversation_with_id(
                            question=request.question,
                            answer=full_answer,
                            model_name=qa_system.model_name,
                            confidence=top_confidence,
                            sources=search_results[:4],
                            response_time=response_time,
                            user_id=user_id
                        )
                        if conversation_result["success"]:
                            conversation_saved = True
                            conversation_id = conversation_result["conversation_id"]
                            print(f"✓ Streaming conversation kaydedildi (ID: {conversation_id}, User: {user_id or 'Anonymous'}, Response: {len(full_answer)} chars)")
                        else:
                            print(f"✗ Streaming conversation kaydetme başarısız (User: {user_id or 'Anonymous'})")
                    except Exception as save_error:
                        print(f"❌ Streaming conversation kaydetme hatası: {save_error}")
                else:
                    if not supabase_logger:
                        print("⚠️ Supabase logger mevcut değil - streaming conversation kaydedilmedi!")
                    elif not full_answer.strip():
                        print("⚠️ Boş answer - streaming conversation kaydedilmedi!")

                yield "data: " + json.dumps({
                    "type": "end",
                    "sources": search_results[:4],
                    "confidence": top_confidence,
                    "method": "ollama_with_rag" if context else "ollama_general",
                    "conversation_saved": conversation_saved,
                    "conversation_id": conversation_id,
                    "user_authenticated": current_user is not None
                }) + "\n\n"

            except Exception as e:
                print(f"❌ Streaming generator hatası: {str(e)}")
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
        print(f"❌ Streaming endpoint hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata oluştu: {str(e)}")

@app.get("/history")
async def get_history(
    current_user = Depends(require_auth),  # Auth zorunlu yap
    limit: int = Query(50)
):
    """Konuşma geçmişini getir - Sadece authenticated user kendi geçmişini görür"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Supabase bağlantısı yok")

    # Sadece authenticated user'ın kendi conversation'larını getir
    history = supabase_logger.get_conversation_history(current_user.id, limit)

    # Conversation count da ekle
    total_count = supabase_logger.get_conversation_count(current_user.id)

    return {
        "conversations": history,
        "user_authenticated": True,
        "user_id": current_user.id,
        "total_conversations": total_count,
        "showing": len(history),
        "limit": limit
    }


@app.get("/history/search")
async def search_history(
    search: str = Query(..., min_length=1),
    current_user = Depends(require_auth),
    limit: int = Query(20)
):
    """Authenticated user'ın conversation'larında arama yap"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Supabase bağlantısı yok")
    
    results = supabase_logger.search_conversations(current_user.id, search, limit)
    
    return {
        "conversations": results,
        "search_term": search,
        "found": len(results),
        "user_id": current_user.id
    }


@app.get("/history/{conversation_id}")
async def get_conversation_by_id(
    conversation_id: int,
    current_user = Depends(get_current_user)
):
    """Belirli bir conversation'ın detaylarını getir"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Supabase bağlantısı yok")
    
    # Conversation'ı getir
    conversation = supabase_logger.get_conversation_by_id(conversation_id, current_user.id if current_user else None)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation bulunamadı")
    
    return {
        "conversation": conversation,
        "user_authenticated": current_user is not None
    }

@app.delete("/history/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user = Depends(require_auth)
):
    """Belirli bir conversation'ı sil"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Supabase bağlantısı yok")
    
    success = supabase_logger.delete_conversation(conversation_id, current_user.id)
    
    if success:
        return {"message": "Conversation başarıyla silindi", "conversation_id": conversation_id}
    else:
        raise HTTPException(status_code=404, detail="Conversation bulunamadı veya silinirken hata oluştu")


@app.get("/history/count")
async def get_conversation_count(current_user = Depends(require_auth)):
    """Kullanıcının toplam conversation sayısını getir"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Supabase bağlantısı yok")
    
    count = supabase_logger.get_conversation_count(current_user.id)
    
    return {
        "user_id": current_user.id,
        "total_conversations": count
    }

@app.get("/history/latest")
async def get_latest_conversations(
    current_user = Depends(require_auth),
    limit: int = Query(5, description="Son kaç conversation getirileceği")
):
    """En son conversation'ları getir - real-time güncellemeler için"""
    if not supabase_logger:
        raise HTTPException(status_code=500, detail="Supabase bağlantısı yok")

    # Sadece authenticated user'ın son conversation'larını getir
    latest = supabase_logger.get_conversation_history(current_user.id, limit)

    return {
        "conversations": latest,
        "user_id": current_user.id,
        "count": len(latest),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
