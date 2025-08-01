from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
from datetime import datetime
from typing import Optional, Dict, Any
import json

class SupabaseLogger:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def log_conversation(self,
                        question: str,
                        answer: str,
                        model_name: str,
                        confidence: float = 0.0,
                        sources: Optional[list] = None,
                        response_time: Optional[float] = None,
                        user_id: Optional[str] = None) -> bool:
        try:
            data = {
                "question": question,
                "answer": answer,
                "model_name": model_name,
                "confidence": confidence,
                "sources": json.dumps(sources) if sources else None,
                "response_time": response_time,
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat()
            }

            result = self.supabase.table("conversations").insert(data).execute()
            return True
        except Exception as e:
            print(f"Supabase kayıt hatası: {e}")
            return False

    def get_conversation_history(self, user_id: Optional[str] = None, limit: int = 50):
        try:
            query = self.supabase.table("conversations").select("*")

            if user_id:
                query = query.eq("user_id", user_id)

            result = query.order("created_at", desc=True).limit(limit).execute()
            return result.data
        except Exception as e:
            print(f"Geçmiş getirme hatası: {e}")
            return []