from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
from datetime import datetime
from typing import Optional, Dict, Any
import json

class SupabaseLogger:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # ====== AUTHENTICATION METHODS ======
    
    def sign_up(self, email: str, password: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Yeni kullanıcı kaydı"""
        try:
            response = self.supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {"data": metadata} if metadata else None
            })
            return {
                "success": True,
                "user": response.user,
                "session": response.session,
                "message": "Kayıt başarılı! Email'inizi kontrol edin."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Kayıt sırasında hata oluştu."
            }

    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Kullanıcı girişi"""
        try:
            response = self.supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return {
                "success": True,
                "user": response.user,
                "session": response.session,
                "access_token": response.session.access_token,
                "message": "Giriş başarılı!"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Giriş başarısız. Email veya şifre hatalı."
            }

    def sign_out(self) -> Dict[str, Any]:
        """Kullanıcı çıkışı"""
        try:
            self.supabase.auth.sign_out()
            return {
                "success": True,
                "message": "Çıkış başarılı!"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Çıkış sırasında hata oluştu."
            }

    def get_user(self, access_token: str) -> Dict[str, Any]:
        """Token'dan kullanıcı bilgilerini getir"""
        try:
            print(f"🔍 get_user çağrıldı - Token: {access_token[:20]}..." if access_token else "🔍 get_user çağrıldı - Token: None")
            
            # Token'ı headers'a set et
            response = self.supabase.auth.get_user(access_token)
            
            if response and response.user:
                print(f"✅ Token doğrulandı - User ID: {response.user.id}, Email: {response.user.email}")
                return {
                    "success": True,
                    "user": response.user,
                    "user_id": response.user.id,
                    "email": response.user.email
                }
            else:
                print("❌ Token geçersiz - user bulunamadı")
                return {
                    "success": False,
                    "message": "Geçersiz token"
                }
        except Exception as e:
            print(f"❌ Token doğrulama hatası: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Token doğrulama hatası"
            }


    # ====== CONVERSATION LOGGING (Güncellenmiş) ======

    def log_conversation(self,
                        question: str,
                        answer: str,
                        model_name: str,
                        confidence: float = 0.0,
                        sources: Optional[list] = None,
                        response_time: Optional[float] = None,
                        user_id: Optional[str] = None) -> bool:
        """
        Conversation'ı Supabase'e kaydet
        user_id artık UUID formatında olmalı (Supabase Auth user ID)
        """
        try:
            data = {
                "question": question,
                "answer": answer,
                "model_name": model_name,
                "confidence": confidence,
                "sources": sources,  # JSONB olduğu için json.dumps gerekmez
                "response_time": response_time,
                "user_id": user_id,  # UUID string
                "created_at": datetime.utcnow().isoformat()
            }

            result = self.supabase.table("conversations").insert(data).execute()
            return True
        except Exception as e:
            print(f"Supabase kayıt hatası: {e}")
            return False

    def get_conversation_history(self, user_id: str, limit: int = 50):
        """
        Kullanıcının conversation geçmişini getir
        user_id parametresi zorunludur - sadece o kullanıcının kayıtları getirilir
        """
        if not user_id:
            print("⚠️ get_conversation_history: user_id boş - hiçbir kayıt döndürülmüyor")
            return []

        try:
            # user_id kontrolü zorunlu - güvenlik için
            result = self.supabase.table("conversations").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()

            print(f"📊 get_conversation_history: {len(result.data)} kayıt bulundu (user_id: {user_id})")
            return result.data
        except Exception as e:
            print(f"Geçmiş getirme hatası: {e}")
            return []

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Kullanıcının conversation istatistiklerini getir"""
        try:
            # View'dan kullanıcı istatistiklerini al
            result = self.supabase.table("user_conversation_stats").select("*").eq("user_id", user_id).execute()
            
            if result.data and len(result.data) > 0:
                return {
                    "success": True,
                    "stats": result.data[0]
                }
            else:
                return {
                    "success": True,
                    "stats": {
                        "total_conversations": 0,
                        "avg_confidence": 0,
                        "last_conversation_at": None,
                        "favorite_model": None
                    }
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "İstatistik getirme hatası"
            }

    def get_conversation_count(self, user_id: str) -> int:
        """Kullanıcının toplam conversation sayısını getir"""
        try:
            # PostgreSQL function'ını çağır
            result = self.supabase.rpc("get_user_conversation_count", {"target_user_id": user_id}).execute()
            return result.data if result.data else 0
        except Exception as e:
            print(f"Conversation count hatası: {e}")
            return 0

    def search_conversations(self, user_id: str, search_term: str, limit: int = 20) -> list:
        """Kullanıcının conversation'larında arama yap"""
        try:
            result = self.supabase.table("conversations").select("*").eq("user_id", user_id).or_(
                f"question.ilike.%{search_term}%,answer.ilike.%{search_term}%"
            ).order("created_at", desc=True).limit(limit).execute()
            
            return result.data
        except Exception as e:
            print(f"Conversation arama hatası: {e}")
            return []

    def delete_conversation(self, conversation_id: int, user_id: str) -> bool:
        """Kullanıcının belirli bir conversation'ını sil"""
        try:
            result = self.supabase.table("conversations").delete().eq("id", conversation_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            print(f"Conversation silme hatası: {e}")
            return False

    def get_conversation_by_id(self, conversation_id: int, user_id: Optional[str] = None) -> Optional[dict]:
        """Belirli bir conversation'ın detaylarını getir"""
        try:
            query = self.supabase.table("conversations").select("*").eq("id", conversation_id)
            
            # Eğer user_id varsa, sadece o kullanıcının conversation'larına erişim ver
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.single().execute()
            return result.data
        except Exception as e:
            print(f"Conversation getirme hatası: {e}")
            return None

    def get_user_models(self, user_id: str) -> list:
        """Kullanıcının kullandığı model'ları ve sayılarını getir"""
        try:
            result = self.supabase.table("conversations").select("model_name").eq("user_id", user_id).execute()
            
            # Model sayımı yap
            models = {}
            for row in result.data:
                model = row.get("model_name", "unknown")
                models[model] = models.get(model, 0) + 1
            
            # Sayıya göre sırala
            sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)
            
            return [{"model_name": model, "count": count} for model, count in sorted_models]
        except Exception as e:
            print(f"Model listesi hatası: {e}")
            return []