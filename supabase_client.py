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
            # Veri validasyonu
            if not question or not answer:
                print(f"❌ log_conversation: Eksik veri - question: {bool(question)}, answer: {bool(answer)}")
                return False

            if not question.strip() or not answer.strip():
                print(f"❌ log_conversation: Boş veri - question length: {len(question.strip())}, answer length: {len(answer.strip())}")
                return False

            data = {
                "question": question.strip(),
                "answer": answer.strip(),
                "model_name": model_name or "unknown",
                "confidence": confidence if confidence is not None else 0.0,
                "sources": sources if sources is not None else [],
                "response_time": response_time,
                "user_id": user_id,  # UUID string veya None
                "created_at": datetime.utcnow().isoformat()
            }

            print(f"🔍 log_conversation: Kayıt edilecek veri - user_id: {user_id}, question length: {len(data['question'])}, answer length: {len(data['answer'])}")

            result = self.supabase.table("conversations").insert(data).execute()

            if result.data:
                print(f"✅ log_conversation: Başarıyla kaydedildi - ID: {result.data[0].get('id', 'unknown')}")
                return True
            else:
                print(f"❌ log_conversation: Kayıt başarısız - result.data boş")
                return False

        except Exception as e:
            print(f"❌ log_conversation: Supabase kayıt hatası: {e}")
            print(f"   - question: {question[:100] if question else 'None'}...")
            print(f"   - answer: {answer[:100] if answer else 'None'}...")
            print(f"   - user_id: {user_id}")
            return False

    def log_conversation_with_id(self,
                        question: str,
                        answer: str,
                        model_name: str,
                        confidence: float = 0.0,
                        sources: Optional[list] = None,
                        response_time: Optional[float] = None,
                        user_id: Optional[str] = None,
                        thread_id: Optional[str] = None,
                        parent_message_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Conversation'ı Supabase'e kaydet ve ID'sini döndür
        thread_id varsa mevcut thread'e ekle, yoksa yeni thread oluştur
        """
        try:
            # Veri validasyonu
            if not question or not answer:
                print(f"❌ log_conversation_with_id: Eksik veri - question: {bool(question)}, answer: {bool(answer)}")
                return {"success": False, "conversation_id": None, "error": "Eksik veri"}

            if not question.strip() or not answer.strip():
                print(f"❌ log_conversation_with_id: Boş veri - question length: {len(question.strip())}, answer length: {len(answer.strip())}")
                return {"success": False, "conversation_id": None, "error": "Boş veri"}

            # Thread ID yoksa yeni thread oluştur
            if not thread_id:
                thread_result = self.supabase.rpc("create_new_thread").execute()
                thread_id = thread_result.data
                message_order = 1
                print(f"🆕 Yeni thread oluşturuldu: {thread_id}")
            else:
                # Mevcut thread'e ekle - sıradaki mesaj numarasını al
                order_result = self.supabase.rpc("get_next_message_order", {"target_thread_id": thread_id}).execute()
                message_order = order_result.data
                print(f"📝 Mevcut thread'e ekleniyor: {thread_id}, Order: {message_order}")

            data = {
                "question": question.strip(),
                "answer": answer.strip(),
                "model_name": model_name or "unknown",
                "confidence": confidence if confidence is not None else 0.0,
                "sources": sources if sources is not None else [],
                "response_time": response_time,
                "user_id": user_id,
                "thread_id": thread_id,
                "parent_message_id": parent_message_id,
                "message_order": message_order,
                "created_at": datetime.utcnow().isoformat()
            }

            print(f"🔍 log_conversation_with_id: Thread {thread_id}, Order {message_order}, User: {user_id}")

            result = self.supabase.table("conversations").insert(data).execute()

            if result.data and len(result.data) > 0:
                conversation_id = result.data[0].get('id')
                print(f"✅ log_conversation_with_id: Başarıyla kaydedildi - ID: {conversation_id}, Thread: {thread_id}")
                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "thread_id": thread_id,
                    "message_order": message_order
                }
            else:
                print(f"❌ log_conversation_with_id: Kayıt başarısız - result.data boş")
                return {"success": False, "conversation_id": None, "error": "Kayıt başarısız"}

        except Exception as e:
            print(f"❌ log_conversation_with_id: Supabase kayıt hatası: {e}")
            return {"success": False, "conversation_id": None, "error": str(e)}

    def get_conversation_threads(self, user_id: str, limit: int = 50):
        """Kullanıcının thread'lerini getir (deleted olanlar hariç) - View uyumlu"""
        try:
            # conversation_threads view'ini kullan ama deleted_by_user filtresini conversations tablosu üzerinden yap
            result = self.supabase.table("conversation_threads").select("*").eq("user_id", user_id).order("last_updated_at", desc=True).limit(limit).execute()

            if not result.data:
                print(f"📊 get_conversation_threads: 0 thread bulundu (user_id: {user_id})")
                return []

            # Her thread için ilk mesajının deleted_by_user durumunu kontrol et
            filtered_threads = []
            for thread in result.data:
                thread_id = thread.get("thread_id")
                if thread_id:
                    # Thread'in deleted durumunu kontrol et
                    check_result = self.supabase.table("conversations").select("deleted_by_user").eq("thread_id", thread_id).eq("user_id", user_id).limit(1).execute()

                    if check_result.data and not check_result.data[0].get("deleted_by_user", False):
                        filtered_threads.append(thread)

            print(f"📊 get_conversation_threads: {len(filtered_threads)} thread bulundu (user_id: {user_id})")
            return filtered_threads
        except Exception as e:
            print(f"Thread listesi getirme hatası: {e}")
            return []

    def soft_delete_thread(self, thread_id: str, user_id: str) -> bool:
        """Thread'i soft delete yap (sadece conversations tablosunda)"""
        try:
            # Thread'deki tüm mesajları soft delete
            result = self.supabase.table("conversations").update({
                "deleted_by_user": True
            }).eq("thread_id", thread_id).eq("user_id", user_id).execute()

            print(f"✅ Thread soft deleted: {thread_id}")
            return True
        except Exception as e:
            print(f"❌ Soft delete hatası: {e}")
            return False

    def get_thread_messages(self, thread_id: str, user_id: str) -> list:
        """Belirli bir thread'in tüm mesajlarını getir (deleted olanlar hariç)"""
        try:
            # Thread'in deleted durumunu kontrol et
            check_result = self.supabase.table("conversations").select("deleted_by_user").eq("thread_id", thread_id).eq("user_id", user_id).limit(1).execute()

            if check_result.data and check_result.data[0].get("deleted_by_user", False):
                print(f"⚠️ Deleted thread access attempt: {thread_id} by {user_id}")
                return []

            result = self.supabase.rpc("get_thread_messages", {"target_thread_id": thread_id}).execute()

            # Güvenlik için user_id kontrolü
            if result.data:
                # İlk mesajın user_id'sini kontrol et
                first_message = result.data[0] if result.data else None
                if first_message and first_message.get('user_id') != user_id:
                    print(f"⚠️ Unauthorized thread access attempt: {thread_id} by {user_id}")
                    return []

                # Deleted mesajları filtrele
                filtered_messages = [msg for msg in result.data if not msg.get('deleted_by_user', False)]
                print(f"📊 get_thread_messages: {len(filtered_messages)} mesaj bulundu (thread: {thread_id})")
                return filtered_messages

            print(f"📊 get_thread_messages: 0 mesaj bulundu (thread: {thread_id})")
            return []
        except Exception as e:
            print(f"Thread mesajları getirme hatası: {e}")
            return []

    def restore_thread(self, thread_id: str, user_id: str) -> bool:
        """Soft deleted thread'i geri getir"""
        try:
            # Thread'deki tüm mesajları restore et
            result = self.supabase.table("conversations").update({
                "deleted_by_user": False
            }).eq("thread_id", thread_id).eq("user_id", user_id).execute()

            print(f"✅ Thread restored: {thread_id}")
            return True
        except Exception as e:
            print(f"❌ Restore hatası: {e}")
            return False

    def get_conversation_history(self, user_id: str, limit: int = 50):
        """Kullanıcının geçmiş sohbetlerini getir (deleted olanlar hariç)"""
        try:
            result = self.supabase.table("conversations").select("*").eq("user_id", user_id).eq("deleted_by_user", False).order("created_at", desc=True).limit(limit).execute()

            print(f"📊 get_conversation_history: {len(result.data)} sohbet bulundu (user_id: {user_id})")
            return result.data
        except Exception as e:
            print(f"Conversation history getirme hatası: {e}")
            return []

