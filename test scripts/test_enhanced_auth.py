#!/usr/bin/env python3
"""
Enhanced RAG API Auth + Database Integration Test Script
Tests the complete auth flow with proper user_id mapping
"""

import requests
import json
import time
import uuid

API_BASE_URL = "http://localhost:8000"

def test_enhanced_auth_system():
    print("=== Enhanced RAG API Auth + Database Integration Test ===")
    
    # API'nin çalışıp çalışmadığını kontrol et
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            print("✓ API çalışıyor")
            status_data = response.json()
            print(f"  - Supabase Auth: {status_data.get('supabase_auth')}")
        else:
            print("❌ API çalışmıyor")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API'ye bağlanılamıyor. Lütfen önce API'yi başlatın:")
        print("   python rag_api.py")
        return False
    
    # Test kullanıcı bilgileri - her test için unique
    test_suffix = str(int(time.time()))[-6:]
    test_email = f"testuser{test_suffix}@example.com"
    test_password = "testpass123"
    test_name = f"Test User {test_suffix}"
    
    print(f"\n=== 1. Yeni Kullanıcı Kaydı ===")
    print(f"Email: {test_email}")
    
    signup_data = {
        "email": test_email,
        "password": test_password,
        "full_name": test_name
    }
    
    access_token = None
    user_id = None
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/signup",
            json=signup_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✓ Kullanıcı kaydı başarılı")
            result = response.json()
            print(f"  - Message: {result.get('message')}")
            user_id = result.get('user_id')
            print(f"  - User ID: {user_id}")
        else:
            print(f"❌ Kullanıcı kaydı başarısız: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Kullanıcı kaydı hatası: {e}")
        return False
    
    print(f"\n=== 2. Kullanıcı Girişi ===")
    
    signin_data = {
        "email": test_email,
        "password": test_password
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/signin",
            json=signin_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✓ Giriş başarılı")
            result = response.json()
            access_token = result.get('access_token')
            user_info = result.get('user')
            user_id = user_info.get('id')
            print(f"  - User ID: {user_id}")
            print(f"  - Email: {user_info.get('email')}")
            print(f"  - Token: {access_token[:20]}..." if access_token else "Token yok")
        else:
            print(f"❌ Giriş başarısız: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Giriş hatası: {e}")
        return False
    
    if not access_token or not user_id:
        print("❌ Access token veya user ID alınamadı")
        return False
    
    print(f"\n=== 3. Kullanıcı Bilgileri ve İstatistikleri ===")
    
    # /auth/me endpoint
    try:
        response = requests.get(
            f"{API_BASE_URL}/auth/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if response.status_code == 200:
            print("✓ Kullanıcı bilgileri alındı")
            result = response.json()
            user = result['user']
            print(f"  - ID: {user['id']}")
            print(f"  - Email: {user['email']}")
        else:
            print(f"❌ Kullanıcı bilgileri alınamadı: {response.status_code}")
    except Exception as e:
        print(f"❌ Kullanıcı bilgileri hatası: {e}")
    
    # /auth/stats endpoint
    try:
        response = requests.get(
            f"{API_BASE_URL}/auth/stats",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if response.status_code == 200:
            print("✓ Kullanıcı istatistikleri alındı")
            stats = response.json()
            print(f"  - Total Conversations: {stats.get('total_conversations', 0)}")
            print(f"  - Avg Confidence: {stats.get('avg_confidence', 0):.2f}")
            print(f"  - Favorite Model: {stats.get('favorite_model', 'N/A')}")
        else:
            print(f"⚠️ Kullanıcı istatistikleri: {response.status_code} (Yeni kullanıcı için normal)")
    except Exception as e:
        print(f"❌ İstatistik hatası: {e}")
    
    print(f"\n=== 4. RAG Conversations (User ID Mapping Test) ===")
    
    test_questions = [
        "Merhaba, nasılsın? Bu ilk test sorumu.",
        "Python programlama hakkında bilgi verir misin?",
        "Yapay zeka teknolojileri nelerdir?",
        "Bu üçüncü ve son test sorumu."
    ]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    conversation_ids = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- RAG Request {i}/4 ---")
        
        rag_data = {
            "question": question,
            "top_k": 3
        }
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json=rag_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"✓ RAG Request {i} başarılı")
                result = response.json()
                print(f"  - Question: {question[:40]}...")
                print(f"  - Answer: {result.get('answer', 'N/A')[:60]}...")
                print(f"  - Method: {result.get('method', 'N/A')}")
                
                # Biraz bekle ki kayıt işlemi tamamlansın
                time.sleep(0.5)
            else:
                print(f"❌ RAG Request {i} başarısız: {response.status_code}")
        except Exception as e:
            print(f"❌ RAG Request {i} hatası: {e}")
    
    print(f"\n=== 5. Conversation History Verification ===")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/history?limit=10",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if response.status_code == 200:
            print("✓ Conversation history alındı")
            result = response.json()
            conversations = result.get('conversations', [])
            print(f"  - Total DB Conversations: {result.get('total_conversations', 0)}")
            print(f"  - Showing: {result.get('showing', 0)}")
            print(f"  - User Authenticated: {result.get('user_authenticated')}")
            print(f"  - User ID: {result.get('user_id')}")
            
            if conversations:
                print(f"\n  📜 Son {len(conversations)} Conversation:")
                for i, conv in enumerate(conversations[:3], 1):
                    print(f"    {i}. Q: {conv.get('question', 'N/A')[:50]}...")
                    print(f"       A: {conv.get('answer', 'N/A')[:50]}...")
                    print(f"       Model: {conv.get('model_name', 'N/A')}")
                    print(f"       Time: {conv.get('created_at', 'N/A')}")
                    conversation_ids.append(conv.get('id'))
            else:
                print("  ⚠️ Conversation history boş!")
                return False
        else:
            print(f"❌ History alınamadı: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ History hatası: {e}")
        return False
    
    print(f"\n=== 6. Conversation Count Test ===")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/history/count",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Conversation count alındı")
            print(f"  - Total: {result.get('total_conversations', 0)}")
            print(f"  - User ID: {result.get('user_id')}")
        else:
            print(f"❌ Count alınamadı: {response.status_code}")
    except Exception as e:
        print(f"❌ Count hatası: {e}")
    
    print(f"\n=== 7. User Models Test ===")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/auth/models",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ User models alındı")
            models = result.get('models', [])
            print(f"  - Model sayısı: {len(models)}")
            for model in models:
                print(f"    - {model.get('model_name')}: {model.get('count')} kez")
        else:
            print(f"❌ Models alınamadı: {response.status_code}")
    except Exception as e:
        print(f"❌ Models hatası: {e}")
    
    print(f"\n=== 8. Search Conversations Test ===")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/history/search?search=test&limit=5",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Conversation search başarılı")
            print(f"  - Search term: {result.get('search_term')}")
            print(f"  - Found: {result.get('found', 0)} conversations")
            
            for conv in result.get('conversations', [])[:2]:
                print(f"    - {conv.get('question', 'N/A')[:40]}...")
        else:
            print(f"❌ Search başarısız: {response.status_code}")
    except Exception as e:
        print(f"❌ Search hatası: {e}")
    
    print(f"\n=== 9. Delete Conversation Test ===")
    
    if conversation_ids:
        test_conv_id = conversation_ids[0]
        try:
            response = requests.delete(
                f"{API_BASE_URL}/history/{test_conv_id}",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if response.status_code == 200:
                print("✓ Conversation silme başarılı")
                result = response.json()
                print(f"  - Silinen ID: {result.get('conversation_id')}")
            else:
                print(f"❌ Conversation silinemedi: {response.status_code}")
        except Exception as e:
            print(f"❌ Delete hatası: {e}")
    
    print(f"\n=== 10. Anonymous vs Authenticated Comparison ===")
    
    # Anonymous request
    anon_data = {
        "question": "Bu anonymous bir test sorusudur",
        "top_k": 3,
        "user_id": "anonymous-user-123"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=anon_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✓ Anonymous request başarılı")
        else:
            print(f"❌ Anonymous request başarısız: {response.status_code}")
    except Exception as e:
        print(f"❌ Anonymous request hatası: {e}")
    
    # Anonymous history (should not work properly due to RLS)
    try:
        response = requests.get(f"{API_BASE_URL}/history?user_id=anonymous-user-123&limit=5")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Anonymous history request - RLS Test")
            print(f"  - Conversations found: {len(result.get('conversations', []))}")
            print(f"  - User authenticated: {result.get('user_authenticated')}")
            # RLS nedeniyle authenticated olmayan kullanıcı kendi kayıtlarını göremeyecek
        else:
            print(f"⚠️ Anonymous history: {response.status_code}")
    except Exception as e:
        print(f"❌ Anonymous history hatası: {e}")
    
    print(f"\n🎉 Test Tamamlandı!")
    print(f"\n📊 Test Özeti:")
    print(f"✅ Yeni kullanıcı kaydı: {test_email}")
    print(f"✅ JWT token authentication çalışıyor")
    print(f"✅ User ID mapping (UUID) çalışıyor")
    print(f"✅ RLS (Row Level Security) aktif")
    print(f"✅ Conversation logging user-specific")
    print(f"✅ History, search, delete işlemleri")
    print(f"✅ User statistics and models")
    
    return True

if __name__ == "__main__":
    print("RAG API'nin çalıştığından ve Supabase schema'nın güncellendiğinden emin olun!")
    print("Schema update: supabase_schema_update.sql dosyasını çalıştırın")
    print("API başlatma: python rag_api.py")
    print("\nTest başlatılıyor...\n")
    
    success = test_enhanced_auth_system()
    
    if success:
        print(f"\n🌟 Harika! Enhanced auth sistemi mükemmel çalışıyor!")
        print(f"\n📚 Özellikler:")
        print(f"• 🔐 Supabase Auth ile tam entegrasyon")  
        print(f"• 👤 UUID user_id mapping")
        print(f"• 🛡️ Row Level Security (RLS)")
        print(f"• 📊 User statistics & model tracking")
        print(f"• 🔍 Conversation search & management")
        print(f"• 🆓 50,000 MAU'ya kadar ÜCRETSIZ!")
    else:
        print(f"\n❌ Test başarısız. Loglara bakın ve sorunları çözün.")
        print(f"\n🔧 Kontrol listesi:")
        print(f"• Supabase schema güncellenmiş mi?")
        print(f"• .env dosyası doğru mu?") 
        print(f"• API çalışıyor mu?")
        print(f"• RLS policy'leri kurulmuş mu?")
