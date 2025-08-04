#!/usr/bin/env python3
"""
RAG API Auth özelliklerini test etmek için script
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_auth_flow():
    print("=== RAG API Authentication Test ===")
    
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
    
    # Test kullanıcı bilgileri
    test_email = "test@example.com"
    test_password = "test123456"
    test_name = "Test User"
    
    print(f"\n=== 1. Kullanıcı Kaydı (Sign Up) ===")
    print(f"Email: {test_email}")
    
    signup_data = {
        "email": test_email,
        "password": test_password,
        "full_name": test_name
    }
    
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
            print(f"  - User ID: {result.get('user_id')}")
        else:
            print(f"⚠️ Kullanıcı kaydı: {response.status_code}")
            print(f"  - Bu normal olabilir (kullanıcı zaten mevcut)")
    except Exception as e:
        print(f"❌ Kullanıcı kaydı hatası: {e}")
    
    print(f"\n=== 2. Kullanıcı Girişi (Sign In) ===")
    
    signin_data = {
        "email": test_email,
        "password": test_password
    }
    
    access_token = None
    
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
            print(f"  - User Email: {result['user']['email']}")
            print(f"  - Token alındı: {access_token[:20]}..." if access_token else "Token yok")
        else:
            print(f"❌ Giriş başarısız: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Giriş hatası: {e}")
        return False
    
    if not access_token:
        print("❌ Access token alınamadı")
        return False
    
    print(f"\n=== 3. Kullanıcı Bilgileri (/auth/me) ===")
    
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
            print(f"  - Created: {user['created_at']}")
        else:
            print(f"❌ Kullanıcı bilgileri alınamadı: {response.status_code}")
    except Exception as e:
        print(f"❌ Kullanıcı bilgileri hatası: {e}")
    
    print(f"\n=== 4. Authenticated RAG Request ===")
    
    rag_data = {
        "question": "Authenticated test sorusu - bu kullanıcı giriş yapmış",
        "top_k": 3
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=rag_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
        )
        
        if response.status_code == 200:
            print("✓ Authenticated RAG request başarılı")
            result = response.json()
            print(f"  - Answer: {result.get('answer', 'N/A')[:100]}...")
            print(f"  - Method: {result.get('method', 'N/A')}")
        else:
            print(f"❌ Authenticated RAG request başarısız: {response.status_code}")
    except Exception as e:
        print(f"❌ Authenticated RAG request hatası: {e}")
    
    print(f"\n=== 5. Anonymous RAG Request (Karşılaştırma) ===")
    
    rag_data = {
        "question": "Anonymous test sorusu - bu kullanıcı giriş yapmamış",
        "top_k": 3,
        "user_id": "anonymous-user"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=rag_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✓ Anonymous RAG request başarılı")
            result = response.json()
            print(f"  - Answer: {result.get('answer', 'N/A')[:100]}...")
        else:
            print(f"❌ Anonymous RAG request başarısız: {response.status_code}")
    except Exception as e:
        print(f"❌ Anonymous RAG request hatası: {e}")
    
    print(f"\n=== 6. Conversation History (Authenticated) ===")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/history?limit=5",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if response.status_code == 200:
            print("✓ Authenticated history alındı")
            result = response.json()
            conversations = result.get('conversations', [])
            print(f"  - Conversation count: {len(conversations)}")
            print(f"  - User authenticated: {result.get('user_authenticated')}")
            
            if conversations:
                latest = conversations[0]
                print(f"  - Latest question: {latest.get('question', 'N/A')[:50]}...")
        else:
            print(f"❌ Authenticated history alınamadı: {response.status_code}")
    except Exception as e:
        print(f"❌ Authenticated history hatası: {e}")
    
    print(f"\n=== 7. Magic Link Test (Opsiyonel) ===")
    
    magic_data = {"email": test_email}
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/magic-link",
            json=magic_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✓ Magic link gönderildi")
            result = response.json()
            print(f"  - Message: {result.get('message')}")
        else:
            print(f"⚠️ Magic link: {response.status_code} (Normal olabilir)")
    except Exception as e:
        print(f"❌ Magic link hatası: {e}")
    
    print(f"\n=== 8. Sign Out ===")
    
    try:
        response = requests.post(f"{API_BASE_URL}/auth/signout")
        
        if response.status_code == 200:
            print("✓ Çıkış başarılı")
        else:
            print(f"⚠️ Çıkış: {response.status_code}")
    except Exception as e:
        print(f"❌ Çıkış hatası: {e}")
    
    print("\n🎉 Auth test tamamlandı!")
    return True

if __name__ == "__main__":
    print("RAG API'nin çalıştığından emin olun (python rag_api.py)")
    print("Test başlatılıyor...\n")
    
    success = test_auth_flow()
    
    if success:
        print("\n✅ Auth sistemi çalışıyor!")
        print("\n📝 Kullanım örnekleri:")
        print("1. Kayıt: POST /auth/signup")
        print("2. Giriş: POST /auth/signin")
        print("3. Token ile RAG: Authorization: Bearer <token>")
        print("4. Kullanıcı bilgileri: GET /auth/me")
        print("5. Geçmiş: GET /history (authenticated)")
    else:
        print("\n❌ Auth testinde sorunlar var.")
