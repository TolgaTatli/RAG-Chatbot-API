#!/usr/bin/env python3
"""
Email confirmation olmadan auth test
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
import time

def test_auth_with_different_approach():
    print("=== Email Confirmation Bypass Test ===")
    
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Mevcut bir kullanıcı ile giriş yapmayı dene
    test_email = "user@hotmail.com"  # Önceki testte başarılı olan
    test_password = "testpass123456"
    
    print(f"Test Email: {test_email}")
    
    try:
        # Giriş yapmayı dene
        print("Giriş deneniyor...")
        response = supabase.auth.sign_in_with_password({
            "email": test_email,
            "password": test_password
        })
        
        if response.session:
            print("✅ Giriş başarılı!")
            print(f"Access Token: {response.session.access_token[:20]}...")
            print(f"User ID: {response.user.id}")
            print(f"Email: {response.user.email}")
            
            return {
                "success": True,
                "email": test_email,
                "password": test_password,
                "token": response.session.access_token,
                "user_id": response.user.id
            }
        else:
            print("❌ Session oluşmadı")
            
    except Exception as e:
        print(f"Giriş hatası: {e}")
        
        # Eğer kullanıcı bulunamadıysa, yeni kayıt dene
        if "invalid" in str(e).lower() or "not found" in str(e).lower():
            print("Kullanıcı bulunamadı, yeni kayıt deneniyor...")
            
            # Yeni unique email
            timestamp = str(int(time.time()))[-6:]
            new_email = f"testuser{timestamp}@gmail.com"
            
            try:
                signup_response = supabase.auth.sign_up({
                    "email": new_email,
                    "password": test_password,
                    "options": {
                        "data": {"full_name": f"Test User {timestamp}"}
                    }
                })
                
                print(f"✅ Yeni kayıt başarılı: {new_email}")
                
                if signup_response.session:
                    print("✅ Session da oluştu!")
                    return {
                        "success": True,
                        "email": new_email,
                        "password": test_password,
                        "token": signup_response.session.access_token,
                        "user_id": signup_response.user.id
                    }
                else:
                    print("⚠️ Kayıt başarılı ama session yok (email confirmation gerekli)")
                    print("Supabase Dashboard > Authentication > Settings > 'Enable email confirmations' KAPATIN")
                    
            except Exception as signup_error:
                print(f"❌ Yeni kayıt da başarısız: {signup_error}")
    
    return {"success": False}

def test_api_with_working_auth(auth_data):
    """Çalışan auth bilgileri ile API test"""
    if not auth_data["success"]:
        return False
    
    print(f"\n=== API Test with Working Auth ===")
    
    import requests
    
    # API'ye test isteği gönder
    api_base = "http://localhost:8000"
    
    try:
        # Auth me endpoint test
        response = requests.get(
            f"{api_base}/auth/me",
            headers={"Authorization": f"Bearer {auth_data['token']}"}
        )
        
        if response.status_code == 200:
            print("✅ /auth/me endpoint çalışıyor")
            result = response.json()
            print(f"  User ID: {result['user']['id']}")
            print(f"  Email: {result['user']['email']}")
        else:
            print(f"❌ /auth/me endpoint hatası: {response.status_code}")
            print(response.text)
            return False
        
        # RAG endpoint test
        rag_data = {
            "question": "Bu bir test sorusudur",
            "top_k": 3
        }
        
        response = requests.post(
            f"{api_base}/generate",
            json=rag_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {auth_data['token']}"
            }
        )
        
        if response.status_code == 200:
            print("✅ RAG endpoint çalışıyor")
            result = response.json()
            print(f"  Answer: {result.get('answer', 'N/A')[:50]}...")
        else:
            print(f"❌ RAG endpoint hatası: {response.status_code}")
            print(response.text)
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ API'ye bağlanılamıyor. python rag_api.py çalıştırıldığından emin olun!")
        return False
    except Exception as e:
        print(f"❌ API test hatası: {e}")
        return False

if __name__ == "__main__":
    # Auth test
    auth_result = test_auth_with_different_approach()
    
    if auth_result["success"]:
        print(f"\n🎉 Auth çalışıyor!")
        print(f"Email: {auth_result['email']}")
        print(f"Password: {auth_result['password']}")
        print(f"User ID: {auth_result['user_id']}")
        
        # API test
        api_success = test_api_with_working_auth(auth_result)
        
        if api_success:
            print(f"\n✅ TÜM SİSTEM ÇALIŞIYOR!")
            print(f"\nTest için kullanabileceğiniz bilgiler:")
            print(f"Email: {auth_result['email']}")
            print(f"Password: {auth_result['password']}")
        else:
            print(f"\n⚠️ Auth çalışıyor ama API'de sorun var")
    else:
        print(f"\n❌ Auth sisteminde sorun var")
        print(f"\n🚨 ÖNEMLİ: Supabase Dashboard ayarları:")
        print(f"1. Authentication > Settings")
        print(f"2. 'Enable email confirmations' seçeneğini KAPATIN")
        print(f"3. Ayarları kaydedin")
        print(f"4. Bu scripti tekrar çalıştırın")
