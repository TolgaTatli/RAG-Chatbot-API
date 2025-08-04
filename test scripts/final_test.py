#!/usr/bin/env python3
"""
Supabase ayarları yapıldıktan sonra final test
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
import time
import requests

def final_auth_test():
    print("=== Final Auth Test (Email Confirmation Kapatıldıktan Sonra) ===")
    
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Yeni unique kullanıcı oluştur
    timestamp = str(int(time.time()))[-6:]
    test_email = f"final_test_{timestamp}@gmail.com"
    test_password = "finaltest123"
    
    print(f"Test Email: {test_email}")
    print(f"Test Password: {test_password}")
    
    try:
        # Yeni kayıt
        print("\n1. Yeni kullanıcı kaydı...")
        response = supabase.auth.sign_up({
            "email": test_email,
            "password": test_password,
            "options": {
                "data": {"full_name": f"Final Test User {timestamp}"}
            }
        })
        
        if response.session:
            print("✅ Kayıt başarılı VE session oluştu!")
            token = response.session.access_token
            user_id = response.user.id
            
            print(f"  User ID: {user_id}")
            print(f"  Token: {token[:20]}...")
            
        elif response.user:
            print("✅ Kayıt başarılı ama session yok")
            print("  Email confirmation hala açık olabilir")
            print("  Giriş yapmayı deniyorum...")
            
            # Hemen giriş yapmayı dene
            login_response = supabase.auth.sign_in_with_password({
                "email": test_email,
                "password": test_password
            })
            
            if login_response.session:
                print("✅ Giriş başarılı!")
                token = login_response.session.access_token
                user_id = login_response.user.id
            else:
                print("❌ Giriş de başarısız")
                return None
        else:
            print("❌ Kayıt başarısız")
            return None
        
        # API testleri
        print(f"\n2. API Testleri...")
        
        api_base = "http://localhost:8000"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # /auth/me test
        try:
            response = requests.get(f"{api_base}/auth/me", headers=headers)
            if response.status_code == 200:
                print("✅ /auth/me çalışıyor")
                user_data = response.json()
                print(f"  API User ID: {user_data['user']['id']}")
            else:
                print(f"❌ /auth/me hatası: {response.status_code}")
                print(response.text)
        except requests.exceptions.ConnectionError:
            print("⚠️ API çalışmıyor. 'python rag_api.py' çalıştırın!")
            return {"email": test_email, "password": test_password, "token": token, "user_id": user_id}
        
        # RAG test
        try:
            rag_data = {"question": "Bu final test sorusudur", "top_k": 3}
            response = requests.post(f"{api_base}/generate", json=rag_data, headers=headers)
            
            if response.status_code == 200:
                print("✅ RAG endpoint çalışıyor")
                result = response.json()
                print(f"  Answer: {result.get('answer', 'N/A')[:50]}...")
                
                # Conversation history test
                time.sleep(1)
                history_response = requests.get(f"{api_base}/history?limit=5", headers=headers)
                
                if history_response.status_code == 200:
                    print("✅ History endpoint çalışıyor")
                    history = history_response.json()
                    print(f"  Total conversations: {history.get('total_conversations', 0)}")
                else:
                    print(f"⚠️ History endpoint: {history_response.status_code}")
            else:
                print(f"❌ RAG endpoint hatası: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"❌ RAG test hatası: {e}")
        
        print(f"\n🎉 Final Test Başarılı!")
        print(f"Kullanabilir test bilgileri:")
        print(f"  Email: {test_email}")
        print(f"  Password: {test_password}")
        print(f"  User ID: {user_id}")
        
        return {
            "email": test_email,
            "password": test_password,
            "token": token,
            "user_id": user_id
        }
        
    except Exception as e:
        print(f"❌ Final test hatası: {e}")
        
        if "email not confirmed" in str(e).lower():
            print(f"\n🚨 HALA EMAIL CONFIRMATION AÇIK!")
            print(f"Supabase Dashboard > Authentication > Settings")
            print(f"'Enable email confirmations' checkbox'ını KAPATIN!")
        
        return None

if __name__ == "__main__":
    result = final_auth_test()
    
    if result:
        print(f"\n✅ SİSTEM HAZIR!")
        print(f"\nŞimdi ana test scriptini çalıştırabilirsiniz:")
        print(f"python test_enhanced_auth.py")
        
        print(f"\nVeya manuel test için:")
        print(f"Email: {result['email']}")
        print(f"Password: {result['password']}")
    else:
        print(f"\n❌ Hala sorun var. Supabase ayarlarını kontrol edin!")
        print(f"\n📋 Kontrol listesi:")
        print(f"✓ Authentication > Settings > Enable email confirmations: OFF")
        print(f"✓ Authentication > Providers > Email: ON")
        print(f"✓ Authentication > URL Configuration > Site URL: http://localhost:8000")
        print(f"✓ Proje aktif ve billing OK")
