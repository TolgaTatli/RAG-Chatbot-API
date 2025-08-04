#!/usr/bin/env python3
"""
Supabase Auth Email Format Test
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
import time

def test_different_email_formats():
    print("=== Email Format Test ===")
    
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Farklı email formatları test et
    test_emails = [
        "test@gmail.com",
        "user@hotmail.com", 
        "testuser@outlook.com",
        "demo@yahoo.com",
        "real.email@protonmail.com"
    ]
    
    for email in test_emails:
        print(f"\n--- Test: {email} ---")
        
        try:
            response = supabase.auth.sign_up({
                "email": email,
                "password": "testpass123456",
                "options": {
                    "data": {"full_name": "Test User"}
                }
            })
            
            print(f"✓ {email} - Kayıt başarılı!")
            
            if response.user:
                print(f"  User ID: {response.user.id}")
                print(f"  Email confirmed: {response.user.email_confirmed_at is not None}")
                
                if response.session:
                    print(f"  Session var: Evet")
                    return email, response.session.access_token
                else:
                    print(f"  Session var: Hayır (email confirmation gerekli olabilir)")
            
            # Başarılı olduysa break
            break
            
        except Exception as e:
            print(f"❌ {email} - Hata: {e}")
            
            # Eğer kullanıcı zaten varsa, giriş yapmayı dene
            if "already registered" in str(e).lower() or "already exists" in str(e).lower():
                print(f"  🔄 {email} zaten kayıtlı, giriş deneniyor...")
                
                try:
                    login_response = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": "testpass123456"
                    })
                    
                    print(f"  ✓ Giriş başarılı!")
                    
                    if login_response.session:
                        return email, login_response.session.access_token
                        
                except Exception as login_error:
                    print(f"  ❌ Giriş de başarısız: {login_error}")
            
            time.sleep(1)  # Rate limiting için bekle
    
    return None, None

def test_simple_auth_flow():
    print("\n=== Basit Auth Flow Test ===")
    
    email, token = test_different_email_formats()
    
    if not email or not token:
        print("❌ Hiçbir email formatı çalışmadı!")
        return False
    
    print(f"\n✅ Başarılı email: {email}")
    print(f"✅ Token: {token[:20]}...")
    
    # Token ile bir API çağrısı dene
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    try:
        # Token'ı set et ve user bilgilerini al
        supabase.auth.set_session(token, token)  # access_token, refresh_token
        user = supabase.auth.get_user()
        
        if user and user.user:
            print(f"✅ User info alındı:")
            print(f"  ID: {user.user.id}")
            print(f"  Email: {user.user.email}")
            print(f"  Created: {user.user.created_at}")
            return True
        else:
            print("❌ User info alınamadı")
            return False
            
    except Exception as e:
        print(f"❌ Token test hatası: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_auth_flow()
    
    if success:
        print(f"\n🎉 Auth sistem çalışıyor!")
        print(f"Artık normal test scriptini çalıştırabilirsiniz.")
    else:
        print(f"\n❌ Auth sisteminde hala sorun var.")
        print(f"\n🔧 Supabase Dashboard kontrolleri:")
        print(f"1. Authentication > Settings > Enable email confirmations: OFF")
        print(f"2. Authentication > Providers > Email: Enabled")
        print(f"3. Authentication > URL Configuration > Site URL: http://localhost:8000")
        print(f"4. Proje aktif ve billing sorunu yok mu?")
