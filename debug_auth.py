#!/usr/bin/env python3


import os
from dotenv import load_dotenv
from supabase import create_client, Client

def debug_supabase_auth():
    print("DEbug")
    
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    print(f"SUPABASE_URL: {'✓' if supabase_url else '✗ EKSIK!'}")
    print(f"SUPABASE_ANON_KEY: {'✓' if supabase_key else '✗ EKSIK!'}")
    
    if not supabase_url or not supabase_key:
        print("\n❌ Environment variables eksik!")
        return False
    
    print(f"URL: {supabase_url}")
    print(f"Key: {supabase_key[:20]}...{supabase_key[-10:]}")
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("Supabase client oluşturuldu")
    except Exception as e:
        print(f"Supabase client hatası: {e}")
        return False
    
    test_email = "debug-test@example.com"
    test_password = "testpass123"
    
    print(f"\n=== Auth Test ===")
    print(f"Test Email: {test_email}")
    
    try:
        print("Eski test kullanıcısını temizlemeye çalışıyor...")
        try:
            pass
        except:
            pass
        
        print("Yeni kullanıcı kaydı deneniyor...")
        response = supabase.auth.sign_up({
            "email": test_email,
            "password": test_password,
            "options": {
                "data": {"full_name": "Debug Test User"}
            }
        })
        
        print("Kayıt başarılı!")
        print(f"User ID: {response.user.id if response.user else 'None'}")
        print(f"Email confirmed: {response.user.email_confirmed_at is not None if response.user else 'Unknown'}")
        print(f"Session: {'Var' if response.session else 'Yok'}")
        
        if response.user and not response.user.email_confirmed_at:
            print("⚠Email confirmation gerekli!")
            print("Supabase Dashboard > Authentication > Settings > 'Enable email confirmations' kontrolü yapın")
        
        return True
        
    except Exception as e:
        print(f"Auth kayıt hatası: {e}")
        print(f"Hata tipi: {type(e).__name__}")
        
        error_str = str(e).lower()
        
        if "email" in error_str and "confirm" in error_str:
            print("\nÇözüm: Email confirmation ayarları")
            print("1. Supabase Dashboard > Authentication > Settings")
            print("2. 'Enable email confirmations' kapatın veya SMTP ayarlayın")
            
        elif "invalid" in error_str or "credentials" in error_str:
            print("\nÇözüm: API credentials kontrol")
            print("1. SUPABASE_URL doğru mu?")
            print("2. SUPABASE_ANON_KEY doğru mu?")
            
        elif "rate" in error_str or "limit" in error_str:
            print("\nÇözüm: Rate limiting")
            print("1. Çok fazla deneme yapılmış olabilir")
            print("2. Birkaç dakika bekleyin")
            
        elif "domain" in error_str or "url" in error_str:
            print("\nÇözüm: Domain ayarları")
            print("1. Supabase Dashboard > Authentication > URL Configuration")
            print("2. Site URL: http://localhost:8000 ekleyin")
            
        else:
            print(f"\n🔧 Genel çözüm önerileri:")
            print("1. Supabase proje aktif mi?")
            print("2. API keys doğru mu?")
            print("3. Auth settings doğru mu?")
        
        return False

def check_supabase_auth_settings():
    print("\n=== Supabase Auth Settings Kontrol Listesi ===")
    print("Supabase Dashboard'da kontrol edilmesi gerekenler:")
    print("")
    print("Authentication > Settings:")
    print("   • Enable email confirmations: KAPALI olmalı (veya SMTP ayarlı)")
    print("   • Enable phone confirmations: İhtiyaca göre")
    print("   • JWT expiry: 3600 (1 saat)")
    print("")
    print("Authentication > URL Configuration:")
    print("   • Site URL: http://localhost:8000")
    print("   • Additional Redirect URLs: http://localhost:3000, http://localhost:8000")
    print("")
    print("Settings > API:")
    print("   • Project URL: .env dosyasındaki SUPABASE_URL ile aynı")
    print("   • anon public key: .env dosyasındaki SUPABASE_ANON_KEY ile aynı")
    print("")
    print("Authentication > Providers:")
    print("   • Email: Enabled")
    print("   • Diğer provider'lar ihtiyaca göre")

if __name__ == "__main__":
    success = debug_supabase_auth()
    
    if not success:
        check_supabase_auth_settings()
        
    print(f"\n{'Debug tamamlandı!' if success else '❌ Sorunlar var, yukarıdaki çözümleri deneyin.'}")
