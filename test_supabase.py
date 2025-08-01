#!/usr/bin/env python3
"""
Supabase bağlantısını ve log_conversation fonksiyonunu test etmek için script
"""

import os
from dotenv import load_dotenv
from supabase_client import SupabaseLogger

def test_supabase_connection():
    print("=== Supabase Bağlantı Testi ===")
    
    # .env dosyasını yükle
    load_dotenv()
    
    # Environment variables kontrolü
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    print(f"SUPABASE_URL: {'✓ Var' if supabase_url else '✗ YOK!'}")
    print(f"SUPABASE_ANON_KEY: {'✓ Var' if supabase_key else '✗ YOK!'}")
    
    if not supabase_url or not supabase_key:
        print("\n❌ Supabase environment variables eksik!")
        print("Lütfen .env dosyasında SUPABASE_URL ve SUPABASE_ANON_KEY değerlerini kontrol edin.")
        return False
    
    # SupabaseLogger oluştur
    try:
        logger = SupabaseLogger()
        print("✓ SupabaseLogger başarıyla oluşturuldu")
    except Exception as e:
        print(f"❌ SupabaseLogger oluşturulamadı: {e}")
        return False
    
    # Test conversation log et
    print("\n=== Test Conversation Kaydetme ===")
    
    test_question = "Bu bir test sorusudur"
    test_answer = "Bu bir test cevabıdır"
    test_model = "test-model"
    test_sources = [{"title": "test-doc", "score": 0.95}]
    
    try:
        result = logger.log_conversation(
            question=test_question,
            answer=test_answer,
            model_name=test_model,
            confidence=0.85,
            sources=test_sources,
            response_time=1.23,
            user_id="test-user"
        )
        
        if result:
            print("✓ Test conversation başarıyla kaydedildi!")
        else:
            print("❌ Test conversation kaydedilemedi!")
            return False
            
    except Exception as e:
        print(f"❌ Test conversation kaydetme hatası: {e}")
        return False
    
    # Conversation history getir
    print("\n=== Conversation History Test ===")
    
    try:
        history = logger.get_conversation_history(limit=5)
        print(f"✓ Conversation history başarıyla getirildi. Kayıt sayısı: {len(history)}")
        
        if history:
            latest_conversation = history[0]
            print(f"Son conversation: {latest_conversation.get('question', 'N/A')[:50]}...")
        
    except Exception as e:
        print(f"❌ Conversation history getirme hatası: {e}")
        return False
    
    print("\n✅ Tüm testler başarılı!")
    return True

if __name__ == "__main__":
    success = test_supabase_connection()
    if not success:
        print("\n🔧 Çözüm önerileri:")
        print("1. .env dosyasının mevcut olduğundan emin olun")
        print("2. SUPABASE_URL ve SUPABASE_ANON_KEY değerlerinin doğru olduğunu kontrol edin")
        print("3. Supabase'de 'conversations' tablosunun mevcut olduğunu kontrol edin")
        print("4. İnternet bağlantınızı kontrol edin")
