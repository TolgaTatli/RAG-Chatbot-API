#!/usr/bin/env python3
"""
RAG API'nin düzgün çalışıp conversation'ları kaydettiğini test etmek için script
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_api_conversation_logging():
    print("=== RAG API Conversation Logging Testi ===")
    
    # API'nin çalışıp çalışmadığını kontrol et
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            print("✓ API çalışıyor")
            status_data = response.json()
            print(f"  - RAG System: {status_data.get('rag_system')}")
            print(f"  - Ollama Status: {status_data.get('ollama_status')}")
            print(f"  - Model: {status_data.get('model')}")
        else:
            print("❌ API çalışmıyor")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API'ye bağlanılamıyor. Lütfen önce API'yi başlatın:")
        print("   python rag_api.py")
        return False
    except Exception as e:
        print(f"❌ API status kontrolü hatası: {e}")
        return False
    
    # Test conversation gönder
    print("\n=== Test Conversation Gönderme ===")
    
    test_data = {
        "question": "Test sorusu - bu bir test conversation'ıdır",
        "top_k": 3,
        "user_id": "test-user-123"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✓ Test conversation başarıyla gönderildi")
            result = response.json()
            print(f"  - Answer: {result.get('answer', 'N/A')[:100]}...")
            print(f"  - Confidence: {result.get('confidence', 0)}")
            print(f"  - Method: {result.get('method', 'N/A')}")
        else:
            print(f"❌ Test conversation gönderme başarısız: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Test conversation gönderme hatası: {e}")
        return False
    
    # Conversation history kontrol et
    print("\n=== Conversation History Kontrolü ===")
    
    time.sleep(1)  # Kayıt için biraz bekle
    
    try:
        response = requests.get(f"{API_BASE_URL}/history?limit=5")
        
        if response.status_code == 200:
            history_data = response.json()
            conversations = history_data.get("conversations", [])
            print(f"✓ Conversation history başarıyla getirildi. Kayıt sayısı: {len(conversations)}")
            
            if conversations:
                latest = conversations[0]
                print(f"  - Son soru: {latest.get('question', 'N/A')[:50]}...")
                print(f"  - Model: {latest.get('model_name', 'N/A')}")
                print(f"  - User ID: {latest.get('user_id', 'N/A')}")
                print(f"  - Tarih: {latest.get('created_at', 'N/A')}")
                
                # Test sorusunun kaydedilip kaydedilmediğini kontrol et
                if "Test sorusu - bu bir test conversation" in latest.get('question', ''):
                    print("✅ Test conversation başarıyla Supabase'e kaydedildi!")
                    return True
                else:
                    print("⚠️ Test conversation bulunamadı, eski kayıtlar gösteriliyor")
            else:
                print("⚠️ Hiç conversation history bulunamadı")
                return False
        else:
            print(f"❌ History getirme başarısız: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ History kontrolü hatası: {e}")
        return False

if __name__ == "__main__":
    print("RAG API'nin çalıştığından emin olun (python rag_api.py)")
    print("Test başlatılıyor...\n")
    
    success = test_api_conversation_logging()
    
    if success:
        print("\n🎉 Tüm testler başarılı! Conversation logging çalışıyor.")
    else:
        print("\n❌ Test başarısız. Loglara bakın ve sorunları çözün.")
