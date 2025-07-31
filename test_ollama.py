import requests
import json

print("🔍 Gemma3 modeli detaylı test ediliyor...")

try:
    # Önce Ollama'nın çalışıp çalışmadığını kontrol et
    response = requests.get('http://localhost:11434/api/tags', timeout=10)
    if response.status_code == 200:
        print("✅ Ollama API çalışıyor!")

        # Sistem belleği bilgisini almaya çalış
        print("\n🧪 Gemma3 basit test...")
        test_payload = {
            "model": "gemma3",
            "prompt": "Hello",
            "stream": False,
            "options": {
                "num_predict": 10  # Sadece 10 kelime
            }
        }
        
        print("📤 İstek gönderiliyor...")
        test_response = requests.post(
            'http://localhost:11434/api/generate',
            json=test_payload,
            timeout=60
        )

        print(f"📨 Yanıt durumu: {test_response.status_code}")

        if test_response.status_code == 200:
            result = test_response.json()
            print(f"✅ Gemma3 çalışıyor!")
            print(f"📝 Cevap: {result.get('response', 'Cevap alınamadı')}")
        else:
            print(f"❌ Gemma3 hatası: {test_response.status_code}")
            try:
                error_data = test_response.json()
                print(f"🔍 Hata detayı: {error_data}")
            except:
                print(f"🔍 Ham hata: {test_response.text}")

    else:
        print(f"❌ Ollama API hatası: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("❌ Ollama sunucusuna bağlanılamadı!")
    print("Lütfen 'ollama serve' komutunu çalıştırın.")
except Exception as e:
    print(f"❌ Beklenmeyen hata: {e}")
