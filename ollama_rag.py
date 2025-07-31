import requests
import json
from typing import Dict, Optional
import warnings

# Transformer uyarılarını bastır
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class OllamaRAGQA:
    """Ollama ile yerel LLM kullanarak RAG sistemi"""

    def __init__(self, retriever, model_name: str = "gemma3"):
        """
        Args:
            retriever: RAG retriever instance
            model_name: Ollama model adı (gemma3, llama3.2, deepseek-r1 vb.)
        """
        self.retriever = retriever
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"

    def check_ollama_status(self) -> bool:
        """Ollama'nın çalışıp çalışmadığını kontrol et"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('models', [])
                model_names = [model['name'] for model in models]
                print(f"📋 Mevcut modeller: {[name.split(':')[0] for name in model_names]}")

                # Kullanmak istediğimiz modelin var olup olmadığını kontrol et
                current_model_exists = any(self.model_name in name for name in model_names)
                if not current_model_exists:
                    print(f"⚠️ Model '{self.model_name}' bulunamadı!")
                    return False

                return True
            return False
        except requests.exceptions.ConnectionError:
            print("❌ Ollama sunucusuna bağlanılamadı. 'ollama serve' komutu çalıştırılmış mı?")
            return False
        except Exception as e:
            print(f"❌ Ollama bağlantı hatası: {e}")
            return False

    def generate_answer(self, question: str, context: str) -> str:

        if not self.check_ollama_status():
            return "Ollama çalışmıyor. Lütfen 'ollama serve' komutunu çalıştırın."

        # Prompt uzunluğunu kontrol et - çok uzun olabilir
        if len(context) > 15000:
            context = context[:15000] + "..."
            print(f"Bağlam çok uzun, kısaltıldı: {len(context)} karakter")

        # Sorunun dilini tespit et
        question_language = "Turkish" if any(turkish_word in question.lower() for turkish_word in ['nedir', 'nasıl', 'neden', 'ne', 'kim', 'hangi']) else "English"
        
        # Yeni hibrit yaklaşım: RAG bilgilerini kullanarak detaylı ve samimi cevap
        prompt = f"""Sen yardımcı ve bilgili bir yapay zeka asistanısın. Kullanıcının sorusunu aşağıdaki bilgileri kullanarak cevaplayacaksın.

VERİ TABANI BİLGİLERİ:
{context}

KULLANICININ SORUSU: {question}

TALİMATLAR:
1. Önce verilen bilgileri analiz et
2. Bu bilgileri kullanarak soruya detaylı bir cevap ver
3. Cevabını samimi ve yardımsever bir tonla yaz
4. Eğer bilgi yoksa açıkça belirt
5. Ek sorular olursa yardım etmeye hazır olduğunu söyle
6. {question_language} dilinde cevap ver

Lütfen soruyu verilen bilgilere dayanarak yanıtla:"""

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,  # Orta seviye yaratıcılık
                    "top_p": 0.9,
                    "max_tokens": 600  # Daha uzun cevaplar için
                }
            }

            print(f"🔍 Ollama API'ye istek gönderiliyor...")
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Cevap alınamadı.')
            else:
                error_detail = ""
                try:
                    error_detail = response.json().get("error", "")
                except:
                    error_detail = response.text[:200]

                print(f"❌ Ollama API hatası: {response.status_code} - {error_detail}")
                return f"Ollama hatası: HTTP {response.status_code} - {error_detail}"

        except requests.exceptions.Timeout:
            return "⏰ Ollama yanıt süresi aşıldı. Model çok büyük olabilir."
        except Exception as e:
            print(f"❌ Ollama bağlantı hatası: {str(e)}")
            return f"Ollama bağlantı hatası: {str(e)}"

    def is_general_chat_question(self, question: str) -> bool:
        """Sorunun genel sohbet sorusu olup olmadığını kontrol et"""
        general_chat_keywords = [
            'merhaba', 'selam', 'hello', 'hi', 'hey',
            'nasılsın', 'nasıl gidiyor', 'how are you', 'how do you do',
            'kimsin', 'who are you', 'what are you', 'ne yapıyorsun',
            'sen kimsin', 'sen nesin', 'yapay zeka', 'artificial intelligence',
            'bot musun', 'robot musun', 'ai misin', 'asistan mısın',
            'teşekkür', 'thank you', 'thanks', 'sağol', 'merci',
            'hoşça kal', 'bye', 'goodbye', 'görüşürüz', 'see you',
            'yardım et', 'help me', 'yardım', 'help', 'nasıl yardım',
            'anlat', 'tell me about yourself', 'kendin hakkında',
            'iyi misin', 'are you ok', 'how you doing'
        ]
        
        question_lower = question.lower()
        
        # Spesifik bilgi soruları: "X nedir?" formatı - RAG moduna gitsin
        if ('nedir' in question_lower or 'what is' in question_lower) and len(question.split()) > 1:
            # Sadece "nedir?" tek başına değilse RAG moduna git
            if question_lower.strip() not in ['nedir?', 'what is?']:
                return False
        
        # Tam eşleşme kontrolü (sadece çok genel sorular için)
        exact_matches = [
            'kimsin?', 'ne yapıyorsun?', 'yardım?',
            'sen kimsin?', 'sen nesin?', 'sen ne yapıyorsun?',
            'yapay zeka mısın?', 'bot musun?', 'ai misin?',
            'merhaba?', 'selam?', 'nedir?'
        ]
        
        if question_lower in exact_matches:
            return True
            
        return any(keyword in question_lower for keyword in general_chat_keywords)

    def generate_general_response(self, question: str) -> str:
        """Genel sohbet soruları için yapay zeka benzeri cevap üret"""
        if not self.check_ollama_status():
            return "Ollama çalışmıyor. Lütfen 'ollama serve' komutunu çalıştırın."

        prompt = f"""Sen yardımcı bir yapay zeka asistanısın. Kullanıcının sorusuna doğal ve samimi bir şekilde cevap ver.

SORU: {question}

Kısa, samimi ve yardımsever bir cevap ver. Türkçe cevap ver."""

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Daha yaratıcı yanıtlar için
                    "top_p": 0.9,
                    "max_tokens": 150
                }
            }

            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Cevap alınamadı.')
            else:
                return "Üzgünüm, şu anda cevap veremiyorum."

        except Exception as e:
            return "Merhaba! Size nasıl yardımcı olabilirim?"

    def answer_question(self, question: str, top_k: int = 3, confidence_threshold: float = 0.2) -> Dict:
        # Önce genel sohbet sorusu mu kontrol et
        if self.is_general_chat_question(question):
            answer = self.generate_general_response(question)
            return {
                'question': question,
                'answer': answer,
                'context': "",
                'sources': [],
                'method': 'general_chat'
            }

        # RAG araması yap
        search_results = self.retriever.search(question, top_k + 2)  # Daha fazla sonuç al, filtreleme sonrası için

        # Sonuç yoksa
        if not search_results:
            return {
                'question': question,
                'answer': "Üzgünüm, bu soruyla ilgili bilgi bulamadım.",
                'context': "",
                'sources': [],
                'method': 'no_results'
            }

        # Sonuçları çeşitlendirmek için, aynı içerikleri filtrele
        filtered_results = []
        seen_texts = set()

        for result in search_results:
            # İlk 50 karakter benzersiz mi kontrol et
            text_signature = result['text'][:50]
            if text_signature not in seen_texts:
                seen_texts.add(text_signature)
                filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break

        # Filtrelenmiş sonuçları kullan
        search_results = filtered_results[:top_k]

        # Bağlam oluştur
        context = self.retriever.get_context_for_query(question, top_k)

        # En yüksek güven skoru
        top_confidence = search_results[0]['score'] if search_results else 0

        # Güven skoru düşükse
        low_confidence = top_confidence < confidence_threshold

        # Ollama ile cevap oluştur
        if self.check_ollama_status():
            # Güven skoru düşükse uyarı ekle
            if low_confidence:
                answer = self.generate_answer(question, context)
                answer = f"⚠️ Güven skoru düşük. Cevap doğru olmayabilir.\n\n{answer}"
                method = 'low_confidence'
            else:
                answer = self.generate_answer(question, context)
                method = 'ollama_generated'
        else:
            # Fallback: En iyi eşleşen dökümanı kullan
            best_match = search_results[0]
            answer = f"🔍 En ilgili bilgi: {best_match['text'][:300]}..."
            method = 'retrieval_only'

        return {
            'question': question,
            'answer': answer,
            'context': context,
            'sources': search_results,
            'confidence': top_confidence,
            'method': method
        }

    def interactive_qa(self):
        """Etkileşimli soru-cevap modu"""
        print("🦙 Ollama RAG Soru-Cevap Sistemi")
        print(f"Model: {self.model_name}")
        print("Çıkmak için 'quit' yazın.")
        print("-" * 50)

        # Ollama durumunu kontrol et
        if not self.check_ollama_status():
            print("⚠️  Ollama çalışmıyor!")
            print("Kurulum için: https://ollama.ai")
            print("Başlatmak için: ollama serve")
            print("Model indirmek için: ollama deepseek-r1")
            print("\nYine de temel arama yapabilirsiniz...")
        else:
            print(f"✅ Ollama aktif - Model: {self.model_name}")

        while True:
            question = input("\n❓ Sorunuz: ").strip()

            # Boş sorgu kontrolü
            if not question:
                print("Lütfen bir soru yazın.")
                continue

            # Çıkış kontrolü
            if question.lower() in ['quit', 'exit', 'çık', 'çıkış']:
                print("👋 Hoşça kalın!")
                break

            try:
                print("🔍 Aranıyor...")
                result = self.answer_question(question)

                print(f"\n💬 Cevap:")
                print(result['answer'])

                if result['sources']:
                    print(f"\n📊 Güven skoru: {result['confidence']:.3f}")
                    print(f"📝 Kaynak sayısı: {len(result['sources'])}")
                    print(f"🔧 Method: {result['method']}")

                    if len(result['sources']) > 0:
                        print(f"\n📚 Kaynaklar:")
                        for i, source in enumerate(result['sources'][:2]):
                            print(f"  {i+1}. {source['text'][:100]}... (skor: {source['score']:.3f})")
                elif result['method'] == 'general_chat':
                    print("💬 Genel sohbet modu")

            except KeyboardInterrupt:
                print("\n\n👋 Kullanıcı tarafından durduruldu!")
                break
            except Exception as e:
                print(f"\n❌ Hata oluştu: {str(e)}")
                print(f"❌ Hata türü: {type(e).__name__}")
                import traceback
                print(f"❌ Detaylı hata:")
                traceback.print_exc()
                print("Tekrar deneyin veya 'quit' yazarak çıkın.")


if __name__ == "__main__":
    from rag_system import RAGRetriever

    print("Ollama RAG sistemi başlatılıyor...")

    retriever = RAGRetriever()

    try:
        retriever.load_from_files("faiss_index.bin", "documents.pkl")
        qa_system = OllamaRAGQA(retriever, model_name="gemma3")
        qa_system.interactive_qa()

    except FileNotFoundError:
        print("❌ Index dosyaları bulunamadı. Önce data_processor.py çalıştırın.")
    except Exception as e:
        print(f"❌ Hata: {e}")
