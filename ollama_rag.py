import requests
import json
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class OllamaRAGQA:

    def __init__(self, retriever, model_name: str = "deepseek-r1:1.5b"):

        self.retriever = retriever
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"

    def check_ollama_status(self) -> bool:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('models', [])
                model_names = [model['name'] for model in models]
                print(f"Mevcut modeller: {[name.split(':')[0] for name in model_names]}")

                current_model_exists = any(self.model_name in name for name in model_names)
                if not current_model_exists:
                    print(f"⚠Model '{self.model_name}' bulunamadı!")
                    return False

                return True
            return False
        except requests.exceptions.ConnectionError:
            print("Ollama sunucusuna bağlanılamadı. 'ollama serve' komutu çalıştırılmış mı?")
            return False
        except Exception as e:
            print(f"Ollama bağlantı hatası: {e}")
            return False

    def generate_answer(self, question: str, context: str) -> str:

        if not self.check_ollama_status():
            return "Ollama çalışmıyor. Lütfen 'ollama serve' komutunu çalıştırın."

        if len(context) > 15000:
            context = context[:15000] + "..."
            print(f"Bağlam çok uzun, kısaltıldı: {len(context)} karakter")

        question_language = "Turkish" if any(turkish_word in question.lower() for turkish_word in ['nedir', 'nasıl', 'neden', 'ne', 'kim', 'hangi']) else "English"
        
        if context and context.strip():
            prompt = f"""Sen yardımcı ve bilgili bir yapay zeka asistanısın. Kullanıcının sorusunu aşağıdaki bilgileri kullanarak cevaplayacaksın.

VERİ TABANI BİLGİLERİ:
{context}

KULLANICININ SORUSU: {question}

TALİMATLAR:
1. Verilen bilgileri kullanarak soruya net ve anlaşılır bir cevap ver
2. Cevabını samimi ve yardımsever bir tonla yaz
3. Ham veri yapıştırma, işleyip düzgün cevap ver
4. {question_language} dilinde cevap ver
5. Eğer verilen bilgiler yetersizse, genel bilginle destekle

Lütfen soruyu yanıtla:"""
        else:
            prompt = f"""Sen yardımcı bir yapay zeka asistanısın. Kullanıcının sorusuna mevcut genel bilginle cevap ver.

KULLANICININ SORUSU: {question}

TALİMATLAR:
1. Soruya samimi ve yardımsever bir tonla cevap ver
2. {question_language} dilinde cevap ver
3. Eğer bilmiyorsan dürüstçe söyle

Lütfen soruyu yanıtla:"""

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "max_tokens": 600
                }
            }

            print(f"Ollama API'ye istek gönderiliyor")
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

                print(f"Ollama API hatası: {response.status_code} - {error_detail}")
                return f"Ollama hatası: HTTP {response.status_code} - {error_detail}"

        except requests.exceptions.Timeout:
            return "Ollama yanıt süresi aşıldı. Model çok büyük olabilir."
        except Exception as e:
            print(f"Ollama bağlantı hatası: {str(e)}")
            return f"Ollama bağlantı hatası: {str(e)}"

    def generate_answer_stream(self, question: str, context: str):
        if not self.check_ollama_status():
            yield "Ollama çalışmıyor. Lütfen 'ollama serve' komutunu çalıştırın."
            return

        if len(context) > 15000:
            context = context[:15000] + "..."

        question_language = "Turkish" if any(turkish_word in question.lower() for turkish_word in ['nedir', 'nasıl', 'neden', 'ne', 'kim', 'hangi']) else "English"

        if context and context.strip():
            prompt = f"""You are a GEODI AI assitant. You will answer the user's question using the provided database information.

DATABASE INFORMATION:
{context}

USER'S QUESTION: {question}

INSTRUCTIONS:
1. Use the provided information to give a clear and concise answer
2. Write your answer in a friendly and helpful tone
3. Do not paste raw data, process it and give a proper answer
4. Answer in {question_language} language
5. If the provided information is insufficient, supplement with your general knowledge

Please answer the question:"""
        else:
            prompt = f"""You are a helpful AI assistant. You will answer the user's question using your general knowledge.

USER'S QUESTION: {question}

INSTRUCTIONS:
1. Provide a friendly and helpful answer
2. Answer in {question_language} language
3. If you don't know, be honest and say so

Please answer the question:"""

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "max_tokens": 600
                }
            }

            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60,
                stream=True
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line.decode('utf-8'))
                            if 'response' in chunk_data:
                                text_chunk = chunk_data['response']
                                if text_chunk:
                                    yield text_chunk
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Ollama hatası: HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            yield "Ollama yanıt süresi aşıldı. Model çok büyük olabilir."
        except Exception as e:
            yield f"Ollama bağlantı hatası: {str(e)}"

    def is_general_chat_question(self, question: str) -> bool:
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
        
        if ('nedir' in question_lower or 'what is' in question_lower) and len(question.split()) > 1:
            if question_lower.strip() not in ['nedir?', 'what is?']:
                return False
        
        exact_matches = [
            'kimsin?', 'ne yapıyorsun?', 'yardım?',
            'sen kimsin?', 'sen nesin?', 'sen ne yapıyorsun?',
            'yapay zeka mısın?', 'bot musun?', 'ai misin?',
            'merhaba?', 'selam?', 'nedir?'
        ]
        
        if question_lower in exact_matches:
            return True
            
        return any(keyword in question_lower for keyword in general_chat_keywords)

    def _contains_technical_terms(self, question: str) -> bool:
        technical_terms = [
            'port', 'api', 'agent', 'server', 'database', 'config', 'ip', 'url',
            'geodi', 'gde', 'discovery', 'communication', 'protocol', 'service',
            'application', 'system', 'network', 'connection', 'authentication',
            'installation', 'configuration', 'deployment', 'monitoring'
        ]
        
        question_lower = question.lower()
        return any(term in question_lower for term in technical_terms)

    def _check_answer_consistency(self, results: List[Dict], question: str) -> Dict:
        if not results:
            return None
            
        best_result = max(results, key=lambda x: x['score'])
        
        similar_count = sum(1 for r in results if r['score'] >= best_result['score'] * 0.8)
        
        return {
            'consistent_answer': best_result['text'],
            'confidence': best_result['score'] * (1 + (similar_count - 1) * 0.1),
            'source_count': similar_count
        }

    def generate_general_response(self, question: str) -> str:
        if not self.check_ollama_status():
            return "Ollama çalışmıyor. Lütfen 'ollama serve' komutunu çalıştırın."

        prompt = f"""You are a helpful AI assistant. You will answer the user's question in a casual and engaging way.

SORU: {question}

Please answer the question in a friendly and conversational tone:"""

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
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
                return "I'am sorry, I cant's answer that right now."

        except Exception as e:
            return "Hello, how can I help you? I'm just a simple AI assistant and can't answer that right now."
    def answer_question(self, question: str, top_k: int = 5, confidence_threshold: float = 0.0) -> Dict:
        search_results = self.retriever.search(question, 10)

        if not search_results:
            return {
                'question': question,
                'answer': "Üzgünüm, bu soruyla ilgili bilgi bulamadım.",
                'context': "",
                'sources': [],
                'method': 'no_results'
            }

        filtered_results = []
        seen_texts = set()

        search_results = sorted(search_results, key=lambda x: x['score'], reverse=True)

        for result in search_results:
            text_signature = result['text'][:50]
            if text_signature not in seen_texts:
                seen_texts.add(text_signature)
                filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break

        high_score_results = [r for r in filtered_results if r['score'] >= 0.4]
        if high_score_results:
            search_results = high_score_results[:top_k]
        else:
            search_results = filtered_results[:top_k]

        context = self.retriever.get_context_for_query(question, top_k)

        consistency_check = self._check_answer_consistency(search_results, question)
        
        top_confidence = search_results[0]['score'] if search_results else 0
        
        if consistency_check and consistency_check.get('source_count', 1) > 1:
            top_confidence = max(top_confidence, consistency_check['confidence'])

        low_confidence = top_confidence < confidence_threshold


        if self.check_ollama_status():
            if search_results and context:
                answer = self.generate_answer(question, context)
                method = 'ollama_with_rag'
            else:
                answer = self.generate_general_response(question)
                method = 'ollama_general'
        else:
            answer = "Üzgünüm, şu anda cevap oluşturamıyorum. Ollama servisi çalışmıyor."
            method = 'service_unavailable'

        return {
            'question': question,
            'answer': answer,
            'context': context,
            'sources': search_results,
            'confidence': top_confidence,
            'method': method
        }

    def interactive_qa(self):
        print("Ollama RAG Soru-Cevap Sistemi")
        print(f"Model: {self.model_name}")
        print("Çıkmak için 'quit' yazın.")
        print("-" * 50)

        if not self.check_ollama_status():
            print("Ollama çalışmıyor!")
            print("Kurulum için: https://ollama.ai")
            print("Başlatmak için: ollama serve")
            print("Model indirmek için: ollama deepseek-r1:1.5b")
            print("\nYine de temel arama yapabilirsiniz...")
        else:
            print(f"Ollama aktif - Model: {self.model_name}")

        while True:
            question = input("\nSorunuz: ").strip()

            if not question:
                print("Lütfen bir soru yazın.")
                continue

            if question.lower() in ['quit', 'exit', 'çık', 'çıkış']:
                print("Hoşça kalın!")
                break

            try:
                print("Aranıyor...")
                result = self.answer_question(question)

                print(f"\nCevap:")
                print(result['answer'])

                if result['sources']:
                    print(f"\nGüven skoru: {result['confidence']:.3f}")
                    print(f"Kaynak sayısı: {len(result['sources'])}")
                    print(f"Method: {result['method']}")

                    if len(result['sources']) > 0:
                        print(f"\nKaynaklar:")
                        for i, source in enumerate(result['sources'][:4]):
                            print(f"  {i+1}. {source['text'][:100]}... (skor: {source['score']:.3f})")
                elif result['method'] == 'general_chat':
                    print("Genel sohbet modu")

            except KeyboardInterrupt:
                print("\n\nKullanıcı tarafından durduruldu!")
                break
            except Exception as e:
                print(f"\nHata oluştu: {str(e)}")
                print(f"Hata türü: {type(e).__name__}")
                import traceback
                print(f"Detaylı hata:")
                traceback.print_exc()
                print("Tekrar deneyin veya 'quit' yazarak çıkın.")


if __name__ == "__main__":
    from rag_system import RAGRetriever

    print("Ollama RAG sistemi başlatılıyor...")

    retriever = RAGRetriever()

    try:
        retriever.load_from_files("faiss_index.bin", "documents.pkl")
        qa_system = OllamaRAGQA(retriever, model_name="deepseek-r1:1.5b")
        qa_system.interactive_qa()

    except FileNotFoundError:
        print("Index dosyaları bulunamadı. Önce data_processor.py çalıştırın.")
    except Exception as e:
        print(f"Hata: {e}")
