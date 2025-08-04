# 🤖 RAG Chat API with Authentication

Bu proje, Supabase Authentication ile desteklenen bir RAG (Retrieval-Augmented Generation) Chat API'sidir.

## ✨ Özellikler

### 🔐 Authentication
- **Email/Password** ile kayıt ve giriş
- **Magic Links** (şifresiz giriş)
- **JWT Token** tabanlı kimlik doğrulama
- **Şifre sıfırlama** 
- **Kullanıcı profil yönetimi**

### 🤖 RAG Chat
- **Ollama** ile yerel LLM
- **FAISS** vector database
- **Streaming responses**
- **Conversation history** (kullanıcı bazında)
- **Source tracking** ve confidence scoring

### 💰 Maliyet
- **TAMAMEN ÜCRETSİZ!** 
- Supabase ücretsiz planı: 50,000 Monthly Active Users
- Ollama yerel LLM - internet gerektirmez
- FAISS yerel vector database

## 🚀 Kurulum

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

### 2. Supabase Kurulumu
1. [Supabase](https://supabase.com) hesabı oluşturun (ücretsiz)
2. Yeni proje oluşturun
3. Settings > API'den URL ve anon key'i alın
4. Authentication > Settings'den email confirmations'ı ayarlayın

### 3. Environment Variables
`.env` dosyası oluşturun:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
```

### 4. Database Schema
Supabase'de şu tabloyu oluşturun:

```sql
-- conversations tablosu
CREATE TABLE conversations (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    model_name TEXT,
    confidence FLOAT DEFAULT 0,
    sources JSONB,
    response_time FLOAT,
    user_id UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS (Row Level Security) aktif edin
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

-- Kullanıcılar sadece kendi kayıtlarını görebilsin
CREATE POLICY "Users can view own conversations" ON conversations
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own conversations" ON conversations
    FOR INSERT WITH CHECK (auth.uid() = user_id);
```

### 5. Ollama Kurulumu
```bash
# Ollama'yı kurun (https://ollama.ai)
ollama pull gemma3  # veya başka bir model
```

## 🏃‍♂️ Çalıştırma

### API'yi Başlatın
```bash
python rag_api.py
```

### Test Scriptleri
```bash
# Supabase bağlantısını test et
python test_supabase.py

# Auth sistemini test et  
python test_auth.py

# API logging'i test et
python test_api_logging.py
```

### Frontend
`auth_frontend.html` dosyasını tarayıcıda açın.

## 📚 API Endpoints

### 🔐 Authentication

#### Kayıt Ol
```http
POST /auth/signup
Content-Type: application/json

{
    "email": "user@example.com",
    "password": "password123",
    "full_name": "John Doe"
}
```

#### Giriş Yap
```http
POST /auth/signin
Content-Type: application/json

{
    "email": "user@example.com", 
    "password": "password123"
}

Response:
{
    "message": "Giriş başarılı!",
    "access_token": "jwt_token_here",
    "user": {
        "id": "user_id",
        "email": "user@example.com",
        "created_at": "2025-01-01T00:00:00Z"
    }
}
```

#### Magic Link (Şifresiz Giriş)
```http
POST /auth/magic-link
Content-Type: application/json

{
    "email": "user@example.com"
}
```

#### Kullanıcı Bilgileri
```http
GET /auth/me
Authorization: Bearer <jwt_token>
```

#### Çıkış
```http
POST /auth/signout
```

### 🤖 RAG Chat

#### Soru Sor (Authenticated)
```http
POST /generate
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    "question": "Merhaba, nasılsın?",
    "top_k": 3
}
```

#### Soru Sor (Anonymous)
```http
POST /generate
Content-Type: application/json

{
    "question": "Merhaba, nasılsın?",
    "top_k": 3,
    "user_id": "anonymous-user"
}
```

#### Streaming Response
```http
POST /generate-stream
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    "question": "Uzun bir açıklama yap",
    "top_k": 3
}
```

#### Conversation History
```http
GET /history?limit=50
Authorization: Bearer <jwt_token>
```

## 🔒 Güvenlik

### JWT Token Kullanımı
- Tüm authenticated endpoint'ler `Authorization: Bearer <token>` header'ı bekler
- Token'lar Supabase tarafından yönetilir ve doğrulanır
- Otomatik token refresh desteklenir

### Row Level Security (RLS)
- Her kullanıcı sadece kendi conversation'larını görebilir
- Database seviyesinde güvenlik
- Supabase Auth ile entegre

### Privacy
- Anonymous kullanıcılar `user_id` belirtebilir
- Authenticated kullanıcıların ID'si otomatik kullanılır
- Conversation'lar kullanıcı bazında izole edilir

## 🌟 Özellik Karşılaştırması

| Özellik | Anonymous | Authenticated |
|---------|-----------|---------------|
| RAG Chat | ✅ | ✅ |
| Conversation History | ❌ | ✅ |
| User Tracking | Manuel user_id | Otomatik |
| Data Privacy | Yok | RLS korumalı |
| Stream Chat | ✅ | ✅ |
| Magic Links | ❌ | ✅ |

## 🎯 Kullanım Senaryoları

### 1. **Anonim Kullanım**
```javascript
// Frontend'de token olmadan
fetch('/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        question: "Soru",
        user_id: "guest-123"
    })
})
```

### 2. **Authenticated Kullanım**
```javascript
// Token ile
fetch('/generate', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({
        question: "Soru"
    })
})
```

### 3. **Magic Link Workflow**
1. Kullanıcı email girer
2. Magic link gönderilir
3. Link'e tıklandığında otomatik giriş
4. Frontend'de token yakalanır

## 🔧 Yapılandırma

### Supabase Auth Settings
```javascript
// Supabase dashboard'da
{
    "SITE_URL": "http://localhost:3000",
    "ADDITIONAL_REDIRECT_URLS": ["http://localhost:8000"],
    "JWT_EXPIRY": 3600,
    "REFRESH_TOKEN_ROTATION": true,
    "EXTERNAL_EMAIL_ENABLED": true,
    "EXTERNAL_PHONE_ENABLED": false
}
```

### Environment Variables
```env
# Gerekli
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key

# Opsiyonel
OLLAMA_MODEL=gemma3
RAG_TOP_K=3
MAX_CONVERSATION_HISTORY=100
```

## 🐛 Sorun Giderme

### Yaygın Hatalar

1. **"Auth servisi mevcut değil"**
   - `.env` dosyasını kontrol edin
   - Supabase credentials'ları doğrulayın

2. **"Token doğrulama hatası"**
   - Token'ın expired olup olmadığını kontrol edin
   - Supabase project settings'ini kontrol edin

3. **"Conversation kaydedilmedi"**
   - Database bağlantısını test edin
   - RLS policy'lerini kontrol edin

### Debug
```python
# test_auth.py çalıştırın
python test_auth.py

# Logları kontrol edin
# Console'da "✓ Başarılı" / "✗ Başarısız" mesajları
```

## 📈 Performans

- **Supabase**: 50,000 MAU ücretsiz
- **Ollama**: Yerel, sınırsız
- **FAISS**: Hızlı vector search
- **JWT**: Stateless authentication

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun
3. Commit edin
4. Push edin
5. Pull Request açın

## 📄 Lisans

MIT License - Detaylar için LICENSE dosyasına bakın.

---

## 🎉 Sonuç

Bu proje tamamen ücretsiz bir şekilde production-ready bir RAG Chat sistemi sunar. Supabase'in ücretsiz planı küçük-orta ölçekli projeler için fazlasıyla yeterlidir!

**50,000 aylık aktif kullanıcıya kadar tamamen ücretsiz!** 🎊
