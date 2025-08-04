-- ===== CONVERSATIONS TABLOSU GÜNCELLEMESİ =====

-- 1. Mevcut policy'leri kaldır
DROP POLICY IF EXISTS "Enable read access for all users" ON conversations;
DROP POLICY IF EXISTS "Enable insert access for all users" ON conversations;

-- 2. user_id tipini UUID'ye çevir ve auth.users'a reference et
ALTER TABLE conversations 
    ALTER COLUMN user_id TYPE UUID USING user_id::UUID;

-- 3. Foreign key constraint ekle (opsiyonel ama önerilen)
ALTER TABLE conversations 
    ADD CONSTRAINT fk_conversations_user_id 
    FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;

-- 4. Güvenli RLS policy'leri oluştur
-- Kullanıcılar sadece kendi kayıtlarını görebilsin
CREATE POLICY "Users can view own conversations" ON conversations
    FOR SELECT USING (auth.uid() = user_id);

-- Kullanıcılar sadece kendi adına kayıt ekleyebilsin
CREATE POLICY "Users can insert own conversations" ON conversations
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Kullanıcılar kendi kayıtlarını güncelleyebilsin (opsiyonel)
CREATE POLICY "Users can update own conversations" ON conversations
    FOR UPDATE USING (auth.uid() = user_id);

-- Kullanıcılar kendi kayıtlarını silebilsin (opsiyonel)
CREATE POLICY "Users can delete own conversations" ON conversations
    FOR DELETE USING (auth.uid() = user_id);

-- 5. Index'leri güncelle (UUID için optimize et)
DROP INDEX IF EXISTS idx_conversations_user_id;
CREATE INDEX idx_conversations_user_id ON conversations(user_id);

-- 6. Ek faydalı index'ler
CREATE INDEX idx_conversations_user_created ON conversations(user_id, created_at DESC);

-- ===== KULLANICI BİLGİLERİ VE İSTATİSTİKLER İÇİN VIEW =====

-- Kullanıcı conversation istatistikleri için view
CREATE OR REPLACE VIEW user_conversation_stats AS
SELECT 
    u.id as user_id,
    u.email,
    u.created_at as user_created_at,
    u.updated_at as user_updated_at,
    COALESCE(c.conversation_count, 0) as total_conversations,
    COALESCE(c.avg_confidence, 0) as avg_confidence,
    c.last_conversation_at,
    c.favorite_model
FROM auth.users u
LEFT JOIN (
    SELECT 
        user_id,
        COUNT(*) as conversation_count,
        AVG(confidence) as avg_confidence,
        MAX(created_at) as last_conversation_at,
        MODE() WITHIN GROUP (ORDER BY model_name) as favorite_model
    FROM conversations 
    GROUP BY user_id
) c ON u.id = c.user_id;

-- RLS for view
ALTER VIEW user_conversation_stats SET (security_invoker = true);

-- ===== HELPFUL FUNCTIONS =====

-- Kullanıcının toplam conversation sayısını getir
CREATE OR REPLACE FUNCTION get_user_conversation_count(target_user_id UUID DEFAULT auth.uid())
RETURNS INTEGER
LANGUAGE SQL
SECURITY DEFINER
AS $$
    SELECT COUNT(*)::INTEGER 
    FROM conversations 
    WHERE user_id = target_user_id;
$$;

-- Kullanıcının son N conversation'ını getir
CREATE OR REPLACE FUNCTION get_user_recent_conversations(
    limit_count INTEGER DEFAULT 10,
    target_user_id UUID DEFAULT auth.uid()
)
RETURNS TABLE (
    id BIGINT,
    question TEXT,
    answer TEXT,
    model_name VARCHAR(100),
    confidence FLOAT,
    sources JSONB,
    response_time FLOAT,
    created_at TIMESTAMPTZ
)
LANGUAGE SQL
SECURITY DEFINER
AS $$
    SELECT 
        c.id, c.question, c.answer, c.model_name, 
        c.confidence, c.sources, c.response_time, c.created_at
    FROM conversations c
    WHERE c.user_id = target_user_id
    ORDER BY c.created_at DESC
    LIMIT limit_count;
$$;
