-- ===== CONVERSATION THREADING SİSTEMİ =====

-- 1. Conversations tablosuna thread yapısı ekle
ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS thread_id UUID DEFAULT NULL,
ADD COLUMN IF NOT EXISTS parent_message_id INTEGER DEFAULT NULL,
ADD COLUMN IF NOT EXISTS message_order INTEGER DEFAULT 1;

-- 2. Thread ID için index ekle
CREATE INDEX IF NOT EXISTS idx_conversations_thread_id ON conversations(thread_id);
CREATE INDEX IF NOT EXISTS idx_conversations_parent_message ON conversations(parent_message_id);
CREATE INDEX IF NOT EXISTS idx_conversations_thread_order ON conversations(thread_id, message_order);

-- 3. Mevcut conversations'ları thread'lere organize et
-- Her conversation kendi thread'inin ilk mesajı olarak ayarla
UPDATE conversations
SET thread_id = gen_random_uuid(),
    message_order = 1
WHERE thread_id IS NULL;

-- 4. Thread başlıkları için view oluştur
CREATE OR REPLACE VIEW conversation_threads AS
SELECT
    c.thread_id,
    c.user_id,
    first_msg.question as thread_title,
    first_msg.created_at as thread_created_at,
    last_msg.created_at as last_updated_at,
    COUNT(c.id) as message_count,
    first_msg.id as first_message_id,
    last_msg.id as last_message_id
FROM conversations c
LEFT JOIN conversations first_msg ON (
    first_msg.thread_id = c.thread_id
    AND first_msg.message_order = 1
)
LEFT JOIN conversations last_msg ON (
    last_msg.thread_id = c.thread_id
    AND last_msg.message_order = (
        SELECT MAX(message_order)
        FROM conversations
        WHERE thread_id = c.thread_id
    )
)
WHERE c.thread_id IS NOT NULL
GROUP BY c.thread_id, c.user_id, first_msg.question, first_msg.created_at, last_msg.created_at, first_msg.id, last_msg.id;

-- 5. Thread mesajları için function oluştur
CREATE OR REPLACE FUNCTION get_thread_messages(target_thread_id UUID)
RETURNS TABLE (
    id INTEGER,
    thread_id UUID,
    parent_message_id INTEGER,
    message_order INTEGER,
    question TEXT,
    answer TEXT,
    model_name TEXT,
    confidence FLOAT,
    sources JSONB,
    response_time FLOAT,
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        c.id,
        c.thread_id,
        c.parent_message_id,
        c.message_order,
        c.question,
        c.answer,
        c.model_name,
        c.confidence,
        c.sources,
        c.response_time,
        c.user_id,
        c.created_at
    FROM conversations c
    WHERE c.thread_id = target_thread_id
    ORDER BY c.message_order ASC;
$$;

-- 6. Yeni thread oluşturma function'ı
CREATE OR REPLACE FUNCTION create_new_thread()
RETURNS UUID
LANGUAGE SQL
AS $$
    SELECT gen_random_uuid();
$$;

-- 7. Thread'e mesaj ekleme function'ı
CREATE OR REPLACE FUNCTION get_next_message_order(target_thread_id UUID)
RETURNS INTEGER
LANGUAGE SQL STABLE
AS $$
    SELECT COALESCE(MAX(message_order), 0) + 1
    FROM conversations
    WHERE thread_id = target_thread_id;
$$;

-- 8. RLS policy'lerini thread yapısına uygun güncelle
DROP POLICY IF EXISTS "Users can view own conversations" ON conversations;
DROP POLICY IF EXISTS "Users can insert own conversations" ON conversations;
DROP POLICY IF EXISTS "Users can update own conversations" ON conversations;
DROP POLICY IF EXISTS "Users can delete own conversations" ON conversations;

-- Thread-aware policy'ler
CREATE POLICY "Users can view own conversation threads" ON conversations
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert into own conversation threads" ON conversations
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own conversation threads" ON conversations
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own conversation threads" ON conversations
    FOR DELETE USING (auth.uid() = user_id);

-- 9. Thread istatistikleri için view
CREATE OR REPLACE VIEW user_thread_stats AS
SELECT
    user_id,
    COUNT(DISTINCT thread_id) as total_threads,
    SUM(message_count) as total_messages,
    AVG(message_count) as avg_messages_per_thread,
    MAX(last_updated_at) as last_activity_at
FROM conversation_threads
GROUP BY user_id;
