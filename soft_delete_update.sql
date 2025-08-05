-- Soft delete için sadece conversations tablosuna kolon ekleme
-- conversation_threads bir view olduğu için ona kolon ekleyemiyoruz

-- conversations tablosuna deleted_by_user kolonu ekle
ALTER TABLE conversations
ADD COLUMN deleted_by_user BOOLEAN DEFAULT FALSE;

-- Index ekle (performans için)
CREATE INDEX IF NOT EXISTS idx_conversations_deleted_by_user
ON conversations(user_id, deleted_by_user, thread_id);

-- Mevcut verileri güncelle (tüm mevcut veriler silinmemiş olarak işaretle)
UPDATE conversations SET deleted_by_user = FALSE WHERE deleted_by_user IS NULL;
