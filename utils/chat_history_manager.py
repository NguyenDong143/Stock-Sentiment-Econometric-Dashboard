"""
Chat History Manager - Quản lý lưu trữ và khôi phục lịch sử hội thoại
Hỗ trợ: JSON format, auto-save, session management
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Quản lý lưu trữ lịch sử chat vào file JSON"""

    def __init__(self, history_dir: str = "data/chat_history"):
        """
        Khởi tạo ChatHistoryManager
        
        Args:
            history_dir: Thư mục lưu trữ file lịch sử chat
        """
        self.history_dir = history_dir
        self._ensure_directory()

    def _ensure_directory(self):
        """Tạo thư mục nếu chưa tồn tại"""
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
            logger.info(f"✅ Đã tạo thư mục: {self.history_dir}")

    def get_session_filepath(self, session_id: str = "default") -> str:
        """
        Lấy đường dẫn file cho session cụ thể
        
        Args:
            session_id: ID của session (mặc định: "default")
            
        Returns:
            Đường dẫn file JSON
        """
        filename = f"chat_{session_id}.json"
        return os.path.join(self.history_dir, filename)

    def save_history(
        self,
        messages: List[Dict],
        session_id: str = "default",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Lưu lịch sử chat vào file JSON
        
        Args:
            messages: Danh sách tin nhắn (user/assistant)
            session_id: ID của session
            metadata: Thông tin bổ sung (ticker, timestamp, etc.)
            
        Returns:
            True nếu lưu thành công
        """
        try:
            filepath = self.get_session_filepath(session_id)
            
            # Chuẩn bị dữ liệu
            data = {
                "session_id": session_id,
                "last_updated": datetime.now().isoformat(),
                "message_count": len(messages),
                "metadata": metadata or {},
                "messages": messages
            }
            
            # Ghi vào file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Đã lưu {len(messages)} tin nhắn vào {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu lịch sử: {e}")
            return False

    def load_history(self, session_id: str = "default") -> Optional[Dict]:
        """
        Tải lịch sử chat từ file JSON
        
        Args:
            session_id: ID của session
            
        Returns:
            Dictionary chứa lịch sử hoặc None nếu không tìm thấy
        """
        try:
            filepath = self.get_session_filepath(session_id)
            
            if not os.path.exists(filepath):
                logger.info(f"📂 Không tìm thấy lịch sử cho session: {session_id}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✅ Đã tải {data.get('message_count', 0)} tin nhắn từ {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải lịch sử: {e}")
            return None

    def get_messages(self, session_id: str = "default") -> List[Dict]:
        """
        Lấy danh sách tin nhắn từ file
        
        Args:
            session_id: ID của session
            
        Returns:
            Danh sách tin nhắn hoặc list rỗng
        """
        data = self.load_history(session_id)
        return data.get("messages", []) if data else []

    def clear_history(self, session_id: str = "default") -> bool:
        """
        Xóa lịch sử chat
        
        Args:
            session_id: ID của session
            
        Returns:
            True nếu xóa thành công
        """
        try:
            filepath = self.get_session_filepath(session_id)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"🗑️ Đã xóa lịch sử: {filepath}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi xóa lịch sử: {e}")
            return False

    def list_sessions(self) -> List[str]:
        """
        Liệt kê tất cả các session có lịch sử
        
        Returns:
            Danh sách session IDs
        """
        try:
            files = [f for f in os.listdir(self.history_dir) if f.startswith("chat_") and f.endswith(".json")]
            sessions = [f.replace("chat_", "").replace(".json", "") for f in files]
            return sorted(sessions)
        except Exception as e:
            logger.error(f"❌ Lỗi khi liệt kê sessions: {e}")
            return []

    def get_session_info(self, session_id: str = "default") -> Optional[Dict]:
        """
        Lấy thông tin tóm tắt về session
        
        Args:
            session_id: ID của session
            
        Returns:
            Dictionary chứa thông tin session
        """
        data = self.load_history(session_id)
        if not data:
            return None
        
        return {
            "session_id": data.get("session_id"),
            "last_updated": data.get("last_updated"),
            "message_count": data.get("message_count"),
            "metadata": data.get("metadata", {})
        }

    def export_to_text(self, session_id: str = "default") -> Optional[str]:
        """
        Xuất lịch sử chat sang định dạng text đọc được
        
        Args:
            session_id: ID của session
            
        Returns:
            String chứa lịch sử chat hoặc None
        """
        data = self.load_history(session_id)
        if not data:
            return None
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"CHAT HISTORY - SESSION: {session_id}")
        lines.append(f"Last Updated: {data.get('last_updated', 'N/A')}")
        lines.append(f"Total Messages: {data.get('message_count', 0)}")
        lines.append("=" * 60)
        lines.append("")
        
        for msg in data.get("messages", []):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
            lines.append("")
        
        return "\n".join(lines)

    def backup_session(self, session_id: str = "default") -> Optional[str]:
        """
        Tạo bản backup của session với timestamp
        
        Args:
            session_id: ID của session
            
        Returns:
            Đường dẫn file backup hoặc None
        """
        try:
            data = self.load_history(session_id)
            if not data:
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"chat_{session_id}_backup_{timestamp}.json"
            backup_filepath = os.path.join(self.history_dir, backup_filename)
            
            with open(backup_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Đã backup session: {backup_filepath}")
            return backup_filepath
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi backup session: {e}")
            return None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_default_manager() -> ChatHistoryManager:
    """Lấy instance mặc định của ChatHistoryManager"""
    return ChatHistoryManager()


def quick_save(messages: List[Dict], session_id: str = "default", **metadata):
    """
    Hàm tiện lợi để lưu nhanh
    
    Args:
        messages: Danh sách tin nhắn
        session_id: ID session
        **metadata: Thông tin bổ sung
    """
    manager = get_default_manager()
    manager.save_history(messages, session_id, metadata)


def quick_load(session_id: str = "default") -> List[Dict]:
    """
    Hàm tiện lợi để tải nhanh
    
    Args:
        session_id: ID session
        
    Returns:
        Danh sách tin nhắn
    """
    manager = get_default_manager()
    return manager.get_messages(session_id)
