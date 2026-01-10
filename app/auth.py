"""
API 认证和并发/频率控制模块

功能：
- API key 验证
- 并发控制：每个 API key 同时只能有一个正在处理的请求
- 频率限制：每个 API key 在指定时间内只能发起一次请求
- 自动释放锁和更新时间戳
"""

import time
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
from fastapi import HTTPException, status
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIKeyConfig:
    """API Key 配置"""
    def __init__(self, key: str, name: str, rate_limit_seconds: int = 30):
        self.key = key
        self.name = name
        self.rate_limit_minutes = rate_limit_seconds / 60
        self.rate_limit_seconds = rate_limit_seconds


import threading

class RequestTracker:
    """请求跟踪器 - 用于并发控制和频率限制"""
    # 类级别变量，用于全服务器全局并发控制（单线程查询）
    _global_busy = False
    _global_lock = threading.Lock()

    def __init__(self):
        self.active_requests: Dict[str, bool] = {}  # key -> is_processing
        self.last_request_time: Dict[str, float] = {}  # key -> timestamp
        self._lock = threading.Lock()
    
    def is_request_active(self, api_key: str) -> bool:
        """检查该 API key 是否有正在处理的请求"""
        with self._lock:
            return self.active_requests.get(api_key, False)
    
    def try_acquire_global_lock(self) -> bool:
        """尝试获取全局查询锁（非阻塞）"""
        with RequestTracker._global_lock:
            if RequestTracker._global_busy:
                return False
            RequestTracker._global_busy = True
            return True

    def release_global_lock(self):
        """释放全局查询锁"""
        with RequestTracker._global_lock:
            RequestTracker._global_busy = False

    def get_remaining_cooldown(self, api_key: str, cooldown_seconds: int) -> Optional[float]:
        """
        获取剩余冷却时间
        返回 None 表示可以请求，返回秒数表示需要等待
        """
        with self._lock:
            last_time = self.last_request_time.get(api_key)
        
        if last_time is None:
            return None
        
        elapsed = time.time() - last_time
        if elapsed >= cooldown_seconds:
            return None
        
        return cooldown_seconds - elapsed
    
    def start_request(self, api_key: str):
        """标记请求开始"""
        with self._lock:
            self.active_requests[api_key] = True
        logger.info(f"🟢 Request started for API key: {api_key[:10]}...")
    
    def end_request(self, api_key: str):
        """标记请求结束，更新时间戳"""
        with self._lock:
            self.active_requests[api_key] = False
            self.last_request_time[api_key] = time.time()
        logger.info(f"🔴 Request ended for API key: {api_key[:10]}...")


class AuthManager:
    """认证管理器"""
    def __init__(self, config_path: str = None):
        if config_path is None:
            # 默认查找同目录下的 api_keys.yaml
            config_path = Path(__file__).parent / "api_keys.yaml"
        self.config_path = Path(config_path)
        self.api_keys: Dict[str, APIKeyConfig] = {}
        self.tracker = RequestTracker()
        self.load_api_keys()
    
    def load_api_keys(self):
        """从配置文件加载 API keys"""
        if not self.config_path.exists():
            logger.warning(f"⚠️ API keys 配置文件不存在: {self.config_path}")
            logger.warning("请创建 api_keys.yaml 文件")
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            for key_info in config.get('api_keys', []):
                key = key_info['key']
                name = key_info.get('name', 'Unknown')
                rate_limit = key_info.get('rate_limit_minutes', 3)
                
                self.api_keys[key] = APIKeyConfig(key, name, rate_limit)
                logger.info(f"✅ Loaded API key: {name} (rate limit: {rate_limit} min)")
        
        except Exception as e:
            logger.error(f"❌ 加载 API keys 失败: {e}")
            raise
    
    def validate_api_key(self, api_key: str) -> APIKeyConfig:
        """
        验证 API key
        返回 APIKeyConfig 或抛出 HTTPException
        """
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Please provide Authorization header with 'Bearer YOUR_API_KEY'"
            )
        
        # 移除 "Bearer " 前缀（如果有）
        if api_key.startswith("Bearer "):
            api_key = api_key[7:]
        
        config = self.api_keys.get(api_key)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return config
    
    def check_concurrency(self, api_key: str):
        """
        检查并发控制
        如果有正在处理的请求，抛出异常
        """
        if self.tracker.is_request_active(api_key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": {
                        "message": "上次请求未完成，请等待。每次只能处理一个请求。",
                        "type": "concurrent_request_error",
                        "code": "request_in_progress"
                    }
                }
            )
    
    def check_rate_limit(self, api_key: str, config: APIKeyConfig):
        """
        检查频率限制
        如果在冷却期内，抛出异常并返回剩余时间
        """
        remaining = self.tracker.get_remaining_cooldown(
            api_key, 
            config.rate_limit_seconds
        )
        
        if remaining is not None:
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": {
                        "message": f"请求过于频繁。请在 {minutes} 分 {seconds} 秒后重试。",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                        "retry_after_seconds": int(remaining)
                    }
                }
            )
    
    def acquire_request_slot(self, api_key: str) -> Tuple[str, APIKeyConfig]:
        """
        获取请求槽位（完整的验证流程）
        
        返回: (api_key, config)
        抛出: HTTPException（如果验证失败）
        """
        # 1. 验证 API key
        config = self.validate_api_key(api_key)
        
        # 2. 检查该 Key 的并发 (针对同一个用户)
        self.check_concurrency(api_key)
        
        # 3. 检查全局并发 (确保全服务器只有一个查询在跑，不排队)
        if not self.tracker.try_acquire_global_lock():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": {
                        "message": "系统正在处理其他查询，请稍后再试。目前全系统只能同时处理一个请求。",
                        "type": "system_busy_error",
                        "code": "system_is_busy"
                    }
                }
            )

        # 4. 检查频率限制 (冷却期)
        try:
            self.check_rate_limit(api_key, config)
        except HTTPException:
            # 如果频率限制没过，记得释放刚拿到的全局锁
            self.tracker.release_global_lock()
            raise
        
        # 5. 标记请求开始
        self.tracker.start_request(api_key)
        
        logger.info(f"✅ Request slot acquired for: {config.name}")
        return api_key, config
    
    def release_request_slot(self, api_key: str, global_only: bool = False):
        """释放请求槽位"""
        if not global_only:
            self.tracker.end_request(api_key)
        self.tracker.release_global_lock()
        logger.info(f"✅ Request slot and global lock released for: {api_key[:10]}...")
