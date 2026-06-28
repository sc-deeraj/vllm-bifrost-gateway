import json
import logging
from typing import Any, Optional

import redis

from .config import settings

logger = logging.getLogger("model_manager.redis")


class RedisClient:
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.enabled = settings.redis_enabled

    async def connect(self):
        if not self.enabled:
            logger.info("Redis caching disabled")
            return
        try:
            self.client = redis.from_url(settings.redis_url, decode_responses=True)
            if self.client.ping():
                logger.info("Redis cache connected")
        except Exception as exc:
            logger.warning("Redis connection failed, continuing without cache: %s", exc)
            self.client = None
            self.enabled = False

    async def disconnect(self):
        if self.client:
            self.client.close()

    def _is_available(self) -> bool:
        return self.enabled and self.client is not None

    async def get(self, key: str) -> Optional[Any]:
        if not self._is_available():
            return None
        try:
            val = self.client.get(key)
            if val:
                return json.loads(val)
        except Exception as exc:
            logger.warning("Redis get failed: %s", exc)
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if not self._is_available():
            return
        try:
            self.client.set(key, json.dumps(value), ex=ttl)
        except Exception as exc:
            logger.warning("Redis set failed: %s", exc)

    async def delete(self, key: str):
        if not self._is_available():
            return
        try:
            self.client.delete(key)
        except Exception as exc:
            logger.warning("Redis delete failed: %s", exc)

    async def exists(self, key: str) -> bool:
        if not self._is_available():
            return False
        try:
            return bool(self.client.exists(key))
        except Exception as exc:
            logger.warning("Redis exists failed: %s", exc)
        return False

    async def hset(self, key: str, mapping: dict, ttl: Optional[int] = None):
        if not self._is_available():
            return
        try:
            self.client.hset(key, mapping=mapping)
            if ttl:
                self.client.expire(key, ttl)
        except Exception as exc:
            logger.warning("Redis hset failed: %s", exc)

    async def hgetall(self, key: str) -> dict:
        if not self._is_available():
            return {}
        try:
            return self.client.hgetall(key) or {}
        except Exception as exc:
            logger.warning("Redis hgetall failed: %s", exc)
        return {}

    async def hdel(self, key: str, *fields):
        if not self._is_available():
            return
        try:
            self.client.hdel(key, *fields)
        except Exception as exc:
            logger.warning("Redis hdel failed: %s", exc)


redis_client = RedisClient()
