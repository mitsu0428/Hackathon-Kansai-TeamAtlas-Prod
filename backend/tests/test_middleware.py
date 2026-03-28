"""Unit tests for middleware rate-limiting logic."""

import time

import pytest

from src.api.middleware import (
    _RATE_LIMIT,
    _RATE_STORE_MAX_SIZE,
    _RATE_WINDOW,
    _is_rate_limited,
    _rate_store,
)


@pytest.fixture(autouse=True)
def _clear_rate_store():
    """Ensure rate store is empty before each test."""
    _rate_store.clear()
    yield
    _rate_store.clear()


class TestIsRateLimited:
    def test_first_request_allowed(self):
        assert _is_rate_limited("192.168.1.1") is False

    def test_under_limit_allowed(self):
        ip = "10.0.0.1"
        for _ in range(50):
            assert _is_rate_limited(ip) is False

    def test_at_limit_blocked(self):
        """After _RATE_LIMIT requests the next one should be blocked."""
        ip = "10.0.0.2"
        for _ in range(_RATE_LIMIT):
            _is_rate_limited(ip)
        assert _is_rate_limited(ip) is True

    def test_different_ips_independent(self):
        ip_a, ip_b = "1.1.1.1", "2.2.2.2"
        for _ in range(_RATE_LIMIT):
            _is_rate_limited(ip_a)
        # ip_a should be blocked, ip_b should not
        assert _is_rate_limited(ip_a) is True
        assert _is_rate_limited(ip_b) is False

    def test_expired_entries_cleaned(self, monkeypatch):
        """Timestamps older than the window should be purged."""
        ip = "10.0.0.3"
        # Manually insert old timestamps
        old_time = time.monotonic() - _RATE_WINDOW - 10
        _rate_store[ip] = [old_time + i * 0.001 for i in range(100)]

        # Next call should clean old entries and allow request
        assert _is_rate_limited(ip) is False
        # The old entries should have been removed
        assert len(_rate_store[ip]) == 1

    def test_store_max_size_guard(self):
        """When store exceeds max size, oldest IPs are evicted."""
        now = time.monotonic()
        # Fill the store beyond max size with staggered timestamps
        for i in range(_RATE_STORE_MAX_SIZE + 10):
            ip = f"192.168.{i // 256}.{i % 256}"
            _rate_store[ip] = [now - (_RATE_STORE_MAX_SIZE - i) * 0.001]

        # Use an existing IP to trigger the eviction path
        # (new IPs return early before the max size check)
        existing_ip = "192.168.0.0"
        result = _is_rate_limited(existing_ip)
        assert result is False
        # Store should have been pruned to roughly half
        assert len(_rate_store) <= _RATE_STORE_MAX_SIZE // 2 + 10
