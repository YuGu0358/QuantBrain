from alpha_miner.modules.llm_cache import LLMCache


def test_llm_cache_hashes_full_payload(tmp_path):
    cache = LLMCache(tmp_path)
    request = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "x"}], "temperature": 0.4}
    changed = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "x"}], "temperature": 0.5}
    first = cache.get(request)
    assert not first.hit
    cache.put(request, {"ok": True})
    second = cache.get(request)
    assert second.hit
    assert cache.key_for(request) != cache.key_for(changed)
