"""SGLang client: transport errors (connection resets under load) are retried, results keep order."""

import json

import httpx
import torch

from lrt import sglang_client


def test_decode_greedy_retries_transport_errors(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        seen[body] = seen.get(body, 0) + 1
        if seen[body] == 1:                                   # every request fails once
            raise httpx.ReadError("connection reset", request=request)
        first = json.loads(body)["input_embeds"][0][0][0]
        return httpx.Response(200, json={"text": f"out{int(first)}"})

    monkeypatch.setattr(sglang_client, "_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(sglang_client.asyncio, "sleep", _no_sleep)
    prefixes = [torch.full((3, 2), float(i)) for i in range(5)]
    texts = sglang_client.decode_greedy("m", prefixes, port=1, max_new_tokens=4, max_in_flight=2)
    assert texts == [f"out{i}" for i in range(5)]
    assert all(n == 2 for n in seen.values()) and len(seen) == 5


async def _no_sleep(_):
    return None
