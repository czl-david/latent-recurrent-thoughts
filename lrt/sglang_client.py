"""Client for a separately launched SGLang server (scripts/launch_server.sh): greedy decoding through
the `input_embeds` path, and text sampling for lrt.distill. It never starts or stops the server.
The server must run with --disable-radix-cache (embedding inputs); never send text and
input_embeds requests to one server at the same time."""

import asyncio
import time
from typing import List

import httpx
import requests
import torch

GREEDY = {"temperature": 0.0, "n": 1}     # SGLang maps temperature < 1e-6 to top_k = 1 (argmax)
RETRIES = 5


def log(msg: str) -> None:
    print(f"[sglang {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_health(port: int) -> None:
    s = requests.Session()
    s.trust_env = False          # a shell proxy must never intercept localhost traffic
    r = s.get(f"http://localhost:{port}/health", timeout=5)
    assert r.status_code == 200, f"sglang on port {port} replied {r.status_code}"


def _client() -> httpx.AsyncClient:
    # No keep-alive: with ~9 MB embedding payloads the event loops on both sides block long enough
    # for the server to close a pooled connection while the client reuses it (httpx.ReadError).
    return httpx.AsyncClient(trust_env=False, limits=httpx.Limits(max_keepalive_connections=0))


async def _one(client, url, payload, idx, sem, out, timeout):
    async with sem:
        for attempt in range(RETRIES + 1):
            try:
                r = await client.post(url, json=payload, timeout=timeout)
                break
            except httpx.TransportError as exc:          # reset/refused/protocol: resend (greedy is deterministic)
                if attempt == RETRIES:
                    raise
                log(f"request {idx}: {type(exc).__name__} {exc!r}; retry {attempt + 1}/{RETRIES}")
                await asyncio.sleep(min(2 ** attempt, 30))
        r.raise_for_status()
        out[idx] = r.json()


async def _submit(model: str, prefixes: List[List[List[float]]], port: int, sampling: dict,
                  max_in_flight: int, timeout: float):
    out = [None] * len(prefixes)
    sem = asyncio.Semaphore(max_in_flight)
    url = f"http://localhost:{port}/generate"
    async with _client() as client:
        await asyncio.gather(*[
            asyncio.create_task(_one(client, url, {"model": model, "input_embeds": [p], "sampling_params": sampling},
                                     i, sem, out, timeout))
            for i, p in enumerate(prefixes)])
    return out


async def _submit_text(model: str, prompts: List[str], port: int, sampling: dict, max_in_flight: int, timeout: float):
    out = [None] * len(prompts)
    sem = asyncio.Semaphore(max_in_flight)
    url = f"http://localhost:{port}/generate"
    async with _client() as client:
        await asyncio.gather(*[
            asyncio.create_task(_one(client, url, {"model": model, "text": p, "sampling_params": sampling},
                                     i, sem, out, timeout))
            for i, p in enumerate(prompts)])
    return out


def sample_texts(model_path: str, prompts: List[str], port: int, n: int, temperature: float, top_p: float,
                 top_k: int, max_new_tokens: int, max_in_flight: int = 32, timeout: float = 3600.0) -> List[List[str]]:
    """n sampled completions per (already chat-templated) text prompt."""
    sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "n": n, "max_new_tokens": max_new_tokens}
    res = asyncio.run(_submit_text(model_path, prompts, port, sampling, max_in_flight, timeout))
    out = []
    for r in res:
        items = r if isinstance(r, list) else [r]
        out.append([it["text"] for it in items])
    return out


def decode_greedy(model_path: str, prefixes: List[torch.Tensor], port: int, max_new_tokens: int,
                  max_in_flight: int = 32, timeout: float = 3600.0) -> List[str]:
    """prefixes: per-example [L_b, d] embeddings (float lists are sent as JSON)."""
    sampling = {**GREEDY, "max_new_tokens": max_new_tokens}
    lists = [p.float().cpu().tolist() for p in prefixes]
    res = asyncio.run(_submit(model_path, lists, port, sampling, max_in_flight, timeout))
    return [(r[0] if isinstance(r, list) else r)["text"] for r in res]
