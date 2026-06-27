# Crafthost — Available Models

Inventory of every AI model on the Crafthost local AI server (RTX 3060 12 GB), organized by
**use case** and **star-ranked** for your stack: **Claude Code · Hermes · OpenClaw**.
Last updated: 2026-06-27. **25 distinct models** (+1 duplicate source GGUF), all served via
**Ollama** (port `11434`).

> **Two star systems (don't confuse them):**
> - **Category stars** (in the per-category tables) = rating *within that category*. ★★★★★ = best in class.
> - **Global stars** (in the Global Leaderboard) = rating *across all 25*. ★★★★★ = top of the whole stack.
> - **Global #** column = where each model ranks overall (1 = best). Holistic pass: capability, VRAM fit,
>   latency, and whether the model fills an irreplaceable role.

---

## 🏆 Top picks for your stack (Claude Code / Hermes / OpenClaw)

| Role | Model | Why |
|---|---|---|
| **Embeddings / RAG** | `nomic-embed-text:latest` | Only viable embedding model; foundational for retrieval |
| **Daily driver (fast)** | `gemma4-fast:latest` | MTP acceleration, agentic-tuned, fits 12 GB with room |
| **Heavy coding (opus-tier)** | `deepseek-coder-v2:16b-lite-instruct-q4_K_M` | 81 HumanEval, best raw code on host |
| **Claude-style assistant** | `qwythos:9b` ⭐ | Claude-tuned, strong tool defaults, usually loaded |
| **Deep reasoning** | `thinker14b:latest` | Best 14B-class reasoning (Qwen3-14B) |
| **Instant background (haiku)** | `qwen-haiku:4b` | Tiny/fast, co-resides with a 7–8 GB model |

---

## 🌐 Global Leaderboard (all 25, regardless of category)

| # | Stars | Model | Category | Size | Why |
|---|---|---|---|---|---|
| 1 | ★★★★★ | `nomic-embed-text:latest` | Embeddings | 274 MB | Irreplaceable — only viable embedder, ~0 VRAM, unlocks RAG for the whole stack |
| 2 | ★★★★★ | `gemma4-fast:latest` | Fast | 7.6 GB | Best utility/VRAM: MTP drafters, agentic-tuned, daily-driver workhorse |
| 3 | ★★★★★ | `deepseek-coder-v2:16b-lite-instruct-q4_K_M` | Coding | 10 GB | Best raw code (HumanEval 81), MoE-efficient; the heavy for hard tasks |
| 4 | ★★★★½ | `qwythos:9b` | Chat | 7.6 GB | Claude-tuned + strong tool defaults; plays nicest with Claude Code prompts |
| 5 | ★★★★½ | `thinker14b:latest` | Reasoning | 9.3 GB | Genuine multi-step planning depth the fast models lack |
| 6 | ★★★★½ | `qwen-haiku:4b` | Fast | 3.4 GB | Tiny + fast enough to co-reside; instant routing/sub-agent role |
| 7 | ★★★★ | `coder14b:latest` | Coding | 9.3 GB | Near-deepseek on many tasks at lower VRAM; solid coder fallback |
| 8 | ★★★★ | `gemma4-vision:latest` | Vision | 7.6 GB | Essential-unique: only image-capable model (screenshots/OCR/UI) |
| 9 | ★★★★ | `qwen3:14b` | Base | 9.3 GB | Most balanced 14B generalist; reliable default |
| 10 | ★★★★ | `hf.co/yuxinlu1/gemma-4-12B-agentic-…-tau2-GGUF:Q4_K_M` | Agentic | 7.4 GB | Only dedicated agentic tune — unique tool-chaining role |
| 11 | ★★★★ | `fast-nt:latest` | Fast | 9.3 GB | 14B quality at no-think speed; good middle gear |
| 12 | ★★★½ | `hf.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF:Q5_K_M` | Chat | 7.6 GB | 1M-ctx Claude-tuned; long-context choice, trails only on integration |
| 13 | ★★★½ | `gemma4:12b` | Base | 7.6 GB | Solid 12B generalist at efficient footprint |
| 14 | ★★★½ | `hf.co/empero-ai/Qwable-9B-Claude-Fable-5-GGUF:Q5_K_M` | Chat | 7.4 GB | Useful Claude-tuned 9B but redundant beside `qwythos:9b` |
| 15 | ★★★½ | `coder14b:8k` | Coding | 9.3 GB | VRAM-safe shrunken ctx; niche — default to full-ctx build |
| 16 | ★★★½ | `gemma4-thinker:latest` | Reasoning | 7.6 GB | Reasonable 12B reasoner, outclassed by `thinker14b` |
| 17 | ★★★ | `hf.co/yuxinlu1/gemma-4-12B-coder-…-v1-GGUF:Q4_K_M` | Coding | 7.4 GB | Competent but redundant — coder14b/gemma4-coder cover this |
| 18 | ★★★ | `gemma4-coder:latest` | Coding | 7.6 GB | Fine 12B coder; near-tie with #17 |
| 19 | ★★★ | `ravenx:12b` | Coding | 7.4 GB | Coder-agent framing; specialists above eat its lunch |
| 20 | ★★★ | `coder-nt:latest` | Coding | 9.3 GB | No-think coder; `fast-nt` is the better no-think choice |
| 21 | ★★★ | `fast14b:latest` | Fast | 9.3 GB | Outpaced by `fast-nt` / `gemma4-fast` |
| 22 | ★★★ | `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M` | Fast | 3.4 GB | Same weights as `qwen-haiku:4b`; GGUF integration lags the Ollama build |
| 23 | ★★½ | `sentence-transformers/all-MiniLM-L6-v2` | Embeddings | 88 MB | 384-dim, 256-tok cap — strictly dominated by nomic |
| 24 | ★★ | `hf.co/mradermacher/Huihui-…-abliterated-GGUF:Q4_K_M` | Coding | 7.5 GB | Abliterated — risky for agentic tool-calling; avoid |
| 25 | ★★ | `hf.co/Jackrong/Qwopus3.6-27B-…-GGUF:Q3_K_M` | Coding | 14 GB | 27B-Q3 starves 12 GB (swap thrash); doesn't fit the card |

**Tiers:** **S** = #1–4 (irreplaceable / best-in-role) · **A** = #5–11 (strong, fit cleanly) · **B** = #12–20 (good but redundant/niche) · **C** = #21–25 (dominated/risky/too big).

---

## How to call them

```bash
ollama list
ollama run qwythos:9b
curl http://localhost:11434/api/generate -d '{"model":"qwythos:9b","prompt":"Hello","stream":false}'
```
OpenAI-compatible endpoint: `http://localhost:11434/v1` (model name = the tag below).

---

## 💬 Chat & Assistant (Claude-tuned)

| # | Global # | Stars | Model | Size | Why |
|---|---|---|---|---|---|
| 1 | 4 | ★★★★ | `qwythos:9b` ⭐ | 7.6 GB | Zero cold-start (loaded), Ollama tool/template defaults |
| 2 | 12 | ★★★½ | `hf.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF:Q5_K_M` | 7.6 GB | 1M context — long agent traces; pays a swap-in cost |
| 3 | 14 | ★★★ | `hf.co/empero-ai/Qwable-9B-Claude-Fable-5-GGUF:Q5_K_M` | 7.4 GB | No ctx edge, no evidence of stronger coding |

**Top pick:** `qwythos:9b`.

## 💻 Coding

| # | Global # | Stars | Model | Size | Why |
|---|---|---|---|---|---|
| 1 | 3 | ★★★★½ | `deepseek-coder-v2:16b-lite-instruct-q4_K_M` | 10 GB | 81 HumanEval, MoE-fast, best raw code |
| 2 | 7 | ★★★★ | `coder14b:latest` | 9.3 GB | Qwen3-14B dense, strong tool-calling |
| 3 | 15 | ★★★½ | `coder14b:8k` | 9.3 GB | Shrunk ctx → safest way to run the strong base |
| 4 | 20 | ★★★ | `coder-nt:latest` | 9.3 GB | No-think cuts latency; loses depth |
| 5 | 19 | ★★★ | `ravenx:12b` | 7.4 GB | Gemma-4-12B coder-agent, roomy fit |
| 6 | 18 | ★★★ | `gemma4-coder:latest` | 7.6 GB | Solid Gemma coder, decent function calling |
| 7 | 17 | ★★★ | `hf.co/yuxinlu1/gemma-4-12B-coder-…-v1-GGUF:Q4_K_M` | 7.4 GB | Verified-pass coder FT; less agent-tuned |
| 8 | 24 | ★★½ | `hf.co/mradermacher/Huihui-…-abliterated-GGUF:Q4_K_M` | 7.5 GB | Abliteration degrades tool-calling — risky |
| 9 | 25 | ★★ | `hf.co/Jackrong/Qwopus3.6-27B-…-GGUF:Q3_K_M` | 14 GB | 27B-Q3 starves the card — slow, KV thrash |

**Top pick:** `deepseek-coder-v2`. **VRAM-safe:** `coder14b:8k`.

## 🧠 Reasoning / Thinker

| # | Global # | Stars | Model | Size | Why |
|---|---|---|---|---|---|
| 1 | 5 | ★★★★ | `thinker14b:latest` | 9.3 GB | Qwen3-14B beats Gemma-3-12B on all 15 benchmarks |
| 2 | 16 | ★★★ | `gemma4-thinker:latest` | 7.6 GB | Weaker base reasoning; 1.7 GB saving = more context |

**Top pick:** `thinker14b`.

## 🤖 Agentic

| # | Global # | Stars | Model | Size | Why |
|---|---|---|---|---|---|
| 1 | 10 | ★★★★ | `hf.co/yuxinlu1/gemma-4-12B-agentic-…-tau2-GGUF:Q4_K_M` | 7.4 GB | Only dedicated tool-use tune; unique role |

## ⚡ Fast / Lightweight

| # | Global # | Stars | Model | Size | Why |
|---|---|---|---|---|---|
| 1 | 2 | ★★★★★ | `gemma4-fast:latest` | 7.6 GB | MTP drafters, agentic-tuned, LiveCodeBench 80% |
| 2 | 6 | ★★★★ | `qwen-haiku:4b` | 3.4 GB | Punches above weight, ~2× tok/s of 14B |
| 3 | 11 | ★★★★ | `fast-nt:latest` | 9.3 GB | Lowest output latency of 14B class (no-think) |
| 4 | 21 | ★★★ | `fast14b:latest` | 9.3 GB | Same capability but emits CoT → more wall-time |
| 5 | 22 | ★★★ | `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M` | 3.4 GB | Same weights as `qwen-haiku:4b`; raw GGUF, less tuned |

**Balanced:** `gemma4-fast`. **Pure speed:** `qwen-haiku:4b`.

## 📚 General-purpose Base

| # | Global # | Stars | Model | Size | Why |
|---|---|---|---|---|---|
| 1 | 9 | ★★★★ | `qwen3:14b` | 9.3 GB | Stronger coding/reasoning + tool-calling training |
| 2 | 13 | ★★★½ | `gemma4:12b` | 7.6 GB | Leaves real context headroom; trails Qwen3 on coding |

**Top pick:** `qwen3:14b`.

## 👁️ Multimodal / Vision

| # | Global # | Stars | Model | Size | Why |
|---|---|---|---|---|---|
| 1 | 8 | ★★★★ | `gemma4-vision:latest` | 7.6 GB | Only image-capable model on the host |

## 🔎 Embeddings

| # | Global # | Stars | Model | Size | Why |
|---|---|---|---|---|---|
| 1 | 1 | ★★★★★ | `nomic-embed-text:latest` | 274 MB | 768-dim, 8K ctx, ~62 MTEB, native Ollama `/v1/embeddings` |
| 2 | 23 | ★★½ | `sentence-transformers/all-MiniLM-L6-v2` | 88 MB | 384-dim, 256-tok cap; strictly dominated by nomic |

**Top pick:** `nomic-embed-text`.

---

## Currently loaded in VRAM

- **`qwythos:9b`** — 6.46 GB, Q5_K_M, family `qwen35`, 9.2B params, ctx 4096
- VRAM: **7180 / 12288 MiB used** (4723 MiB free) — fits one small extra model, not a second 7B+
- Auto-unloads: 2026-06-28

## Storage & capacity notes

- Ollama blobs: `/usr/share/ollama/.ollama/models/` (~84 GB, deduplicated — 14B variants share one base blob).
- HF cache: `~/.cache/huggingface/hub/` — `ravenx:12b` source GGUF (6.9 GB, same weights as the Ollama build — the unranked "26th" entry) + MiniLM.
- Native `ollama.service` **intentionally disabled** — inference runs only in the Dokploy `ollama` container (VRAM ceiling rule).
- **No other runtimes:** no LM Studio, vLLM, TGI, llama.cpp, sglang, xinference, or localai.
- `langfuse-openai-proxy` (port 8000) is an adapter in front of Ollama, not a model server.
