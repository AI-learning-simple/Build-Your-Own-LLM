#!/usr/bin/env python3
"""Collect public repository metadata and render the book's ecosystem index."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets"
CATALOG = DATA / "catalog.json"
DATE = "2026-11-26"
API = "https://api.github.com"
CATEGORIES = [
    (1, "主流模型训练框架——从 PyTorch 到 Ray", ["distributed-training", "fine-tuning", "deep-learning"]),
    (2, "推理框架与模型服务——从 vLLM 到端侧运行时", ["llm-inference", "inference-engine", "model-serving"]),
    (3, "分布式与集群编排——Kubernetes、Ray 与云原生", ["kubernetes", "distributed-computing", "mlops"]),
    (4, "硬件、编译器与算子——AI 计算的底座", ["cuda", "gpu", "triton"]),
    (5, "数据工程、存储与标注——从原始数据到训练样本", ["data-engineering", "etl", "data-validation", "annotation"]),
    (6, "经典机器学习、AutoML 与科学计算", ["machine-learning", "automl", "time-series", "graph-neural-networks"]),
    (7, "模型生态、分词与压缩——开放权重和模型工程", ["large-language-models", "transformer", "quantization"]),
    (8, "检索、向量数据库与 RAG", ["retrieval-augmented-generation", "vector-database", "semantic-search"]),
    (9, "Agent 框架、记忆与工作流", ["ai-agents", "llm-agent", "multi-agent", "agent-framework"]),
    (10, "协议、工具、浏览器与执行环境", ["model-context-protocol", "browser-automation", "web-scraping"]),
    (11, "AI 编程与开发者工具", ["ai-coding", "code-generation", "coding-agent"]),
    (12, "评测、可观测性与 MLOps", ["llm-evaluation", "llm-observability", "experiment-tracking"]),
    (13, "安全、隐私与可信 AI", ["privacy-preserving-machine-learning", "guardrails", "ai-safety"]),
    (14, "计算机视觉与图像生成", ["computer-vision", "image-generation", "object-detection"]),
    (15, "视频、三维与空间智能", ["text-to-video", "3d-reconstruction", "nerf", "gaussian-splatting"]),
    (16, "语音、音频与实时交互", ["text-to-speech", "speech-recognition", "audio-generation"]),
    (17, "机器人、强化学习与仿真", ["robotics", "reinforcement-learning", "simulation"]),
    (18, "科学、医疗、地理与专业 AI", ["bioinformatics", "medical-imaging", "geospatial"]),
    (19, "本地 AI、知识管理与个人应用", ["ollama", "chatbot", "knowledge-management"]),
    (20, "低代码、业务自动化与应用开发", ["workflow-automation", "low-code", "llm-app"]),
]

SEEDS = {
    1: ["pytorch/pytorch", "jax-ml/jax", "tensorflow/tensorflow", "keras-team/keras", "ray-project/ray", "deepspeedai/DeepSpeed", "NVIDIA/Megatron-LM", "pytorch/torchtitan", "huggingface/accelerate", "huggingface/trl", "huggingface/peft", "hiyouga/LLaMA-Factory", "axolotl-ai-cloud/axolotl", "unslothai/unsloth", "volcengine/verl", "OpenRLHF/OpenRLHF", "THUDM/slime", "radixark/miles", "NVIDIA/NeMo", "Lightning-AI/pytorch-lightning"],
    2: ["vllm-project/vllm", "sgl-project/sglang", "NVIDIA/TensorRT-LLM", "ggml-org/llama.cpp", "ollama/ollama", "kvcache-ai/ktransformers", "lmdeploy/lmdeploy", "huggingface/text-generation-inference", "triton-inference-server/server", "ai-dynamo/dynamo", "BentoML/BentoML", "microsoft/onnxruntime", "openvinotoolkit/openvino", "ml-explore/mlx"],
    3: ["kubernetes/kubernetes", "ray-project/kuberay", "kubernetes-sigs/kueue", "volcano-sh/volcano", "kubeflow/trainer", "kubeflow/pipelines", "kserve/kserve", "llm-d/llm-d", "NVIDIA/gpu-operator", "kubernetes-sigs/jobset", "SchedMD/slurm"],
    4: ["triton-lang/triton", "apache/tvm", "openxla/xla", "llvm/llvm-project", "NVIDIA/cutlass", "NVIDIA/nccl", "Dao-AILab/flash-attention", "flashinfer-ai/flashinfer", "ROCm/ROCm"],
    5: ["huggingface/datasets", "apache/arrow", "apache/spark", "apache/airflow", "dagster-io/dagster", "Polars/Polars", "duckdb/duckdb", "HumanSignal/label-studio", "docling-project/docling", "Unstructured-IO/unstructured"],
    6: ["scikit-learn/scikit-learn", "dmlc/xgboost", "microsoft/LightGBM", "catboost/catboost", "optuna/optuna", "autogluon/autogluon", "numpy/numpy", "scipy/scipy"],
    7: ["huggingface/transformers", "huggingface/tokenizers", "google/sentencepiece", "ggml-org/ggml", "bitsandbytes-foundation/bitsandbytes"],
    8: ["milvus-io/milvus", "qdrant/qdrant", "weaviate/weaviate", "chroma-core/chroma", "pgvector/pgvector", "facebookresearch/faiss", "run-llama/llama_index", "infiniflow/ragflow", "deepset-ai/haystack"],
    9: ["langchain-ai/langchain", "langchain-ai/langgraph", "crewAIInc/crewAI", "microsoft/autogen", "microsoft/agent-framework", "openai/openai-agents-python", "google/adk-python", "pydantic/pydantic-ai", "agno-agi/agno", "mem0ai/mem0", "letta-ai/letta"],
    10: ["modelcontextprotocol/python-sdk", "modelcontextprotocol/typescript-sdk", "a2aproject/A2A", "ag-ui-protocol/ag-ui", "browser-use/browser-use", "microsoft/playwright", "firecrawl/firecrawl", "unclecode/crawl4ai", "e2b-dev/E2B"],
    11: ["openai/codex", "anthropics/claude-code", "google-gemini/gemini-cli", "cline/cline", "Aider-AI/aider", "All-Hands-AI/OpenHands", "continuedev/continue", "sst/opencode"],
    12: ["mlflow/mlflow", "wandb/wandb", "langfuse/langfuse", "Arize-ai/phoenix", "explodinggradients/ragas", "confident-ai/deepeval", "EleutherAI/lm-evaluation-harness", "open-compass/opencompass", "openai/evals"],
    13: ["NVIDIA/NeMo-Guardrails", "guardrails-ai/guardrails", "microsoft/presidio", "open-policy-agent/opa", "gitleaks/gitleaks", "pytorch/opacus", "adap/flower"],
    14: ["huggingface/diffusers", "comfyanonymous/ComfyUI", "AUTOMATIC1111/stable-diffusion-webui", "ultralytics/ultralytics", "open-mmlab/mmdetection", "facebookresearch/sam2", "opencv/opencv"],
    15: ["nerfstudio-project/nerfstudio", "graphdeco-inria/gaussian-splatting", "NVlabs/instant-ngp", "Wan-Video/Wan2.1", "hpcaitech/Open-Sora", "Tencent-Hunyuan/HunyuanVideo", "microsoft/TRELLIS"],
    16: ["openai/whisper", "SYSTRAN/faster-whisper", "FunAudioLLM/CosyVoice", "fishaudio/fish-speech", "pipecat-ai/pipecat", "livekit/agents", "coqui-ai/TTS"],
    17: ["huggingface/lerobot", "google-deepmind/mujoco", "isaac-sim/IsaacLab", "Genesis-Embodied-AI/Genesis", "Farama-Foundation/Gymnasium", "DLR-RM/stable-baselines3", "ros2/ros2", "Physical-Intelligence/openpi"],
    18: ["google-deepmind/alphafold", "MONAI/MONAI", "Project-MONAI/MONAI", "microsoft/torchgeo", "deepchem/deepchem", "RDKit/rdkit", "biopython/biopython"],
    19: ["open-webui/open-webui", "Mintplex-Labs/anything-llm", "labring/FastGPT", "CherryHQ/cherry-studio", "janhq/jan", "nomic-ai/gpt4all", "khoj-ai/khoj", "obsidianmd/obsidian-releases"],
    20: ["langgenius/dify", "FlowiseAI/Flowise", "langflow-ai/langflow", "n8n-io/n8n", "vercel/ai", "streamlit/streamlit", "gradio-app/gradio", "microsoft/semantic-kernel"],
}
SEEDS[1] += ["NVIDIA-NeMo/Megatron-Bridge", "NVIDIA-NeMo/RL", "hiyouga/LlamaFactory", "verl-project/verl"]
SEEDS[2] += ["InternLM/lmdeploy", "microsoft/onnxruntime", "pytorch/executorch"]
SEEDS[3] += ["kuberay-project/kuberay"]
SEEDS[4] += ["ROCm/hip", "iree-org/iree", "NVIDIA/TransformerEngine", "pytorch/gloo", "NVIDIA/cuda-python", "gpuweb/gpuweb", "KhronosGroup/SYCL-Docs", "oneapi-src/oneDNN", "intel/llvm"]
SEEDS[5] += ["pola-rs/polars"]
SEEDS[6] += ["lightgbm-org/LightGBM"]
SEEDS[9] += ["deepseek-ai/dsh-superagent-prototype"]
SEEDS[11] += ["OpenHands/OpenHands", "anomalyco/opencode"]
SEEDS[12] += ["vibrantlabsai/ragas"]
SEEDS[13] += ["NVIDIA-NeMo/Guardrails", "data-privacy-stack/presidio", "meta-pytorch/opacus", "flwrlabs/flower"]
SEEDS[14] += ["Comfy-Org/ComfyUI"]
SEEDS[16] += ["QwenAudio/CosyVoice", "NVIDIA-NeMo/Speech"]
SEEDS[18] += ["torchgeo/torchgeo"]

EXCLUDE_NAME = re.compile(r"(^|[-_])(awesome|books?|courses?|tutorials?|cheatsheets?|interview|roadmap|resources?|papers?|reading|curriculum|exercises|100-days|learning-path|prompts)([-_]|$)", re.I)
EXCLUDE_TEXT = re.compile(r"(?:curated (?:list|collection)|list of (?:awesome|resources|papers)|jailbreak|pornograph|nsfw|deepnude|weapon|malware|ransomware|phishing|politic|election|military|face.swap|deepfake)", re.I)
PINNED = {repo.casefold(): category for category, repos in SEEDS.items() for repo in repos}
FIELDS = ["id", "full_name", "html_url", "description", "homepage", "stargazers_count", "forks_count", "language", "license", "archived", "disabled", "fork", "created_at", "updated_at", "pushed_at", "topics", "size", "default_branch"]


def request_json(path, params=None):
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "Build-Your-Own-LLM-ecosystem-catalog", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                payload = json.load(response)
                return payload, {"url": url, "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "http_date": response.headers.get("Date"), "etag": response.headers.get("ETag")}
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 429, 502, 503, 504) or attempt == 3:
                raise
            retry = exc.headers.get("Retry-After")
            if exc.code in (403, 429):
                if exc.headers.get("X-RateLimit-Remaining") == "0":
                    reset = float(exc.headers.get("X-RateLimit-Reset", time.time() + 65))
                    wait = max(5, reset - time.time() + 2)
                    if wait > 120:
                        raise
                else:
                    wait = float(retry or 65)
            else:
                wait = 4 * (attempt + 1)
            print("Retry", exc.code, "in", round(wait), "seconds", flush=True)
            time.sleep(wait)
        except (TimeoutError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Request failed")


def eligible(item, pinned=False):
    name = item.get("name", item["full_name"].split("/")[-1])
    description = item.get("description") or ""
    if item.get("fork") or item.get("disabled"):
        return False, "fork-or-disabled"
    if EXCLUDE_TEXT.search(name + " " + description):
        return False, "out-of-scope-content"
    if not pinned and (EXCLUDE_NAME.search(name) or not item.get("language") or item.get("size", 0) < 30):
        return False, "resource-list-or-no-code"
    if not pinned and item.get("archived"):
        return False, "archived"
    return True, "explicit-anchor" if pinned else "topic-stars-recency"


def save(data):
    CATALOG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_entry(previous, item, category, evidence, reason):
    entry = dict(previous or {})
    entry.update({key: item.get(key) for key in FIELDS})
    entry.update(category=PINNED.get(item["full_name"].casefold(), entry.get("category", category)),
                 discovered_categories=sorted(set(entry.get("discovered_categories", [])) | {category}),
                 evidence_urls=sorted(set(entry.get("evidence_urls", [])) | {evidence["url"]}),
                 fetched_at=evidence["fetched_at"], inclusion=reason, review=entry.get("review", "metadata-only"))
    return entry


def collect(limit_queries=None, seed_only=False, refresh=False):
    DATA.mkdir(exist_ok=True)
    data = json.loads(CATALOG.read_text()) if CATALOG.exists() else {
        "schema_version": 1, "editorial_date": DATE, "method": {"stars_minimum": 300, "pushed_since": "2025-01-01", "search_sort": "stars desc", "search_per_topic": 100, "count_unit": "canonical GitHub repository id", "notes": "Metadata snapshot, not README review, security audit or functional testing."},
        "queries": [], "repositories": [], "excluded": [], "errors": [],
    }
    repos = {item["id"]: item for item in data["repositories"]}
    completed = set() if refresh else {item["query"] for item in data["queries"] if item.get("incomplete_results") is False}
    data["collection_mode"] = "refresh" if refresh else "resume"
    count = 0
    for category, title, topics in CATEGORIES:
        if seed_only:
            break
        for topic in topics:
            query = "topic:" + topic + " stars:>=300 archived:false fork:false pushed:>=2025-01-01"
            if query in completed:
                continue
            if limit_queries is not None and count >= limit_queries:
                return
            count += 1
            try:
                payload, evidence = request_json("/search/repositories", {"q": query, "sort": "stars", "order": "desc", "per_page": 100})
                evidence.update(query=query, category=category, total_count=payload.get("total_count"), incomplete_results=payload.get("incomplete_results"), returned_ids=[x["id"] for x in payload.get("items", [])])
                data["queries"].append(evidence)
                for item in payload.get("items", []):
                    pinned = item["full_name"].casefold() in PINNED
                    ok, reason = eligible(item, pinned)
                    if not ok:
                        data["excluded"].append({"id": item["id"], "full_name": item["full_name"], "reason": reason, "query": query})
                        continue
                    repos[item["id"]] = merge_entry(repos.get(item["id"]), item, category, evidence, reason)
                data["repositories"] = list(repos.values())
                save(data)
                print(f"{category:02d} {topic}: {len(payload.get('items', []))} returned; {len(repos)} unique", flush=True)
            except Exception as exc:
                data["errors"].append({"query": query, "error": str(exc), "at": dt.datetime.now(dt.timezone.utc).isoformat()})
                save(data)
                print("ERROR", topic, str(exc), flush=True)
            time.sleep(7)
    names = {item["full_name"].casefold() for item in repos.values()}
    alias_map = {k.casefold(): v for k, v in json.loads((DATA / "curation.json").read_text()).get("aliases", {}).items()}
    for category, seeds in SEEDS.items():
        canonical_seeds = sorted({alias_map.get(name.casefold(), name) for name in seeds})
        missing = [name for name in canonical_seeds if refresh or name.casefold() not in names]
        for offset in range(0, len(missing), 5):
            batch = missing[offset:offset + 5]
            query = " ".join("repo:" + name for name in batch)
            try:
                payload, evidence = request_json("/search/repositories", {"q": query, "per_page": 100})
                evidence.update(query=query, category=category, total_count=payload.get("total_count"), incomplete_results=payload.get("incomplete_results"), returned_ids=[x["id"] for x in payload.get("items", [])], purpose="explicit-anchors")
                data["queries"].append(evidence)
                for item in payload.get("items", []):
                    ok, reason = eligible(item, True)
                    if not ok:
                        continue
                    repos[item["id"]] = merge_entry(repos.get(item["id"]), item, category, evidence, reason)
                    names.add(item["full_name"].casefold())
                    print("ANCHOR", item["full_name"], flush=True)
                data["repositories"] = list(repos.values())
                save(data)
            except Exception as exc:
                data["errors"].append({"query": query, "error": str(exc)})
                save(data)
                print("ANCHOR ERROR", str(exc), flush=True)
                if "403" in str(exc) or "429" in str(exc):
                    return
            time.sleep(7)


TOPIC_WEIGHTS = {topic: (1 if topic in {"deep-learning", "machine-learning", "gpu", "cuda", "transformer", "chatbot", "simulation"} else 5) for _, _, topics in CATEGORIES for topic in topics}
CATEGORY_KEYWORDS = {
    1: r"distributed.train|fine.?tun|pre.?train|training.framework|training.library|post.?train|rlhf|grpo|fsdp|deepspeed|megatron|lora",
    2: r"inference.engine|inference.server|inference.framework|model.serving|llm.serving|llm.inference|llama.cpp|onnx.runtime",
    3: r"kubernetes|container.orchestrat|cluster.scheduler|service.mesh|gitops|cloud.native",
    4: r"gpu.kernel|cuda.kernel|gpu.compiler|deep.learning.compiler|tensor.compiler|flash.attention|collective.communication|matrix.multiplication",
    5: r"data.pipeline|data.quality|data.validation|data.label|data.annotation|data.warehouse|data.lake|document.pars|pdf.pars|extract.*transform.*load",
    6: r"time.series|forecasting|automl|tabular|gradient.boost|statistical|scientific.comput|graph.neural|bayesian",
    7: r"tokeniz|language.model.architecture|model.quantiz|quantization|model.compression|model.merg|model.prun|large.language.model.repository",
    8: r"retrieval.augmented|vector.database|semantic.search|retrieval.framework|rag.framework|embedding.model|knowledge.graph",
    9: r"agent.framework|agentic.framework|multi.agent.framework|multi.agent.system|agent.memory|agent.orchestrat|build.*ai.agents",
    10: r"mcp.server|mcp.client|model.context.protocol|browser.automation|web.scrap|web.crawl|sandbox|web.browsing",
    11: r"coding.agent|code.assistant|coding.assistant|code.editor|code.generation|code.review|ai.coding|software.engineer.*agent|code.intelligence",
    12: r"evaluat|benchmark|observability|experiment.tracking|llm.trac|llm.monitor|model.monitor",
    13: r"privacy.preserving|differential.privacy|guardrail|pii|secret.detect|policy.engine|federated.learning|model.safety",
    14: r"image.generation|text.to.image|object.detection|image.segment|computer.vision|image.edit|image.restor|image.classif|vision.library",
    15: r"video.generat|text.to.video|image.to.video|3d.reconstruct|gaussian.splatt|neural.radiance|nerf|3d.generat|spatial.intelligence",
    16: r"text.to.speech|speech.to.text|speech.recog|speech.synth|voice.agent|audio.generat|music.generat|voice.clon|audio.process",
    17: r"robot|reinforcement.learning|physics.simulat|physics.engine|motion.planning|embodied",
    18: r"bioinformat|genomic|protein|medical.imag|medical.segment|geospatial|geograph|molecular|cheminformat|healthcare|quantitative.finance",
    19: r"personal.assistant|local.ai|knowledge.management|note.taking|meeting.note|second.brain|chatgpt.desktop|chat.interface|chat.ui",
    20: r"low.code|no.code|workflow.automation|business.process|internal.tools|build.*web.app|data.app|application.framework",
}
EDUCATION = re.compile(r"(?:for.beginners|from.scratch|engineering.guide|prompt.engineering.guide|made.with.ml|llm.action|hands.on|dive.into|learning.notes|interview|cheatsheet|roadmap|tutorial|course|curriculum|resources.for|curated.collection|a.collection.of.*(?:papers|resources)|learn.how.to|\\d+ weeks.*lessons|学习资料|学习笔记|教程|课程|资源汇总)", re.I)
OUT_OF_SCOPE = re.compile(r"(?:crypto(?:currency)?|bitcoin|miner|mining.pool|password.crack|game.cheat|bypass.*(?:paywall|captcha)|captcha.*bypass|government|military|politic|election|porn|nsfw|deepfake|face.swap|prompt.injection.attack|jailbreak|ransom|malware|phishing)", re.I)
ALIASES = {"hiyouga/llamafactory": 1, "verl-project/verl": 1, "nvidia-nemo/speech": 16, "ggml-org/llama.cpp": 2, "openhands/openhands": 11, "anomalyco/opencode": 11, "langchain-ai/deepagents": 9}


def curate():
    data = json.loads(CATALOG.read_text())
    data.setdefault("candidates", {str(r["id"]): r for r in data["repositories"]})
    for r in data["repositories"]:
        data["candidates"][str(r["id"])] = r
    for query in data["queries"]:
        for repo_id in query.get("returned_ids", []):
            candidate = data["candidates"].get(str(repo_id))
            if candidate is not None:
                candidate["evidence_urls"] = sorted(set(candidate["evidence_urls"]) | {query["url"]})
                candidate["discovered_categories"] = sorted(set(candidate["discovered_categories"]) | {query["category"]})
    overrides = json.loads((DATA / "curation.json").read_text())
    exclusions = {k.casefold(): v for k, v in overrides["exclude"].items()}
    moves = {k.casefold(): v for k, v in overrides["move"].items()}
    accepted = []
    rejected = []
    for original in data["candidates"].values():
        r = dict(original)
        name = r["full_name"].casefold()
        text = name + " " + (r.get("description") or "")
        pinned = name in PINNED or name in ALIASES
        topics = set(r.get("topics") or [])
        reason = None
        if name in exclusions:
            reason = exclusions[name]
        elif OUT_OF_SCOPE.search(text):
            reason = "outside-editorial-scope"
        elif not pinned and (EDUCATION.search(text) or topics.intersection({"awesome-list", "awesome", "tutorial", "interview-questions", "cheatsheet"})):
            reason = "learning-or-resource-material-not-counted"
        elif not pinned and re.search(r"(^|[-_/])(templates?|boilerplates?|examples?|papers?|lists?)([-_/]|$)", name):
            reason = "template-or-resource-not-counted"
        elif "annotation" in topics and not re.search(r"label|image|vision|dataset|text|document|data.annotation|annotat.*data", text, re.I):
            reason = "programming-annotation-not-data-labeling"
        if reason:
            rejected.append({"id": r["id"], "full_name": r["full_name"], "reason": reason})
            continue
        scores = {c: sum(TOPIC_WEIGHTS[t] for t in topic_list if t in topics) for c, _, topic_list in CATEGORIES}
        for c, pattern in CATEGORY_KEYWORDS.items():
            if re.search(pattern, text, re.I):
                scores[c] += 7
        if name in moves:
            category = moves[name]
        elif pinned:
            category = ALIASES.get(name, PINNED.get(name))
        else:
            category = max(scores, key=lambda c: (scores[c], c == r["category"], -c))
        if not pinned and name not in moves and category in {1, 4, 7, 17} and scores[category] < 7:
            rejected.append({"id": r["id"], "full_name": r["full_name"], "reason": "insufficient-specific-category-evidence"})
            continue
        r.update(category=category, classification="anchor" if pinned else ("editorial-override" if name in moves else "topic-and-description-rules"), classification_score=scores[category])
        accepted.append(r)
    selected = []
    for category, _, _ in CATEGORIES:
        group = sorted((r for r in accepted if r["category"] == category), key=lambda r: (r["classification"] != "anchor", -r["stargazers_count"], r["full_name"].casefold()))
        selected.extend(group[:100])
        rejected.extend({"id": r["id"], "full_name": r["full_name"], "reason": "chapter-cap-100"} for r in group[100:])
    data["repositories"] = selected
    data["curation"] = {"per_chapter_cap": 100, "candidate_count": len(data["candidates"]), "selected_count": len(selected), "rejected": rejected, "at": dt.datetime.now(dt.timezone.utc).isoformat()}
    save(data)
    validate(data)
    for category, _, _ in CATEGORIES:
        print(category, sum(r["category"] == category for r in selected))


def cell(text):
    clean = re.sub(r"[\U0001f000-\U0001ffff\u2600-\u27bf\ufe0f\u200d]", "", str(text or "未提供"))
    return re.sub(r"\s+", " ", clean).replace("|", "&#124;").replace("<", "&lt;").replace(">", "&gt;").replace("[", "&#91;").replace("]", "&#93;").replace("`", "'").strip()


def filename(number, title):
    return f"{number}. {title}.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(data):
    require(data.get("schema_version") == 1, "Unsupported catalog schema")
    repos = data.get("repositories")
    require(isinstance(repos, list) and len(repos) >= 1000, "At least 1000 repositories required")
    for r in repos:
        require(isinstance(r, dict), "Repository must be an object")
        for key in ("id", "category", "stargazers_count"):
            require(type(r.get(key)) is int, "Invalid integer field: " + key)
        require(r["category"] in range(1, 21), "Invalid category")
        require(r["id"] > 0 and r["stargazers_count"] >= 0, "Invalid numeric value")
        require(isinstance(r.get("full_name"), str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", r["full_name"]), "Invalid repository name")
        require(r.get("html_url") == "https://github.com/" + r["full_name"], "Invalid canonical URL")
        require(type(r.get("archived")) is bool, "Invalid archived flag")
        require(r.get("description") is None or isinstance(r["description"], str), "Invalid description")
        require(r.get("license") is None or isinstance(r["license"], dict), "Invalid license")
        require(r.get("review") == "metadata-only", "Unsupported review classification")
        require(isinstance(r.get("evidence_urls"), list) and r["evidence_urls"], "Missing evidence")
        require(all(isinstance(u, str) and u.startswith(API + "/") for u in r["evidence_urls"]), "Invalid evidence URL")
        dt.datetime.fromisoformat(r["fetched_at"].replace("Z", "+00:00"))
        require(isinstance(r.get("pushed_at"), str), "Missing push date")
    require(len({r['id'] for r in repos}) == len(repos), "Duplicate repository ids")
    require(len({r['full_name'].casefold() for r in repos}) == len(repos), "Duplicate names")
    print("Validated", len(repos), "unique repositories")


def replace_block(text, name, replacement):
    start, end = "<!-- " + name + ":START -->", "<!-- " + name + ":END -->"
    require(text.count(start) == 1 and text.count(end) == 1, "Expected exactly one " + name + " block")
    require(text.index(start) < text.index(end), "Reversed block markers")
    updated, count = re.subn(re.escape(start) + r".*?" + re.escape(end), lambda _: replacement, text, flags=re.S)
    require(count == 1, "Block was not replaced")
    return updated


def project_row(r):
    license_name = (r.get("license") or {}).get("spdx_id") or "待核验"
    status = "已归档，历史参考" if r["archived"] else "未归档"
    return f"| G{r['id']} | [{cell(r['full_name'])}]({r['html_url']}) | {cell(r['description'])} | {r['stargazers_count']:,} | {r['pushed_at'][:10]}；{status} | {cell(license_name)} |"


def build_summary(data):
    repos = data["repositories"]
    names = {r["full_name"].casefold() for r in repos}
    aliases = {k.casefold(): v.casefold() for k, v in json.loads((DATA / "curation.json").read_text()).get("aliases", {}).items()}
    unresolved = sorted({name for group in SEEDS.values() for name in group if aliases.get(name.casefold(), name.casefold()) not in names})
    latest_queries = {q["query"]: q for q in data["queries"]}
    return {"editorial_date": DATE, "unique_repositories": len(repos), "candidate_count": len(data.get("candidates", repos)), "by_chapter": {str(n): sum(r["category"] == n for r in repos) for n, _, _ in CATEGORIES}, "query_records": len(data["queries"]), "historical_errors": len(data.get("errors", [])), "incomplete_latest_queries": [q for q, result in latest_queries.items() if result.get("incomplete_results")], "unresolved_seed_names": unresolved, "unresolved_meaning": "Names not mapped to selected canonical records; may be renamed, unavailable, excluded or unindexed. Not evidence of project nonexistence.", "archived_anchors": sum(r["archived"] for r in repos), "metadata_only": len(repos), "fetched_min": min(r["fetched_at"] for r in repos), "fetched_max": max(r["fetched_at"] for r in repos)}


def render():
    data = json.loads(CATALOG.read_text())
    validate(data)
    repos = data["repositories"]
    pending = {}
    for category, title, _ in CATEGORIES:
        path = ROOT / filename(category, title)
        if not path.exists():
            raise FileNotFoundError("Author the chapter before rendering: " + str(path))
        original = path.read_text()
        selected = sorted((r for r in repos if r["category"] == category), key=lambda r: (-r["stargazers_count"], r["full_name"].casefold()))
        rows = [f"## 本章项目索引（{len(selected)} 个去重软件仓库）", "", "本表是 GitHub 公开元数据快照，不等于逐项实测或推荐。英文短说明保留仓库自述以避免机器翻译改变功能含义；分类采用一个主类别，跨领域标签见 `assets/catalog.json`。★ 为采集时 Star 数，不代表性能、下载量或采用率。许可字段为空、`NOASSERTION` 或 `Other` 时必须人工阅读许可文件；仓库许可也不替代权重、数据或商用条款。", "", "| 编号 | 项目与官方仓库 | 定位（仓库自述） | ★快照 | 最近推送／状态 | 代码许可 |", "|---|---|---|---:|---|---|"]
        rows.extend(project_row(r) for r in selected)
        rows += ["", "仅在一个主章节计数；没有收录不代表不重要。`pushed_at` 是仓库推送时间，不等于稳定版本发布日期或维护质量。本章表格由正式脚本 `assets/catalog.py` 从已保存证据生成。"]
        block = "<!-- CATALOG:START -->\n" + "\n".join(rows) + "\n<!-- CATALOG:END -->"
        pending[path] = replace_block(original, "CATALOG", block)
    tree_path = ROOT / "23. AI 生态全景树——从硬件到应用的总索引.md"
    original = tree_path.read_text()
    lines = ["## 全量项目树", "", f"以下树覆盖本版 **{len(repos)} 个去重软件仓库**。编号与各章表格、JSON 一一对应；每项只在一个主分支出现。树按本书检索习惯组织，不宣称穷尽未来所有 AI 范畴。", "", "```text", "AI 软件项目索引"]
    for category, title, _ in CATEGORIES:
        selected = sorted((r for r in repos if r["category"] == category), key=lambda r: r["full_name"].casefold())
        final = category == len(CATEGORIES)
        lines.append(("└── " if final else "├── ") + f"{category}. {title}（{len(selected)}）")
        for index, r in enumerate(selected):
            lines.append(("    " if final else "│   ") + ("└── " if index == len(selected) - 1 else "├── ") + f"G{r['id']} {r['full_name']}")
    lines += ["```", "", "### 分章跳转", ""]
    lines += [f"- [{n}. {title}](<{filename(n, title)}>)" for n, title, _ in CATEGORIES]
    block = "<!-- TREE:START -->\n" + "\n".join(lines) + "\n<!-- TREE:END -->"
    pending[tree_path] = replace_block(original, "TREE", block)
    summary = build_summary(data)
    pending[DATA / "summary.json"] = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    for path, content in pending.items():
        path.write_text(content, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_documents(data):
    chapters = sorted(ROOT.glob("[0-9]*. *.md"), key=lambda p: int(p.name.split(".", 1)[0]))
    require([int(p.name.split(".", 1)[0]) for p in chapters] == list(range(1, 24)), "Expected chapters 1 through 23")
    guide = ROOT / "导读——如何使用本附录.md"
    require(guide.is_file(), "Missing unnumbered reading guide")
    docs = [guide, *chapters]
    require(set(ROOT.glob("*.md")) == set(docs), "Unexpected appendix documents")
    ids = []
    for path in docs:
        text = path.read_text()
        require(len(re.findall(r"(?m)^# ", text)) == 1, path.name + ": H1")
        require(DATE in text, path.name + ": missing editorial date")
        require(text.count("```") % 2 == 0, path.name + ": code fences")
        for target in re.findall(r"\]\(<([^>]+)>\)", text):
            if not target.startswith(("http://", "https://", "#")):
                require((path.parent / urllib.parse.unquote(target)).exists(), target)
        for block in re.findall(r"(?:^\|.*\n)+", text, re.M):
            require(len({line.count("|") for line in block.splitlines()}) == 1, path.name + ": table columns")
        chapter = 0 if path == guide else int(path.name.split(".", 1)[0])
        if 1 <= chapter <= 20:
            rows = [line for line in text.splitlines() if re.match(r"^\| G\d+ \|", line)]
            selected = sorted((r for r in data["repositories"] if r["category"] == chapter), key=lambda r: (-r["stargazers_count"], r["full_name"].casefold()))
            require(rows == [project_row(r) for r in selected], path.name + ": stale or misplaced rows")
            replace_block(text, "CATALOG", "")
        ids.extend(int(x) for x in re.findall(r"(?m)^\| G(\d+) \|", text))
    expected = {r["id"] for r in data["repositories"]}
    require(len(ids) == len(expected) and set(ids) == expected, "Catalog rows do not match JSON")
    tree = (ROOT / "23. AI 生态全景树——从硬件到应用的总索引.md").read_text()
    replace_block(tree, "TREE", "")
    pairs = re.findall(r"[├└]── G(\d+) ([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", tree)
    expected_pairs = [(str(r["id"]), r["full_name"]) for category, _, _ in CATEGORIES for r in sorted((r for r in data["repositories"] if r["category"] == category), key=lambda r: r["full_name"].casefold())]
    require(pairs == expected_pairs, "Tree name/order mismatch")
    headings = re.findall(r"(?m)^[├└]── (\d+)\. (.+)（(\d+)）$", tree)
    expected_headings = [(str(n), title, str(sum(r["category"] == n for r in data["repositories"]))) for n, title, _ in CATEGORIES]
    require(headings == expected_headings, "Tree chapter headings mismatch")
    require(json.loads((DATA / "summary.json").read_text()) == build_summary(data), "Stale summary")
    print("Validated 23 chapters and reading guide, local links, exact catalog rows, summary and complete tree")


def audit_sources(refresh=False):
    import concurrent.futures
    import html

    audit_path = DATA / "source_audit.json"
    previous = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    cached = {} if refresh else {r["url"]: r for r in previous.get("sources", [])}
    urls = set()
    for path in ROOT.glob("*.md"):
        text = re.sub(r"<!-- CATALOG:START -->.*?<!-- CATALOG:END -->", "", path.read_text(), flags=re.S)
        urls.update(re.findall(r"\]\((https?://[^\s)]+)\)", text))
    urls = sorted(u for u in urls if urllib.parse.urlparse(u).hostname not in {"github.com", "api.github.com"})

    def inspect(url):
        if url in cached:
            result = dict(cached[url])
            if result.get("status") in (301, 302, 303, 307, 308):
                result["access"] = "redirect-not-followed"
            return result
        result = {"url": url, "checked_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Build-Your-Own-LLM-source-audit"})
            with urllib.request.urlopen(request, timeout=15) as response:
                raw = response.read(700000)
                page = raw.decode("utf-8", errors="replace")
                title = re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I)
                result.update(status=response.status, final_url=response.url, http_date=response.headers.get("Date"), title=html.unescape(re.sub(r"\s+", " ", title.group(1)).strip()) if title else None)
                result["access"] = "challenge-or-login" if re.search(r"just a moment|verify you are human|access denied|enable javascript|sign in", result.get("title") or "", re.I) else "page-response"
        except Exception as exc:
            code = getattr(exc, "code", None)
            result.update(status=code, access="redirect-not-followed" if code in (301, 302, 303, 307, 308) else "unavailable", error=str(exc))
            if isinstance(exc, urllib.error.HTTPError) and exc.headers.get("Location"):
                result["redirect_url"] = urllib.parse.urljoin(url, exc.headers["Location"])
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(inspect, urls))
    payload = {"editorial_date": DATE, "meaning": "HTTP response/title checks only; not feature, ownership, availability or scientific verification.", "sources": results}
    (DATA / "source_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Sources", len(results), "page responses", sum(r["access"] == "page-response" for r in results))
    for r in results:
        if r["access"] != "page-response":
            print(r.get("status"), r["url"], r["access"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["collect", "curate", "render", "validate", "audit"])
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.action == "collect":
        collect(args.limit_queries, args.seed_only, args.refresh)
    elif args.action == "curate":
        curate()
    elif args.action == "render":
        render()
    elif args.action == "audit":
        audit_sources(args.refresh)
    else:
        data = json.loads(CATALOG.read_text())
        validate(data)
        validate_documents(data)


if __name__ == "__main__":
    main()
