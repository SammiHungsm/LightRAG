import os
import sys
import json
import asyncio
import numpy as np
import docker
import networkx as nx
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

from numpy.linalg import norm

# LightRAG 核心
from lightrag.base import QueryParam
from lightrag.utils import logger, EmbeddingFunc
from lightrag.api.config import global_args 
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.lightrag import LightRAG
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.routers.query_routes import QueryRequest, QueryResponse 

# Nanobot 核心
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.agent.tools.base import Tool
from nanobot.agent.skills import SkillsLoader

router = APIRouter(tags=["query"])

# ==========================================
# 🌟 全局儲存庫路徑設定
# ==========================================
RAG_STORAGE_NAME = os.getenv("RAG_STORAGE_NAME", "rag_storage")
BASE_STORAGE_PATH = Path(f"/app/data/{RAG_STORAGE_NAME}")

# ==========================================
# 1. 基礎設施：Embedding 與 RAG 緩存
# ==========================================
_rag_cache: Dict[str, LightRAG] = {}

def _get_dedicated_embedding():
    async def embed_wrapper(texts: list[str]) -> np.ndarray:
        return await openai_embed.func(
            texts=texts, 
            model=global_args.embedding_model,
            api_key=global_args.llm_binding_api_key, 
            base_url=global_args.llm_binding_host
        )
    embed_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    return EmbeddingFunc(embedding_dim=embed_dim, max_token_size=8192, func=embed_wrapper)

def _get_dedicated_llm():
    async def llm_wrapper(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await openai_complete_if_cache(
            model=global_args.llm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=global_args.llm_binding_api_key,
            base_url=global_args.llm_binding_host,
            **kwargs
        )
    return llm_wrapper

async def get_rag_instance(ws_path_from_ui: str) -> LightRAG:
    if RAG_STORAGE_NAME in ws_path_from_ui or "data/" in ws_path_from_ui:
        full_path = str(Path(ws_path_from_ui).absolute())
    else:
        full_path = str((BASE_STORAGE_PATH / ws_path_from_ui.lstrip("/")).absolute())
    
    if full_path not in _rag_cache:
        logger.info(f"🧬 正在初始化物理路徑: {full_path}")
        rag = LightRAG(
            working_dir=full_path,
            llm_model_func=_get_dedicated_llm(),
            embedding_func=_get_dedicated_embedding()
        )
        await rag.initialize_storages()
        _rag_cache[full_path] = rag
        
    return _rag_cache[full_path]

# ==========================================
# 2. Agent 工具集 
# ==========================================
class WorkspaceDiscoveryTool(Tool):
    name = "list_available_workspaces"
    description = "掃描系統內所有已建庫的資料夾，返回 index 和 financial_report 類別及其可用年份。"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, base_path: Path):
        super().__init__()
        self.base_path = base_path

    async def execute(self, **kwargs) -> str:
        structure = {"index": [], "financial_report": {}}
        for cat in ["index", "financial_report"]:
            cat_path = self.base_path / cat
            if not cat_path.exists(): continue
            if cat == "index":
                structure[cat] = [f.name for f in cat_path.iterdir() if f.is_dir()]
            else:
                for company in cat_path.iterdir():
                    if company.is_dir():
                        structure["financial_report"][company.name] = [y.name for y in company.iterdir() if y.is_dir()]
        return json.dumps(structure, indent=2, ensure_ascii=False)

class FinancialRAGTool(Tool):
    name = "access_financial_data"
    description = "訪問特定知識庫獲取數據。需提供 category, workspace_name, query, year(財報必填)。"
    parameters = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": ["index", "financial_report"]},
            "workspace_name": {"type": "string"},
            "year": {"type": "string"},
            "query": {"type": "string"}
        },
        "required": ["category", "workspace_name", "query"]
    }

    def __init__(self, base_path: Path, refs_list: list):
        super().__init__()
        self.base_path = base_path
        self.refs_list = refs_list

    async def execute(self, category: str, workspace_name: str, query: str, year: str = None, **kwargs) -> str:
        if category == "index":
            target_dir = self.base_path / "index" / workspace_name
        else:
            if not year: return "Error: year is required for financial reports."
            target_dir = self.base_path / "financial_report" / workspace_name / year

        if not target_dir.exists(): return f"Error: Workspace {workspace_name} not found."
        
        try:
            rag = await get_rag_instance(str(target_dir))
            param = QueryParam(mode="mix", stream=False, top_k=60)
            
            if hasattr(rag, "aquery_llm"):
                result = await rag.aquery_llm(query, param=param)
            else:
                result = await rag.aquery(query, param=param)
                
            if isinstance(result, dict):
                if "data" in result and "references" in result["data"]:
                    self.refs_list.extend(result["data"]["references"])
                content = result.get("llm_response", {}).get("content") or result.get("response") or str(result)
            else:
                content = str(result)

            return f"\n--- [數據來源: {workspace_name}] ---\n{content}\n"
        except Exception as e:
            return f"Error querying LightRAG: {str(e)}"

class PythonSandboxTool(Tool):
    name = "python_sandbox"
    description = "執行純 Python 代碼。當需要對多間公司的數據進行「排序 (Sorting)」、「排名 (Ranking)」、或「數學計算」時【絕對強制使用】。嚴禁使用 LLM 自身能力進行數字排序或比大小。必須使用 print() 輸出結果。"
    parameters = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}

    async def execute(self, code: str, **kwargs) -> str:
        clean_code = code.replace("```python", "").replace("```", "").strip()
        try:
            client = docker.from_env()
            logs = client.containers.run(
                image="python:3.10-slim", command=["python", "-c", clean_code],
                remove=True, network_disabled=True, mem_limit="128m"
            )
            return logs.decode("utf-8").strip()
        except Exception as e:
            return f"Python Runtime Error: {str(e)}"

class BatchFinancialRAGTool(Tool):
    name = "batch_access_financial_data"
    description = "一次性並發查詢多間公司的財務數據。當需要對比、列表或獲取多間公司數據時必須使用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "workspace_names": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "公司資料夾名稱列表，例如 ['3SBio Inc._1530', 'Akeso_9926']"
            },
            "year": {"type": "string", "description": "財報年份，例如 '2024'"},
            "query": {"type": "string", "description": "要查詢的問題"}
        },
        "required": ["workspace_names", "year", "query"]
    }

    def __init__(self, base_path: Path, refs_list: list):
        super().__init__()
        self.base_path = base_path
        self.refs_list = refs_list

    async def execute(self, workspace_names: list, year: str, query: str, **kwargs) -> str:
        async def fetch_single(company: str):
            target_dir = self.base_path / "financial_report" / company / year
            if not target_dir.exists():
                return f"[{company}]: 找不到 {year} 年的數據庫"
            
            try:
                rag = await get_rag_instance(str(target_dir).replace("/app/data/rag_storage/", ""))
                param = QueryParam(mode="mix", stream=False, top_k=30) 
                
                if hasattr(rag, "aquery_llm"):
                    result = await rag.aquery_llm(query, param=param)
                else:
                    result = await rag.aquery(query, param=param)
                    
                if isinstance(result, dict):
                    if "data" in result and "references" in result["data"]:
                        self.refs_list.extend(result["data"]["references"])
                    content = result.get("llm_response", {}).get("content") or result.get("response") or str(result)
                else:
                    content = str(result)
                    
                return f"--- [{company} 2024 數據] ---\n{content}\n"
            except Exception as e:
                return f"[{company}]: 查詢失敗 ({str(e)})"

        results = await asyncio.gather(*(fetch_single(comp) for comp in workspace_names))
        return "\n".join(results)

# ==========================================
# 3. Agent 初始化與混合式 Prompt
# ==========================================
def _create_master_agent(collected_refs: list) -> AgentLoop:
    nanobot_root = Path("/app/nanobot") 
    workspace_path = nanobot_root / ".workspace"
    workspace_path.mkdir(exist_ok=True)

    provider = LiteLLMProvider(
        api_key=global_args.llm_binding_api_key,
        api_base=global_args.llm_binding_host,
        default_model=global_args.llm_model
    )
    
    agent = AgentLoop(bus=MessageBus(), provider=provider, workspace=workspace_path, max_iterations=100)
    loader = SkillsLoader(workspace=nanobot_root)
    skills_context = loader.load_skills_for_context(loader.get_always_skills())

    agent.system_prompt = f"""
    You are a Senior Financial Research Agent.

    CRITICAL CONCEPTS & CONSTRAINTS:
    1. **DIRECT EXTRACTION (直接提取)**: If the user asks for specific figures (e.g., exact revenue, percentage of shareholding, list of directors, debt amounts) from a single company's report, use `access_financial_data` to retrieve and output the exact text/numbers. Do not over-complicate.
    
    2. **CROSS-COMPANY (跨庫檢索)**: If the query involves multiple companies or an index (e.g., "all biotech companies"), you MUST use `batch_access_financial_data` exactly ONCE to cross-query all workspaces concurrently. Do NOT loop single queries.
    
    3. **RANKING & SORTING (腳本排序)**: You are FORBIDDEN to mentally rank, sort, or calculate numbers. If the user asks to "rank by revenue", "sort the list", or "find the highest/lowest", you MUST:
       - First, fetch the raw data.
       - Second, pass the raw data into `python_sandbox` to write a sorting script.
       - Third, use the script's `print()` output as your final answer.
       
    4. **ZERO HALLUCINATION**: Answers MUST be 100% based on retrieved chunks.

    NAVIGATION RULES:
    - Always use `list_available_workspaces` first if you don't know the exact company names.
    
    請用專業的香港粵語回應用戶。
    
    {skills_context}
    """
    
    agent.tools.register(WorkspaceDiscoveryTool(BASE_STORAGE_PATH))
    agent.tools.register(FinancialRAGTool(BASE_STORAGE_PATH, collected_refs))
    agent.tools.register(BatchFinancialRAGTool(BASE_STORAGE_PATH, collected_refs))
    agent.tools.register(PythonSandboxTool())
    
    return agent

# ==========================================
# 🌟 Golden DB (黃金庫) 查詢與攔截邏輯
# ==========================================
async def search_golden_db(new_query: str, ws_path: str) -> Optional[str]:
    try:
        golden_db_path = BASE_STORAGE_PATH / ws_path / "golden_db.jsonl"
        if not golden_db_path.exists(): return None
            
        db_records = []
        with open(golden_db_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): db_records.append(json.loads(line))
        if not db_records: return None

        embed_func = _get_dedicated_embedding()
        vectors = await embed_func(texts=[new_query])
        new_vec_np = np.array(vectors[0])
        
        best_match, highest_score = None, 0.0
        for record in db_records:
            db_vec_np = np.array(record["vector"])
            norm_new, norm_db = norm(new_vec_np), norm(db_vec_np)
            if norm_new == 0 or norm_db == 0: continue
            score = np.dot(new_vec_np, db_vec_np) / (norm_new * norm_db)
            if score > highest_score:
                highest_score = score
                best_match = record
                
        if highest_score > 0.92 and best_match:
            logger.info(f"✨ 命中 Verified 黃金庫！相似度: {highest_score:.4f}")
            return best_match["answer"]
        return None
    except Exception:
        return None

async def process_query_logic(request: QueryRequest, ws_path: str) -> Tuple[str, list]:
    # 第一關：攔截器 (Golden DB)。如果有被 Like 過嘅答案，直接秒回！
    verified_answer = await search_golden_db(request.query, ws_path)
    if verified_answer:
        return f"✨ **[已核實答案 Verified Result]**\n\n{verified_answer}", []

    # 第二關：常規 Agent 查詢
    collected_refs = []
    agent = _create_master_agent(collected_refs)
    
    # 🌟 核心修改：徹底拔除 Conversation History，實現「無狀態 (Stateless)」查詢！
    # 無論前端傳幾多歷史過嚟，AI 都只會睇到當下呢一句，唔會受過去對話干擾。
    prompt_payload = f"USER REQUEST: {request.query}"

    unique_chat_id = f"session_{int(time.time())}"

    ans = await agent.process_direct(content=prompt_payload, channel="webui", chat_id=unique_chat_id)
    
    unique_refs = []
    seen = set()
    for ref in collected_refs:
        rid = ref.get('reference_id')
        if rid and rid not in seen:
            seen.add(rid)
            unique_refs.append(ref)
            
    return ans, unique_refs

# ==========================================
# 4. API Endpoints
# ==========================================
def create_adapter_routes(rag, api_key=None, top_k=60):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/workspaces/list")
    async def get_workspaces_for_ui():
        structure = []
        for cat in ["index", "financial_report"]:
            cat_path = BASE_STORAGE_PATH / cat
            if not cat_path.exists(): continue
            if cat == "index":
                for d in cat_path.iterdir():
                    if d.is_dir(): structure.append({"label": f"📌 Index: {d.name}", "value": f"index/{d.name}"})
            else:
                for company in cat_path.iterdir():
                    if company.is_dir():
                        for year in company.iterdir():
                            if year.is_dir():
                                structure.append({"label": f"📊 {company.name} ({year.name})", "value": f"financial_report/{company.name}/{year.name}"})
        return structure

    @router.get("/graphs")
    async def get_dynamic_graphs(request: Request, label: str = "*", max_depth: int = 3, max_nodes: int = 1000):
        ws_path = request.query_params.get("workspace")
        if not ws_path: return {"nodes": [], "edges": []}
        full_path = (BASE_STORAGE_PATH / ws_path.lstrip("/")).resolve() / "graph_chunk_entity_relation.graphml"
        if not full_path.exists(): return {"nodes": [], "edges": []}
        try:
            G = nx.read_graphml(full_path)
            if len(G.nodes()) > max_nodes:
                nodes_by_importance = sorted(G.degree(), key=lambda x: x[1], reverse=True)
                nodes_to_keep = [n for n, deg in nodes_by_importance[:max_nodes]]
                G = G.subgraph(nodes_to_keep).copy()

            nodes = []
            for n, data in G.nodes(data=True):
                real_type = data.get("d1") or data.get("entity_type") or data.get("type") or "Unknown"
                clean_props = {k: v for k, v in data.items() if k not in ["label", "name", "entity_type", "type", "d1", "d0"]}
                nodes.append({
                    "id": str(n), "label": str(n), "labels": [real_type],
                    "properties": {**clean_props, "entity_type": real_type, "name": str(n), "label": str(n)},
                    "x": np.random.uniform(-100, 100), "y": np.random.uniform(-100, 100), "size": 10 + (G.degree(n) * 2) 
                })
            
            edges = [{"id": f"{u}-{v}", "source": str(u), "target": str(v), "type": "arrow", "label": data.get("description") or data.get("label") or "connected", "properties": data} for u, v, data in G.edges(data=True)]
            return {"nodes": nodes, "edges": edges}
        except Exception: return {"nodes": [], "edges": []}
        
    @router.get("/documents")
    async def get_dynamic_documents(request: Request):
        ws_path = request.query_params.get("workspace")
        if not ws_path: return {"statuses": {}}
        target_dir = str(BASE_STORAGE_PATH / ws_path)
        rag_instance = await get_rag_instance(target_dir)
        docs = await rag_instance.doc_status.get_all()
        status_groups = defaultdict(list)
        for doc in docs: status_groups[doc.get("status", "processed")].append(doc)
        return {"statuses": status_groups}

    @router.get("/graph/label/popular")
    async def get_dynamic_labels(request: Request, limit: int = 300):
        ws_path = request.query_params.get("workspace")
        if not ws_path: return []
        full_path = (BASE_STORAGE_PATH / ws_path.lstrip("/")).resolve() / "graph_chunk_entity_relation.graphml"
        if not full_path.exists(): return []
        try:
            G = nx.read_graphml(full_path)
            labels_count = defaultdict(int)
            for n, data in G.nodes(data=True):
                real_type = data.get("d1") or data.get("entity_type") or data.get("type")
                if real_type and isinstance(real_type, str):
                    for t in [t.strip() for t in real_type.split(",")]:
                        if t: labels_count[t] += 1
            sorted_labels = sorted(labels_count.items(), key=lambda x: x[1], reverse=True)
            return [label for label, count in sorted_labels[:limit]]
        except Exception: return []

    @router.post("/graph/entity/edit")
    async def edit_entity(request: Request, data: dict):
        ws_path = request.query_params.get("workspace")
        if not ws_path: return {"status": "error"}
        try:
            rag_instance = await get_rag_instance(str(BASE_STORAGE_PATH / ws_path))
            result = await rag_instance.update_entity(data["entity_name"], data["updated_data"])
            return {"status": "success", "data": result}
        except Exception as e: return {"status": "error", "message": str(e)}

    @router.post("/graph/relation/edit")
    async def edit_relation(request: Request, data: dict):
        ws_path = request.query_params.get("workspace")
        if not ws_path: return {"status": "error"}
        try:
            rag_instance = await get_rag_instance(str(BASE_STORAGE_PATH / ws_path))
            result = await rag_instance.update_relation(data["source_id"], data["target_id"], data["updated_data"])
            return {"status": "success", "data": result}
        except Exception as e: return {"status": "error", "message": str(e)}

    @router.post("/feedback/like")
    async def save_feedback_like(request: Request):
        try:
            ws_path = request.query_params.get("workspace", "general")
            payload = await request.json()
            ws_path = payload.get("company_year", ws_path)
            user_query, verified_answer = payload.get("query", ""), payload.get("verified_answer", "")
            if not user_query or not verified_answer: return JSONResponse(status_code=400, content={"error": "Missing data"})
                
            embed_func = _get_dedicated_embedding()
            vectors = await embed_func(texts=[user_query])
            query_vector = vectors[0].tolist() if isinstance(vectors[0], np.ndarray) else vectors[0]
            record_id = hashlib.md5(user_query.encode()).hexdigest()
            
            golden_record = {"id": record_id, "query": user_query, "answer": verified_answer, "vector": query_vector, "timestamp": time.time()}
            feedback_dir = BASE_STORAGE_PATH / ws_path
            feedback_dir.mkdir(parents=True, exist_ok=True)
            golden_db_path = feedback_dir / "golden_db.jsonl"
            
            existing_records = []
            if golden_db_path.exists():
                with open(golden_db_path, "r", encoding="utf-8") as f:
                    existing_records = [json.loads(line) for line in f if line.strip()]
                            
            existing_records = [r for r in existing_records if r["id"] != record_id]
            existing_records.append(golden_record)
            
            with open(golden_db_path, "w", encoding="utf-8") as f:
                for r in existing_records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return {"status": "success"}
        except Exception as e: return JSONResponse(status_code=500, content={"error": str(e)})

    @router.post("/query", response_model=QueryResponse, dependencies=[Depends(combined_auth)])
    async def master_query_endpoint(request: Request, query_req: QueryRequest):
        ws_path = request.query_params.get("workspace", "general")
        ans, refs = await process_query_logic(query_req, ws_path)
        return QueryResponse(response=ans, references=refs if refs else None)

    @router.post("/query/stream", dependencies=[Depends(combined_auth)])
    async def master_query_stream_endpoint(request: Request, query_req: QueryRequest):
        ws_path = request.query_params.get("workspace", "general")
        async def event_generator():
            yield json.dumps({"response": "⏳ **AI 正在多庫導航並執筆分析**...\n\n---\n\n"}) + "\n"
            task = asyncio.create_task(process_query_logic(query_req, ws_path))
            while not task.done():
                await asyncio.sleep(2.0)
                if not task.done(): yield json.dumps({"response": "."}) + "\n"
            try:
                ans, refs = task.result()
                payload = {"response": "\n\n" + ans}
                if refs: payload["references"] = refs
                yield json.dumps(payload) + "\n"
            except Exception as e: yield json.dumps({"error": str(e)}) + "\n"
        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    return router