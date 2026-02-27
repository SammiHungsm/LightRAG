import os
import sys
import json
import asyncio
import numpy as np
import docker
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

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
# 1. 基礎設施：Embedding 與 RAG 緩存
# ==========================================
_rag_cache: Dict[str, LightRAG] = {}

def _get_dedicated_embedding():
    """修正：返傳一個 EmbeddingFunc 物件，但要確保 LightRAG 接收到正確格式"""
    async def embed_wrapper(texts: list[str]) -> np.ndarray:
        return await openai_embed.func(
            texts=texts, 
            model=global_args.embedding_model,
            api_key=global_args.llm_binding_api_key, 
            base_url=global_args.llm_binding_host
        )
    # 🌟 呢度返傳成個 EmbeddingFunc，因為佢內部實作咗 __call__
    return EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embed_wrapper)
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

def get_rag_instance(ws_path_from_ui: str) -> LightRAG:
    """ws_path_from_ui 應該係例如 'financial_report/3SBio/2024'"""
    base_dir = Path("/app/data/rag_storage")
    
    # 🌟 避免重複拼接：如果 ws_path 已經包含 base_dir 嘅名，要處理返
    if "data/rag_storage" in ws_path_from_ui:
        # 如果傳入嚟已經係完整路徑，就直接用
        full_path = str(Path(ws_path_from_ui).absolute())
    else:
        full_path = str((base_dir / ws_path_from_ui.lstrip("/")).absolute())
    
    if full_path not in _rag_cache:
        logger.info(f"🧬 正在初始化物理路徑: {full_path}")
        # 🌟 修正： embedding_func 要傳入 Callable 
        _rag_cache[full_path] = LightRAG(
            working_dir=full_path,
            llm_model_func=_get_dedicated_llm(),
            embedding_func=_get_dedicated_embedding() # 佢係一個 Callable 物件
        )
    return _rag_cache[full_path]

# ==========================================
# 2. Agent 工具集 (完全保留引用收集與導航邏輯)
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
        
        rag = get_rag_instance(str(target_dir))
        param = QueryParam(mode="mix", stream=False, top_k=60)
        result = await rag.aquery_llm(query, param=param)
        
        data = result.get("data", {})
        if "references" in data:
            self.refs_list.extend(data["references"])
            
        content = result.get("llm_response", {}).get("content", "")
        return f"\n--- [數據來源: {workspace_name}] ---\n{content}\n"

class PythonSandboxTool(Tool):
    name = "python_sandbox"
    description = "執行純 Python 代碼。計算財務指標、排名、對比數據時必用。必須使用 print() 輸出結果。"
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

# ==========================================
# 3. Agent 初始化與混合式 Prompt
# ==========================================
def _create_master_agent(collected_refs: list) -> AgentLoop:
    storage_path = Path("./data/rag_storage")
    
    # 🌟 因為你冇 nanobot_workspace，Skills 喺 nanobot/skills 
    nanobot_root = Path("/app/nanobot") 
    
    # 用嚟擺 Session 嘅臨時位
    workspace_path = nanobot_root / ".workspace"
    workspace_path.mkdir(exist_ok=True)

    provider = LiteLLMProvider(
        api_key=global_args.llm_binding_api_key,
        api_base=global_args.llm_binding_host,
        default_model=global_args.llm_model
    )
    
    agent = AgentLoop(bus=MessageBus(), provider=provider, workspace=workspace_path, max_iterations=100)
    
    # 🌟 修正：Loader 去 /app/nanobot 搵 skills 
    loader = SkillsLoader(workspace=nanobot_root)
    skills_context = loader.load_skills_for_context(loader.get_always_skills())

    # 🌟 混合式強效 Prompt：保留嚴格約束 + 增加導航指引
    agent.system_prompt = f"""
    You are a Senior Financial Research Agent tailored for MULTI-WORKSPACE internal data analysis.
    
    STRICT CONSTRAINTS:
    1. **INTERNAL DATA ONLY**: You must search facts using `access_financial_data`. NEVER use your pre-trained knowledge.
    2. **ZERO HALLUCINATION**: Answers MUST be 100% based on retrieved chunks. If no data, state "內部資料庫中沒有相關資訊".
    3. **DATA ANALYSIS & MATH**: You MUST use `python_sandbox` for calculations. NEVER do mental math.
    
    NAVIGATION RULES:
    - First, use `list_available_workspaces` to see which companies or indexes are available.
    - Use `access_financial_data` to hop between isolated workspaces.
    
    {skills_context}
    """
    
    agent.tools.register(WorkspaceDiscoveryTool(storage_path))
    agent.tools.register(FinancialRAGTool(storage_path, collected_refs))
    agent.tools.register(PythonSandboxTool())
    return agent

async def process_query_logic(request: QueryRequest) -> Tuple[str, list]:
    collected_refs = []
    agent = _create_master_agent(collected_refs)
    history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in request.conversation_history[-3:]]) if request.conversation_history else ""
    ans = await agent.process_direct(content=f"HISTORY:\n{history}\n\nUSER REQUEST: {request.query}", channel="webui", chat_id="master_session")
    
    # 引用去重
    unique_refs = []
    seen = set()
    for ref in collected_refs:
        rid = ref.get('reference_id')
        if rid and rid not in seen:
            seen.add(rid); unique_refs.append(ref)
    return ans, unique_refs

# ==========================================
# 4. API Endpoints (全面整合所有功能)
# ==========================================
def create_adapter_routes(rag, api_key=None, top_k=60):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/workspaces/list")
    async def get_workspaces_for_ui():
        storage_path = Path("./data/rag_storage")
        structure = []
        for cat in ["index", "financial_report"]:
            cat_path = storage_path / cat
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
    async def get_dynamic_graphs(
        request: Request, 
        label: str = "*", 
        max_depth: int = 3, 
        max_nodes: int = 1000 
    ):
        """WebUI 專用接口：自動校正屬性映射、修復 Label 顯示並優化效能"""
        ws_path = request.query_params.get("workspace")
        
        # 1. 取得 Workspace 路徑 (確保安全且不重複拼接)
        if not ws_path:
            return {"nodes": [], "edges": []}

        # 這裡根據你 Docker 映射的物理路徑
        base_dir = Path("./data/rag_storage").resolve()
        # 移除 ws_path 開頭的斜槓防止 Path 拼接跳回根目錄
        full_path = (base_dir / ws_path.lstrip("/")).resolve() / "graph_chunk_entity_relation.graphml"
        
        print(f"🔍 [Graph-Request] Workspace: '{ws_path}' | Path: {full_path}")
        
        if not full_path.exists():
            print(f"❌ [Graph-Request] File NOT found at: {full_path}")
            return {"nodes": [], "edges": []}

        try:
            G = nx.read_graphml(full_path)
            
            # 🌟 優化 1: 按 Degree 排序，保留重要節點
            if len(G.nodes()) > max_nodes:
                # 搵出連接數最高嘅點
                nodes_by_importance = sorted(G.degree(), key=lambda x: x[1], reverse=True)
                nodes_to_keep = [n for n, deg in nodes_by_importance[:max_nodes]]
                G = G.subgraph(nodes_to_keep).copy()

            # 搵返 backend_router.py 入面迴圈處理 nodes 嘅位置
            nodes = []
            # 喺 backend_router.py 的 get_dynamic_graphs 內修改 loop
            for n, data in G.nodes(data=True):
                # 1. 確定正確的 Type 同 Name
                real_type = data.get("d1") or data.get("entity_type") or data.get("type") or "Unknown"
                real_name = str(n) # n 通常係 "HKFRS 9"

                # 2. 🌟 高級清理邏輯：唔好用 data.copy()，改為「只拿需要的」
                # 這樣可以確保原本 GraphML 裡面那個裝著 "regulation" 的 'label' 徹底消失
                clean_props = {}
                for k, v in data.items():
                    # 排除所有會干擾前端渲染的系統 Key
                    if k not in ["label", "name", "entity_type", "type", "d1", "d0"]:
                        clean_props[k] = v

                nodes.append({
                    "id": str(n),
                    "label": real_name,       # 🌟 Sigma 渲染引擎主要讀呢個
                    "labels": [real_type],    # 過濾用
                    "properties": {
                        **clean_props,        # 只有純粹的描述、時間等數據
                        "entity_type": real_type, # 上色用
                        "name": real_name,
                        "label": real_name    # 🌟 雙重保險：覆蓋 properties 內可能存在的 label
                    },
                    "x": np.random.uniform(-100, 100),
                    "y": np.random.uniform(-100, 100),
                    "size": 10 + (G.degree(n) * 2) 
                })
            # 5. 轉換 Edge 格式
            edges = []
            for u, v, data in G.edges(data=True):
                # 提取關係描述
                e_description = (
                    data.get("description") or 
                    data.get("label") or 
                    data.get("d8") or   # LightRAG 常見關係 ID
                    "connected"
                )
                
                edges.append({
                    "id": f"{u}-{v}",
                    "source": str(u),
                    "target": str(v),
                    "type": "arrow", # 設定連線樣式
                    "label": e_description,
                    "properties": data
                })

            print(f"✅ [Graph-Request] Success! Sent {len(nodes)} nodes and {len(edges)} edges.")
            return {"nodes": nodes, "edges": edges}

        except Exception as e:
            print(f"❌ [Graph-Request] Critical Error: {str(e)}")
            import traceback
            traceback.print_exc() # 輸出詳細錯誤到 Docker Log
            return {"nodes": [], "edges": []}
        
    @router.get("/documents")
    async def get_dynamic_documents(request: Request):
        ws_path = request.query_params.get("workspace")
        if not ws_path: return {"statuses": {}}
        target_dir = str(Path("./data/rag_storage") / ws_path)
        rag_instance = get_rag_instance(target_dir)
        docs = await rag_instance.doc_status.get_all()
        status_groups = defaultdict(list)
        for doc in docs:
            status_groups[doc.get("status", "processed")].append(doc)
        return {"statuses": status_groups}

    @router.get("/graph/label/popular")
    async def get_dynamic_labels(request: Request, limit: int = 300):
        ws_path = request.query_params.get("workspace")
        if not ws_path: return []
        
        rag_instance = get_rag_instance(ws_path)
        entities = await rag_instance.full_entities.get_all()
        
        # 🌟 修正：喺 KV 儲存入面都要搵 d1 同 type
        return list(set([
            (e.get("entity_type") or e.get("type") or e.get("d1")) 
            for e in entities 
            if (e.get("entity_type") or e.get("type") or e.get("d1"))
        ]))[:limit]

    @router.post("/graph/entity/edit")
    async def edit_entity(request: Request, data: dict):
        ws_path = request.query_params.get("workspace")
        if not ws_path: return {"status": "error", "message": "Missing workspace"}
        rag_instance = get_rag_instance(str(Path("./data/rag_storage") / ws_path))
        try:
            result = await rag_instance.update_entity(data["entity_name"], data["updated_data"])
            return {"status": "success", "data": result}
        except Exception as e: return {"status": "error", "message": str(e)}

    @router.post("/graph/relation/edit")
    async def edit_relation(request: Request, data: dict):
        ws_path = request.query_params.get("workspace")
        if not ws_path: return {"status": "error", "message": "Missing workspace"}
        rag_instance = get_rag_instance(str(Path("./data/rag_storage") / ws_path))
        try:
            result = await rag_instance.update_relation(data["source_id"], data["target_id"], data["updated_data"])
            return {"status": "success", "data": result}
        except Exception as e: return {"status": "error", "message": str(e)}

    @router.post("/query", response_model=QueryResponse, dependencies=[Depends(combined_auth)])
    async def master_query_endpoint(request: QueryRequest):
        ans, refs = await process_query_logic(request)
        return QueryResponse(response=ans, references=refs if refs else None)

    @router.post("/query/stream", dependencies=[Depends(combined_auth)])
    async def master_query_stream_endpoint(request: QueryRequest):
        async def event_generator():
            yield json.dumps({"response": "⏳ **AI 正在多庫導航並執筆分析**...\n\n---\n\n"}) + "\n"
            task = asyncio.create_task(process_query_logic(request))
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