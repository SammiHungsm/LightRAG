import os
import sys
import json
import asyncio
import numpy as np
import docker
from pathlib import Path
from typing import List, Dict, Any, Tuple
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

# LightRAG 核心
from lightrag.base import QueryParam
from lightrag.utils import logger, EmbeddingFunc
from lightrag.api.config import global_args 
from lightrag.llm.openai import openai_embed
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
# 工具 1：動態目錄掃描器 (Workspace Discovery)
# ==========================================
class WorkspaceDiscoveryTool(Tool):
    name = "list_available_workspaces"
    description = "掃描系統內所有已建庫的資料。你可以看到有咩 index (名單) 同埋有咩 financial_report (具體公司年份)。"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, base_path: Path):
        super().__init__()
        self.base_path = base_path

    async def execute(self, **kwargs) -> str:
        structure = {}
        for cat in ["index", "financial_report"]:
            cat_path = self.base_path / cat
            if not cat_path.exists(): continue
            if cat == "index":
                structure[cat] = [f.name for f in cat_path.iterdir() if f.is_dir()]
            else:
                report_map = {}
                for company in cat_path.iterdir():
                    if company.is_dir():
                        years = [y.name for y in company.iterdir() if y.is_dir()]
                        report_map[company.name] = years
                structure[cat] = report_map
        return json.dumps(structure, indent=2, ensure_ascii=False)

# ==========================================
# 工具 2：動態圖譜查詢 (Multi-RAG Router)
# ==========================================
class FinancialRAGTool(Tool):
    name = "access_financial_data"
    description = "入去某個特定嘅知識庫攞數據。你必須提供 category (index/financial_report), workspace_name 同埋 query。"
    parameters = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": ["index", "financial_report"]},
            "workspace_name": {"type": "string", "description": "資料夾名稱，例如 '3SBio_1530'"},
            "year": {"type": "string", "description": "年份，如果是財報類則必填"},
            "query": {"type": "string", "description": "具體要問嘅問題"}
        },
        "required": ["category", "workspace_name", "query"]
    }

    def __init__(self, base_path: Path, embed_func):
        super().__init__()
        self.base_path = base_path
        self.embed_func = embed_func
        self._rag_cache = {}

    async def execute(self, category: str, workspace_name: str, query: str, year: str = None, **kwargs) -> str:
        if category == "index":
            target_dir = self.base_path / "index" / workspace_name
        else:
            if not year: return "錯誤：查詢財報類數據必須提供 year 參數。"
            target_dir = self.base_path / "financial_report" / workspace_name / year

        if not target_dir.exists():
            return f"錯誤：找不到工作空間 {workspace_name}。"

        dir_key = str(target_dir)
        if dir_key not in self._rag_cache:
            logger.info(f"🧬 動態載入圖譜實例: {dir_key}")
            self._rag_cache[dir_key] = LightRAG(working_dir=dir_key, embedding_func=self.embed_func)
        
        rag = self._rag_cache[dir_key]
        result = await rag.aquery(query, param=QueryParam(mode="mix"))
        return f"\n--- [數據來源: {workspace_name} ({category})] ---\n{result}\n"

# ==========================================
# 工具 3：Python 沙盒 (Docker)
# ==========================================
class PythonSandboxTool(Tool):
    name = "python_sandbox"
    description = "執行純 Python 代碼。計算財務指標、排名或處理複雜邏輯時必用。必須 print() 結果。"
    parameters = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python Code"}}
    }

    async def execute(self, code: str, **kwargs) -> str:
        clean_code = code.replace("```python", "").replace("```", "").strip()
        try:
            client = docker.from_env()
            logs = client.containers.run(
                image="python:3.10-slim",
                command=["python", "-c", clean_code],
                remove=True, network_disabled=True, mem_limit="128m"
            )
            return logs.decode("utf-8").strip()
        except Exception as e:
            return f"Python 執行出錯: {str(e)}"

# ==========================================
# 核心大腦配置與邏輯
# ==========================================
def _get_dedicated_embedding():
    async def embed_wrapper(texts: list[str]) -> np.ndarray:
        return await openai_embed.func(
            texts=texts, model=global_args.embedding_model,
            api_key=global_args.llm_binding_api_key, base_url=global_args.llm_binding_host
        )
    return EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embed_wrapper)

def _create_master_agent() -> AgentLoop:
    storage_path = Path("./data/rag_storage")
    workspace_path = Path(global_args.working_dir) / "nanobot_workspace"
    
    provider = LiteLLMProvider(
        api_key=global_args.llm_binding_api_key,
        api_base=global_args.llm_binding_host,
        default_model=global_args.llm_model
    )
    
    agent = AgentLoop(bus=MessageBus(), provider=provider, workspace=workspace_path, max_iterations=20)
    
    loader = SkillsLoader(workspace=workspace_path)
    skills_context = loader.load_skills_for_context(loader.get_always_skills())

    agent.system_prompt = f"""
    你係一個專業嘅「全方位財務分析 Agent」。你負責導航多個知識庫。
    
    工作守則：
    1. **資訊發現**：如果你唔知有咩公司，先用 `list_available_workspaces`。
    2. **物理跳轉**：數據係分開儲存嘅，用 `access_financial_data` 喺庫之間跳轉攞數據。
    3. **嚴謹計算**：涉及數字處理，請參考 `batch_analyzer` 技能，並使用 `python_sandbox`。
    4. **基於事實**：所有回答必須來自工具返回嘅數據，唔可以老作。
    
    {skills_context}
    """
    
    agent.tools.register(WorkspaceDiscoveryTool(storage_path))
    agent.tools.register(FinancialRAGTool(storage_path, _get_dedicated_embedding()))
    agent.tools.register(PythonSandboxTool())
    
    return agent

async def process_query_logic(request: QueryRequest) -> Tuple[str, list]:
    """統一大腦執行邏輯"""
    agent = _create_master_agent()
    history_str = ""
    if request.conversation_history:
        history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in request.conversation_history[-3:]])
    
    full_query = f"HISTORY:\n{history_str}\n\nUSER REQUEST: {request.query}"
    
    # 執行 Agent 思考流程
    ans = await agent.process_direct(content=full_query, channel="webui", chat_id="master_session")
    return ans, []

# ==========================================
# API Endpoints (包含原本消失的 Streaming!)
# ==========================================
def create_adapter_routes(api_key=None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post("/query", response_model=QueryResponse, dependencies=[Depends(combined_auth)])
    async def master_query_endpoint(request: QueryRequest):
        logger.info(f"🛡️ [Standard] 收到查詢請求: {request.query}")
        ans, refs = await process_query_logic(request)
        return QueryResponse(response=ans, references=None)

    @router.post("/query/stream", dependencies=[Depends(combined_auth)])
    async def master_query_stream_endpoint(request: QueryRequest):
        logger.info(f"🌊 [Stream] 啟動串流查詢: {request.query}")
        
        async def event_generator():
            # 1. 發送初始等待消息
            yield json.dumps({"response": "⏳ **AI 正在導航多個知識庫並分析數據**...\n請稍候...\n\n---\n\n"}) + "\n"
            
            # 2. 啟動 Agent 背景任務
            task = asyncio.create_task(process_query_logic(request))
            
            # 3. 心跳機制：當 Agent 仲諗緊嘢嗰陣，每 2 秒噴一個點點
            while not task.done():
                await asyncio.sleep(2.0)
                if not task.done():
                    yield json.dumps({"response": "."}) + "\n"
            
            # 4. 任務完成，發送最終答案
            try:
                ans, refs = task.result()
                yield json.dumps({"response": "\n\n---\n\n" + ans}) + "\n"
                if refs:
                    yield json.dumps({"references": refs}) + "\n"
            except Exception as e:
                logger.error(f"Stream generation error: {e}", exc_info=True)
                yield json.dumps({"error": f"處理失敗: {str(e)}"}) + "\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    return router