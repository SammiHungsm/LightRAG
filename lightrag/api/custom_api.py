import os
import sys
import re
import json
import docker
import asyncio
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import QueryParam
from lightrag.utils import logger
from lightrag.api.config import global_args 

# [Imports] 引入原本的 Models
from lightrag.api.routers.query_routes import QueryRequest, QueryResponse 

# [Imports] 引入 Nanobot 相關組件
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.agent.tools.base import Tool
from nanobot.agent.skills import SkillsLoader

from litellm import acompletion

router = APIRouter(tags=["query"])

# ==========================================
# 動態獲取特定領域技能 (Dynamic Skill Loader)
# ==========================================
def get_specific_skill_content(domain_name: str) -> str:
    if not domain_name:
        return ""
    
    keyword = domain_name.lower().split()[0]
    skill_key = f"{keyword}_knowledge"
    
    workspace_path = Path(global_args.working_dir) / "nanobot_workspace" / "skills" / skill_key / "SKILL.md"
    
    if workspace_path.exists():
        try:
            content = workspace_path.read_text(encoding="utf-8")
            logger.info(f"📚 成功動態載入技能: {skill_key}")
            return f"\n\n=== 相關領域知識 ({skill_key}) ===\n{content}\n==========================\n"
        except Exception as e:
            logger.error(f"讀取技能文件失敗: {e}")
    return ""

def get_always_active_skills_content() -> str:
    workspace_path = Path(global_args.working_dir) / "nanobot_workspace"
    try:
        loader = SkillsLoader(workspace=workspace_path)
        always_skills = loader.get_always_skills()
        if always_skills:
            skills_content = loader.load_skills_for_context(always_skills)
            return f"\n\n=== 附加領域知識 (Skills) ===\n{skills_content}\n==========================\n"
    except Exception as e:
        logger.error(f"載入技能時發生錯誤: {e}")
    return ""

# ==========================================
# 工具 1: LightRAG 知識庫工具
# ==========================================
class LightRAGTool(Tool):
    name = "search_knowledge_base"
    description = """
    The internal Knowledge Base containing annual reports and financial data.
    You MUST use this tool to answer questions.
    """
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Specific search query."}
        },
        "required": ["query"]
    }

    def __init__(self, rag_instance, refs_list):
        super().__init__()
        self.rag = rag_instance
        self.refs_list = refs_list

    async def execute(self, query: str, **kwargs) -> str:
        param = QueryParam(mode="mix", stream=False, top_k=60)
        try:
            result = await self.rag.aquery_llm(query, param=param)
            logger.info(f"-> 🔍 [DEBUG] LightRAGTool 取得數據成功。") 
            data = result.get("data", {})
            if "references" in data:
                self.refs_list.extend(data["references"])
            content = result.get("llm_response", {}).get("content", "")
            return content if content else "No relevant data found in internal database."
        except Exception as e:
            logger.error(f"Tool Error: {e}")
            return f"Error: {str(e)}"

# ==========================================
# 工具 2: Python 沙盒代碼直譯器 (ROUTE C 升級版)
# ==========================================
class PythonSandboxTool(Tool):
    name = "python_sandbox"
    description = """
    Executes Python code in a secure Docker sandbox. 
    CRITICAL: You MUST use this tool to calculate sums, averages, or math. NEVER do mental math.
    Instructions:
    1. Write pure Python code.
    2. You MUST use print() to output the final exact answer.
    3. Do not include markdown formatting (like ```python) in the code payload.
    """
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The pure Python code to execute. MUST end with a print() statement."}
        },
        "required": ["code"]
    }

    async def execute(self, code: str, **kwargs) -> str:
        clean_code = code.replace("```python", "").replace("```", "").strip()
        try:
            client = docker.from_env()
            logger.info(f"\n========== [DEBUG] Agent 觸發了 Python Sandbox Tool ==========\n{clean_code}\n=========================================================")
            
            logs = client.containers.run(
                image="python:3.10-slim",      
                command=["python", "-c", clean_code], 
                remove=True, network_disabled=True, mem_limit="128m", cpu_quota=50000, user="1000:1000", environment={"PYTHONUNBUFFERED": "1"} 
            )
            result = logs.decode("utf-8").strip()
            
            logger.info(f"-> 🎯 [DEBUG] 沙盒執行結果: {result}")
            return result if result else "Code executed successfully, but nothing was printed. Please modify the code to use print()."
        
        except docker.errors.ContainerError as e:
            error_msg = e.stderr.decode('utf-8').strip()
            logger.error(f"-> ❌ [DEBUG] 沙盒代碼報錯: {error_msg}")
            return f"Execution Error: {error_msg}\nPlease fix the python code and try again."
        except Exception as e:
            return f"System Error: {str(e)}"

# ==========================================
# 初始化 Nanobot 
# ==========================================
def _create_agent(rag, collected_refs: list) -> AgentLoop:
    workspace_path = Path(global_args.working_dir) / "nanobot_workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)
    provider = LiteLLMProvider(
        api_key=global_args.llm_binding_api_key,
        api_base=global_args.llm_binding_host,
        default_model=os.getenv("NANOBOT_MODEL", global_args.llm_model)
    )
    agent = AgentLoop(
        bus=MessageBus(), provider=provider, workspace=workspace_path,
        max_iterations=15, restrict_to_workspace=True
    )
    
    skills_context = get_always_active_skills_content()
    
    agent.system_prompt = f"""
    You are a Senior Financial Research Agent tailored for internal data analysis.
    
    STRICT CONSTRAINTS:
    1. **NO WEB SEARCH**: You do NOT have access to the internet.
    2. **INTERNAL DATA ONLY**: You must search facts using `search_knowledge_base`.
    3. **DATA ANALYSIS & MATH**: If the user asks for calculations, sums, or complex data manipulation, you MUST:
       - Step 1: Use `search_knowledge_base` to retrieve the raw data.
       - Step 2: Use `python_sandbox` to write and execute a Python script. NEVER perform mental math.
    4. **ZERO HALLUCINATION**: 
       - Your answers MUST be 100% based ONLY on the text chunks returned by `search_knowledge_base`. 
       - NEVER use your pre-trained internal knowledge to answer questions.
       - If the `search_knowledge_base` does not contain the answer, you MUST clearly state: "內部資料庫中沒有相關資訊".
       - NEVER make up or fabricate fake references.
    
    {skills_context}
    """
    
    agent.tools.register(LightRAGTool(rag, collected_refs))
    agent.tools.register(PythonSandboxTool())
    
    return agent

def _format_context(history: List[Dict[str, str]], query: str) -> str:
    if not history: return query
    context_str = "\n".join([f"{m.get('role','U').upper()}: {m.get('content','')}" for m in history[-3:]])
    return f"HISTORY:\n{context_str}\n\nCURRENT REQUEST:\n{query}"

# ==========================================
# 共用 LLM 函數
# ==========================================
async def _ask_llm_async(prompt: str, system_msg: str = "你是一個聰明的 AI 助手") -> str:
    model_name = os.getenv("NANOBOT_MODEL", global_args.llm_model)
    
    if global_args.llm_binding == "openai" and not model_name.startswith("openai/"):
        model_name = f"openai/{model_name}"
    elif global_args.llm_binding == "ollama" and not model_name.startswith("ollama/"):
        model_name = f"ollama/{model_name}"
        
    response = await acompletion(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        api_key=global_args.llm_binding_api_key,
        base_url=global_args.llm_binding_host,
        temperature=0.1 
    )
    return response.choices[0].message.content

# ==========================================
# 神級工作流：Batch Map-Reduce 分析 (ROUTE D)
# ==========================================
async def _is_batch_analysis_question(user_query: str) -> bool:
    prompt = f"""
    判斷以下問題是否需要執行「兩步走」的批次分析：
    第一步：先列出多個實體 (例如所有公司)。
    第二步：然後必須針對這些實體，逐一查詢它們的「特定財務數據或細節」(例如負債、利潤、營收)。
    
    【嚴格限制】：
    如果用戶只是單純要求「列出名單」、「有什麼公司」或「附上股票代號」，而沒有要求查詢每間公司的具體財務數據，請必須回答 'NO'。
    只有在明確需要對多個實體進行橫向數據比較時，才回答 'YES'。
    
    問題：{user_query}
    """
    response = await _ask_llm_async(prompt)
    logger.info(f"-> 🤖 [DEBUG] _is_batch_analysis_question 判斷結果: {response.strip()}")
    return "YES" in response.upper()

async def _run_batch_analysis_pipeline(rag, user_query: str) -> str:
    logger.info("-> 🚀 Step 1: 觸發批次分析模式 (Map-Reduce Loop 包含智能拆解)")
    
    # [更新] 意圖提取加入子項目拆解 (Query Decomposition)
    extract_prompt = f"""
    分析用戶問題：{user_query}
    請提取以下四個資訊：
    1. "entities": 目標群體 (例如 "biotech companies", "銀行")。
    2. "detail": 需要查詢的整體數據 (例如 "total liability", "利潤")。
    3. "condition": 查詢的特定條件或指定的報告來源 (例如 "2024", "Q3財報")。如果沒有特別指定，請填 "無"。
    4. "sub_queries": 為了計算這個整體數據，是否需要拆分成更小的會計科目來查詢？
       - 例如：查詢 "total liability" 時，通常需要拆成 ["current liabilities", "non-current liabilities"]。
       - 例如：查詢 "gross profit" 時，通常需要拆成 ["revenue", "cost of sales"]。
       - 如果不需要拆分，請返回空陣列 []。
    
    必須返回純 JSON，格式：{{"entities": "...", "detail": "...", "condition": "...", "sub_queries": ["...", "..."]}}
    """
    extract_json = await _ask_llm_async(extract_prompt)
    extract_json = extract_json.replace("```json", "").replace("```", "").strip()
    
    logger.info(f"\n========== [DEBUG] AI 提取的 JSON 意圖 (包含拆解邏輯) ==========\n{extract_json}\n==================================================================")
    
    try:
        info = json.loads(extract_json)
        entities = info.get("entities", "biotech companies")
        detail = info.get("detail", "total liability")
        condition = info.get("condition", "無")
        sub_queries = info.get("sub_queries", [])
    except Exception as e:
        logger.error(f"-> ❌ [DEBUG] JSON 解析失敗: {e}")
        entities, detail, condition, sub_queries = "biotech companies", "total liability", "無", []

    skill_context = get_specific_skill_content(entities)

    logger.info(f"-> 📂 Step 2: 查詢 {entities} 名單...")
    list_prompt = f"""
    任務：根據用戶問題「{user_query}」，列出所有相關的 {entities}。
    {skill_context}
    請優先參考上述「相關領域知識」中的名單（若有）。如果知識庫中有其他相關公司，也可以加入。
    只返回純 JSON Array，例如 ["WUXI BIO (2269)", "3SBIO (1530)"]，不要 Markdown。
    """
    list_json = await _ask_llm_async(list_prompt)
    list_json = list_json.replace("```json", "").replace("```", "").strip()
    
    logger.info(f"\n========== [DEBUG] 提取的公司名單 JSON ==========\n{list_json}\n=================================================")
    
    try:
        company_list = json.loads(list_json)
    except Exception as e:
        return f"名單提取失敗，無法執行 Loop 分析。錯誤: {e}"
        
    if not isinstance(company_list, list) or len(company_list) == 0:
        return "在知識庫中找不到相關的公司名單。"

    logger.info(f"-> 🔄 Step 3: 進入迴圈，準備併發查詢 {len(company_list)} 間公司...")
    
    # 3. 定義 Map 函數 (支援多步檢索 Multi-hop Retrieval)
    async def fetch_company_data(company: str):
        logger.info(f"   -> 🔍 [DEBUG] 開始查詢: {company} (條件: {condition})")
        
        combined_company_result = ""
        
        # [核心更新] 執行多步檢索
        if sub_queries and len(sub_queries) > 0:
            logger.info(f"   -> 🧩 [DEBUG] 觸發拆解檢索，分別查詢: {sub_queries}")
            for sub_query in sub_queries:
                if condition == "無":
                    sq = f"Find the exact value of {sub_query} for {company}."
                else:
                    sq = f"According strictly to the context containing '{condition}', find the exact value of {sub_query} for {company}."
                
                logger.info(f"      -> 🗣️ [DEBUG] 發送 Sub-Query: {sq}")
                res = await rag.aquery(sq, param=QueryParam(mode="mix", stream=False, top_k=80))
                combined_company_result += f"\n--- Data for {sub_query} ---\n{res}\n"
        else:
            # 沒有拆解項目，使用原本的寬鬆檢索策略
            base_query = f"Find ALL financial data, numbers, and figures related to {detail} for {company}."
            if condition == "無":
                specific_query = base_query
            else:
                specific_query = f"According strictly to the context containing '{condition}', {base_query}"
                
            logger.info(f"   -> 🗣️ [DEBUG] 發送 Query: {specific_query}")
            combined_company_result = await rag.aquery(specific_query, param=QueryParam(mode="mix", stream=False, top_k=80))

        logger.info(f"   -> 📄 [DEBUG] {company} 取得原始數據總長度: {len(combined_company_result)}")
        logger.debug(f"   -> 📄 [DEBUG] {company} 原始數據預覽: {combined_company_result[:150]}...")
        
        # 嚴格的 Python 計算防呆判斷
        calc_check_prompt = f"""
        你是一個嚴謹的會計數據分析助手。請閱讀以下來自知識庫的檢索結果：
        
        【檢索結果開始】
        {combined_company_result}
        【檢索結果結束】
        
        用戶想要查詢 {company} 在 {condition} (若為'無'則忽略) 的 {detail}。
        
        請執行以下嚴格的判斷邏輯：
        1. 【查無數據】：如果檢索結果明確表示「找不到」、「沒有相關資料」或完全沒有相關數字，請直接回答：「查無數據」。
        2. 【直接提取】：如果檢索結果中已經包含了明確的 {detail} 總數，請直接提取並回答該數字和單位。
        3. 【嚴格 Python 計算】：如果你發現檢索結果中沒有直接的總數，但是提供了碎片化的數據（例如："流動負債 為 50 萬，非流動負債 為 30 萬"），你必須寫一段純 Python 程式碼來將它們加總。
           
           【⚠️ 極度重要警告】：
           - 絕對不可自行進行數學運算！必須依賴 Python 執行。
           - 你必須非常小心地確認提取的數字「確實屬於」{detail} 的組成部分！絕對不可以把「現金流 (Cash Flow)」、「虧損 (Loss)」或無關的數字拿來加總。
           - 程式碼必須可以直接執行。
           - 必須使用 print() 輸出最終計算結果。
           - 不要包含任何 markdown 標記 (如 ```python)。
           - 如果你發現文本中雖然有數字，但無法確定它們加起來等於 {detail}，請誠實回答「數據不足，無法計算」，不要硬湊數字！
        """
        
        check_result = await _ask_llm_async(calc_check_prompt)
        
        logger.info(f"\n========== [DEBUG] LLM 判斷結果 ({company}) ==========\n{check_result}\n=========================================================")
        
        # 攔截 Python 代碼執行 (使用安全原生沙盒)
        if "print(" in check_result and "查無數據" not in check_result and "數據不足" not in check_result:
            logger.info(f"   -> 🧮 [DEBUG] 為 {company} 觸發 Python 安全沙盒計算...")
            code = check_result.replace("```python", "").replace("```", "").strip()
            lines = code.split('\n')
            python_code = "\n".join([line for line in lines if not line.startswith("查無數據") and not line.startswith("檢索") and not line.startswith("數據不足")])
            
            logger.info(f"\n========== [DEBUG] 準備在沙盒執行的純 Python 代碼 ({company}) ==========\n{python_code}\n=======================================================================")
            
            try:
                # 【原生安全沙盒修復版】不再依賴外部 Docker
                import io
                import contextlib
                
                output_capture = io.StringIO()
                with contextlib.redirect_stdout(output_capture):
                    # 使用 exec 執行，並限制其可以訪問的全域變數，防止惡意代碼
                    exec(python_code, {"__builtins__": {"print": print, "abs": abs, "round": round, "sum": sum, "min": min, "max": max}})
                
                calc_answer = output_capture.getvalue().strip()
                logger.info(f"   -> 🎯 [DEBUG] {company} 執行結果: {calc_answer}")
                return f"### {company}\n根據數據計算得出：{calc_answer}\n"
            
            except Exception as e:
                logger.error(f"   -> ❌ [DEBUG] {company} 計算失敗: {e}")
                return f"### {company}\n計算失敗。原始數據：{combined_company_result}\n"
        
        else:
            logger.info(f"   -> ⏩ [DEBUG] 無需 Python 計算，直接輸出文字。")
            return f"### {company}\n{check_result}\n"

    all_results = []
    batch_size = 5 
    
    for i in range(0, len(company_list), batch_size):
        batch = company_list[i : i + batch_size]
        logger.info(f"-> 📦 正在處理 Batch {i//batch_size + 1} / {(len(company_list) - 1)//batch_size + 1}...")
        batch_results = await asyncio.gather(*(fetch_company_data(comp) for comp in batch))
        all_results.extend(batch_results)

    logger.info("-> 📝 Step 4: 整合所有數據...")
    if len(company_list) > 10:
        intro_prompt = f"請為這份包含 {len(company_list)} 間 {entities} 的 {detail} 分析報告寫一段簡短的繁體中文前言（不要總結數據，只寫開場白）："
        intro = await _ask_llm_async(intro_prompt)
        final_report = f"{intro}\n\n" + "\n---\n".join(all_results)
        return final_report
    else:
        combined_text = "\n".join(all_results)
        final_prompt = f"""
        用戶問題：{user_query}
        以下是我們逐一查詢後得出的原始數據：
        {combined_text}
        
        請將這些數據整合成一份清晰、專業的繁體中文(粵語語氣)分析報告，使用表格或列表展示。
        
        【極度重要警告】：
        1. 如果原始數據中顯示找不到數據 (例如查無數據或數據不足)，請如實填寫「查無數據」。
        2. 絕對不可捏造參考資料 (References)。
        3. 嚴禁在回答中使用系統範例名稱作 Reference！只有在原始數據中真實出現的文件名才能作為 Reference。
        """
        return await _ask_llm_async(final_prompt)

# ==========================================
# 核心大腦：統一處理邏輯
# ==========================================
async def process_query_logic(request: QueryRequest, rag) -> tuple[str, list]:
    is_batch = await _is_batch_analysis_question(request.query)
    if is_batch:
        logger.info("🚀 觸發 ROUTE D: Map-Reduce 批次分析模式")
        ans = await _run_batch_analysis_pipeline(rag, request.query)
        return ans, []

    triggers = ["list all", "qualified opinion", "companies with", "show me all"]
    if any(t in request.query.lower() for t in triggers):
        logger.info("🔀 觸發 ROUTE A: 執行 LightRAG Global Mode")
        ans = await rag.aquery(request.query, param=QueryParam(mode="global", stream=False))
        return ans, []

    logger.info("🤖 觸發 ROUTE B: Standard Agent Mode (含 Python Sandbox 支援)")
    try:
        collected_refs = []
        agent = _create_agent(rag, collected_refs)
        agent_input = _format_context(request.conversation_history, request.query)

        ans = await agent.process_direct(content=agent_input, channel="webui", chat_id="session")

        unique_refs = []
        if request.include_references and collected_refs:
            seen = set()
            for ref in collected_refs:
                rid = ref.get('reference_id')
                if rid and rid not in seen:
                    seen.add(rid)
                    unique_refs.append(ref)
        return ans, unique_refs
    except Exception as e:
        logger.error(f"Nanobot Error: {e}", exc_info=True)
        return f"Agent process failed: {str(e)}", []


# ==========================================
# 接口設定 (Endpoints)
# ==========================================
def create_adapter_routes(rag, api_key=None, top_k=60):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post("/query", response_model=QueryResponse, dependencies=[Depends(combined_auth)])
    async def query_text_adapter(request: QueryRequest):
        logger.info(f"🛡️ Adapter received query: {request.query}")
        ans, refs = await process_query_logic(request, rag)
        return QueryResponse(response=ans, references=refs if refs else None)

    @router.post("/query/stream", dependencies=[Depends(combined_auth)])
    async def query_stream_adapter(request: QueryRequest):
        logger.info(f"🌊 Stream endpoint triggered for: {request.query}")
        
        async def event_generator():
            yield json.dumps({"response": "⏳ **系統正在處理您的請求**...\n請稍候...\n\n---\n\n"}) + "\n"
            
            task = asyncio.create_task(process_query_logic(request, rag))
            
            while not task.done():
                done, pending = await asyncio.wait([task], timeout=3.0)
                if not done:
                    yield json.dumps({"response": "."}) + "\n"
            
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