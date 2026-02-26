#!/usr/bin/env python
"""
Step 2: Financial Multi-Modal Graph Builder (Category-Aware Version)
(Supports: Index mapping & Financial reports with context-injection)
"""

import os
import argparse
import asyncio
import logging
import glob
import json
import re
import shutil
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# 強制載入 .env 檔案
load_dotenv()

# Add project root directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

# === API 設定 ===
API_KEY = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BINDING_HOST") or os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo") 
VISION_MODEL = os.getenv("VISION_MODEL", LLM_MODEL) 
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

logger.setLevel(logging.INFO)
set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")

async def extract_metadata_with_llm(md_file_path: str, fallback_name: str, doc_type: str) -> dict:
    """根據 doc_type 用唔同嘅 Prompt 抽 Metadata，用於建立資料夾路徑"""
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            cover_text = f.read(2000) # 讀多啲，確保讀到指數日期
    except Exception:
        return {"name": fallback_name, "stock_code": "N/A", "year": "2024"}

    if doc_type == "index":
        # 🌟 針對指數：攞指數名同呢份名單嘅發佈日期
        system_prompt = """你是一個專業的金融數據萃取專家。請閱讀這份「指數成分股名單」的內容，
        提取這份名單的「完整官方名稱 (name)」以及「發佈日期或年份 (year)」。
        注意：這份文件包含多家公司，不需要提取個別公司的代號，stock_code 請填寫 "INDEX"。
        輸出格式: {"name": "恆生生科指數", "stock_code": "INDEX", "year": "2026-Jan"}"""
    else:
        # 🌟 針對財報：精準攞單一公司嘅 Metadata
        system_prompt = """你是一個專業的金融數據萃取專家。請閱讀財報封面，
        提取出這家公司的「完整官方名稱 (name)」、「股票代號 (stock_code)」及「財報年份 (year)」。
        輸出格式: {"name": "三生製藥", "stock_code": "1530", "year": "2024"}"""

    try:
        logger.info(f"   🤖 正在分析封面及提取標籤 ({doc_type})...")
        response = await openai_complete_if_cache(
            model=LLM_MODEL, prompt=cover_text, system_prompt=system_prompt,
            api_key=API_KEY, base_url=BASE_URL
        )
        match = re.search(r'\{.*\}', response.replace('\n', ''), re.DOTALL)
        metadata = json.loads(match.group(0)) if match else {}
        
        # 清洗名稱，確保 Folder 名合法
        raw_name = metadata.get("name", fallback_name)
        safe_name = re.sub(r'[\\/*?:"<>|]', "", raw_name).strip()
        
        return {
            "name": safe_name,
            "stock_code": metadata.get("stock_code", "N/A"),
            "year": str(metadata.get("year", "2024"))
        }
    except Exception as e:
        logger.error(f"❌ LLM Metadata 萃取失敗: {e}")
        return {"name": fallback_name, "stock_code": "N/A", "year": "2024"}

async def build_financial_knowledge_graph(content_list: list, original_file_name: str, workspace_dir: str):
    """通用建庫函數 (RAG-Anything 核心)"""
    os.makedirs(workspace_dir, exist_ok=True)
    logger.info(f"📁 初始化 Workspace: {workspace_dir}")

    try:
        config = RAGAnythingConfig(
            working_dir=workspace_dir, 
            enable_image_processing=True,
            enable_table_processing=True, 
            enable_equation_processing=False, 
        )

        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await openai_complete_if_cache(
                model=LLM_MODEL, prompt=prompt, system_prompt=system_prompt,
                history_messages=history_messages, api_key=API_KEY, base_url=BASE_URL, **kwargs
            )

        async def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
            if image_data:
                return await openai_complete_if_cache(
                    model=VISION_MODEL, prompt="", system_prompt=None, history_messages=[],
                    messages=[
                        {"role": "system", "content": "You are an expert in financial chart analysis. Please extract stock codes, names, and exact weightings from tables or charts."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        ]}
                    ],
                    api_key=API_KEY, base_url=BASE_URL, **kwargs
                )
            return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        async def embedding_func_wrapper(texts: list[str]) -> np.ndarray:
            return await openai_embed.func(texts=texts, model=EMBED_MODEL, api_key=API_KEY, base_url=BASE_URL)

        my_embedding_func = EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=8192, func=embedding_func_wrapper)

        rag = RAGAnything(config=config, llm_model_func=llm_model_func, vision_model_func=vision_model_func, embedding_func=my_embedding_func)

        logger.info(f"⚡ 注入 Content List (Multimodal Mode)...")
        await rag.insert_content_list(content_list=content_list, file_path=original_file_name)
        logger.info(f"✅ 建庫完成！\n" + "-"*40)

    except Exception as e:
        logger.error(f"❌ 建庫失敗: {str(e)}")

async def auto_batch_process():
    base_mineru_dir = "./data/output/step1_vlm_output"
    doc_types = ["index", "financial_report"]

    for doc_type in doc_types:
        category_path = os.path.join(base_mineru_dir, doc_type)
        if not os.path.exists(category_path):
            continue

        projects = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
        logger.info(f"🔍 掃描類別 [{doc_type}]: 發現 {len(projects)} 個專案。")

        for folder_name in projects:
            project_dir = os.path.join(category_path, folder_name)
            
            md_files = glob.glob(os.path.join(project_dir, "**/*.md"), recursive=True)
            json_files = glob.glob(os.path.join(project_dir, "**/*_content_list.json"), recursive=True)
            
            if not md_files or not json_files:
                continue
            
            # 1. 抽 Metadata (用於路由資料夾)
            metadata = await extract_metadata_with_llm(md_files[0], folder_name, doc_type)
            
            # 2. 決定 Workspace 路徑 (Index 同財報分家)
            if doc_type == "index":
                # 指數路徑: rag_storage/index/恆生生科指數_2026-Jan
                workspace_dir = f"./data/rag_storage/index/{metadata['name']}_{metadata['year']}"
            else:
                # 財報路徑: rag_storage/financial_report/三生製藥_1530/2024
                stock_tag = f"_{metadata['stock_code']}" if metadata['stock_code'] != "N/A" else ""
                workspace_dir = f"./data/rag_storage/financial_report/{metadata['name']}{stock_tag}/{metadata['year']}"

            # 3. 自動清理殘留
            if os.path.exists(workspace_dir):
                if os.path.exists(os.path.join(workspace_dir, "graph_chunk_entity_relation.graphml")):
                    logger.info(f"⏭️  跳過已完成專案: {metadata['name']}")
                    continue
                shutil.rmtree(workspace_dir)

            # 4. 準備 Content List 並注入「控制上下文 (Context Control)」
            json_dir = os.path.dirname(os.path.abspath(json_files[0]))
            with open(json_files[0], 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            for item in content_list:
                if isinstance(item, dict):
                    # 🌟 核心控制：如果是 Index，喺 Content 注入強烈提示，確保圖譜抽到 Stock Code
                    if doc_type == "index" and "content" in item and item["content"]:
                        context_prefix = f"[DOC TYPE: INDEX LIST | INDEX NAME: {metadata['name']} | DATE: {metadata['year']}] "
                        item["content"] = context_prefix + item["content"]

                    # 修正圖片路徑
                    for field in ["img_path", "table_img_path", "equation_img_path"]:
                        if field in item and item[field] and not os.path.isabs(item[field]):
                            item[field] = os.path.abspath(os.path.join(json_dir, item[field])).replace('\\', '/')

            # 5. 開始建庫
            logger.info(f"▶️  處理中: {metadata['name']} ({doc_type})")
            await build_financial_knowledge_graph(content_list, f"{folder_name}.pdf", workspace_dir)

def main():
    asyncio.run(auto_batch_process())

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Step 2: Financial Multi-Modal Graph Builder (Dual-Track Context Mode)")
    print("=" * 60)
    main()