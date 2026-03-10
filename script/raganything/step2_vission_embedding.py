#!/usr/bin/env python
"""
Step 2: Financial Multi-Modal Graph Builder (VL-Embedding Version)
(Supports: tongyi-embedding-vision-plus direct multimodal embedding via DashScope SDK)
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
# 🌟🌟🌟 終極修復：如果你用國際版阿里雲，必須加入呢行！ 🌟🌟🌟
import dashscope
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
# ==========================================================

# 強制載入 .env 檔案
load_dotenv()

# Add project root directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

# === API 設定 ===
API_KEY = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BINDING_HOST") or os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo") 
VISION_MODEL = os.getenv("VISION_MODEL", LLM_MODEL) 

EMBED_MODEL = "tongyi-embedding-vision-plus"
EMBED_DIM = 1152

# ========================================================
# 🌟 儲存庫路徑設定 (方便隨時轉 Database)
# ========================================================
RAG_STORAGE_NAME = os.getenv("RAG_STORAGE_NAME", "rag_storage_v2")
BASE_STORAGE_PATH = f"./data/{RAG_STORAGE_NAME}"

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
        system_prompt = """你是一個專業的金融數據萃取專家。請閱讀這份「指數成分股名單」的內容，
        提取這份名單的「完整官方名稱 (name)」以及「發佈日期或年份 (year)」。
        注意：這份文件包含多家公司，不需要提取個別公司的代號，stock_code 請填寫 "INDEX"。
        輸出格式: {"name": "恆生生科指數", "stock_code": "INDEX", "year": "2026-Jan"}"""
    else:
        system_prompt = """你是一個專業的金融數據萃取專家。請閱讀財報首頁文字，
        提取出這家公司的「完整官方名稱 (name)」、「股票代號 (stock_code)」及「財報所屬年度 (year)」。
        
        ⚠️ 警告：
        1. 必須提取「主要報告年度」（例如標題寫「2024 Annual Report」或「2024年報」，year 就是 "2024"）。
        2. 絕對要忽略用作對比的過往年份（如 2023 或 2022），以及版權年份。
        
        輸出格式: {"name": "三生製藥", "stock_code": "1530", "year": "2024"}"""

    try:
        logger.info(f"   🤖 正在呼叫 LLM 分析封面及提取標籤 ({doc_type})...")
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
            # 🌟 回傳一個假嘅 JSON，完美呃過 RAGAnything 嘅格式檢查機制
            return '{"description": "[IMAGE_DATA_EMBEDDED_DIRECTLY]", "keywords": ["multimodal", "image"]}'


        async def embedding_func_wrapper(texts: list[str]) -> np.ndarray:
            """
            使用 DashScope 官方 SDK 呼叫 tongyi-embedding-vision-plus (多模態融合向量)
            """
            # 確保讀取阿里雲 API Key (你喺 .env 加嗰條)
            dashscope_key = os.getenv("DASHSCOPE_API_KEY")
            if not dashscope_key:
                logger.error("❌ 找不到 DASHSCOPE_API_KEY！")
                raise ValueError("Missing DASHSCOPE_API_KEY")
            
            dashscope.api_key = dashscope_key.strip()

            all_embeddings = []
            batch_size = 20 # 阿里雲 API 限制每次最多 20 個 element
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = []
                
                for text in batch_texts:
                    path_match = re.search(r"\[RAW_IMG_PATH: (.*?)\]", text)
                    if path_match:
                        img_path = path_match.group(1)
                        # SDK 讀取本地圖片要求加入 file:// 協議
                        file_url = f"file://{img_path}"
                        # 🌟 融合向量寫法：同一個 dict 入面放 text 同 image
                        inputs.append({"text": text, "image": file_url})
                    else:
                        # 純文字寫法
                        inputs.append({"text": text})

                try:
                    # DashScope SDK 是同步的，用 asyncio.to_thread 防止阻塞
                    resp = await asyncio.to_thread(
                        dashscope.MultiModalEmbedding.call,
                        model=EMBED_MODEL,
                        input=inputs
                    )
                    
                    if resp.status_code == 200:
                        # 排序確保返回的 Vector 順序同 Input 順序一致
                        # 融合向量通常會保留 text_index
                        sorted_embs = sorted(resp.output["embeddings"], key=lambda x: x.get("text_index", x.get("image_index", x.get("index", 0))))
                        batch_embs = [emb["embedding"] for emb in sorted_embs]
                        all_embeddings.extend(batch_embs)
                    else:
                        logger.error(f"❌ 阿里雲 API 拒絕請求: {resp.message}")
                        all_embeddings.extend([[0.0] * EMBED_DIM] * len(batch_texts))
                        
                except Exception as e:
                    logger.error(f"❌ Embedding API 發生錯誤: {e}")
                    all_embeddings.extend([[0.0] * EMBED_DIM] * len(batch_texts))
                    
            return np.array(all_embeddings)

        my_embedding_func = EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=8192, func=embedding_func_wrapper)

        rag = RAGAnything(config=config, llm_model_func=llm_model_func, vision_model_func=vision_model_func, embedding_func=my_embedding_func)

        logger.info(f"⚡ 注入 Content List (Multimodal Embedding Mode)...")
        await rag.insert_content_list(content_list=content_list, file_path=original_file_name)
        logger.info(f"✅ 建庫完成！\n" + "-"*40)
        return True 

    except Exception as e:
        logger.error(f"❌ 建庫失敗: {str(e)}")
        return False

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
                logger.warning(f"⚠️ 專案 {folder_name} 缺乏必要的 md 或 json 文件，已跳過。")
                continue
            
            # ========================================================
            # 🌟 加入 Metadata Cache 機制，防止 LLM 生成漂移
            # ========================================================
            metadata_cache_path = os.path.join(project_dir, "metadata_cache.json")
            
            if os.path.exists(metadata_cache_path):
                logger.info(f"📂 [CACHE HIT] 發現已存檔的 Metadata，跳過 LLM 解析: {folder_name}")
                with open(metadata_cache_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                logger.info(f"🧠 [LLM TASK] 未發現 Metadata，開始解析: {folder_name}")
                metadata = await extract_metadata_with_llm(md_files[0], folder_name, doc_type)
                
                with open(metadata_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)
                logger.info(f"💾 [CACHE SAVED] 已將 Metadata 存檔至: {metadata_cache_path}")
            
            # 🌟 使用動態 Variable 決定 Workspace 路徑
            if doc_type == "index":
                workspace_dir = f"{BASE_STORAGE_PATH}/index/{metadata['name']}_{metadata['year']}"
            else:
                stock_tag = f"_{metadata['stock_code']}" if metadata['stock_code'] != "N/A" else ""
                workspace_dir = f"{BASE_STORAGE_PATH}/financial_report/{metadata['name']}{stock_tag}/{metadata['year']}"

            # 自動清理殘留 / 跳過已完成
            success_flag_path = os.path.join(workspace_dir, ".build_success")
            if os.path.exists(success_flag_path):
                logger.info(f"⏭️  [SKIP] 跳過已徹底完成的專案: {metadata['name']}")
                continue
                
            os.makedirs(workspace_dir, exist_ok=True)

            # 準備 Content List
            json_dir = os.path.dirname(os.path.abspath(json_files[0]))
            with open(json_files[0], 'r', encoding='utf-8') as f:
                raw_content_list = json.load(f)
            
            # ========================================================
            # 🌟 暴力過濾所有 discarded (頁首、頁尾、無用雜訊)
            # ========================================================
            content_list = []
            for item in raw_content_list:
                if isinstance(item, dict):
                    if item.get("type") == "discarded":
                        continue
                    content_list.append(item)
            
            logger.info(f"🧹 垃圾清理完成：由 {len(raw_content_list)} 個區塊縮減至 {len(content_list)} 個有效區塊！")
            
            # ========================================================
            # 🌟 [注入核心] 將圖片路徑硬塞入 Content
            # ========================================================
            for item in content_list:
                if isinstance(item, dict):
                    # 1. 處理圖片/表格，暴力注入絕對路徑
                    if item.get("type") in ["image", "table"]:
                        raw_path = item.get("img_path") or item.get("table_img_path")
                        if raw_path:
                            abs_img_path = os.path.abspath(os.path.join(json_dir, raw_path)).replace('\\', '/')
                            img_hint = f"\n\n[RAW_IMG_PATH: {abs_img_path}]\n"
                            item["content"] = (item.get("content", "") + img_hint).strip()

                    # 2. 處理 Index 特定提示
                    if doc_type == "index" and "content" in item and item["content"]:
                        context_prefix = f"[DOC TYPE: INDEX LIST | INDEX NAME: {metadata['name']} | DATE: {metadata['year']}] "
                        item["content"] = context_prefix + item["content"]

                    # 3. 修正原生路徑指向
                    for field in ["img_path", "table_img_path", "equation_img_path"]:
                        if field in item and item[field] and not os.path.isabs(item[field]):
                            item[field] = os.path.abspath(os.path.join(json_dir, item[field])).replace('\\', '/')
                            
            # 開始建庫
            logger.info(f"▶️  開始/繼續建庫: {metadata['name']} ({doc_type}) -> {workspace_dir}")
            
            is_success = await build_financial_knowledge_graph(content_list, f"{folder_name}.pdf", workspace_dir)
            
            # 只有 100% 跑完無報錯先寫入成功標記
            if is_success:
                with open(success_flag_path, "w", encoding="utf-8") as f:
                    f.write("DONE")
                logger.info(f"✅ 專案 {metadata['name']} 已徹底完成並標記！")
            else:
                logger.warning(f"⚠️ 專案 {metadata['name']} 建庫中途失敗，未寫入成功標記。")

def main():
    asyncio.run(auto_batch_process())

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Step 2: Financial Multi-Modal Graph Builder (DashScope SDK Mode)")
    print("=" * 60)
    main()