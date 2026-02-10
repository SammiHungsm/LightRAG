import os
import json
import asyncio
import base64
import sqlite3
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from lightrag.utils import logger, EmbeddingFunc
from lightrag import LightRAG

# 1. 嘗試從 config 引入全域設定 (支援 Module 或 Script 執行)
try:
    from .config import global_args
except ImportError:
    from config import global_args

# 動態引入 OpenAI/Azure 函數，如果無安裝 Azure 庫都不會報錯
from lightrag.llm.openai import openai_complete, openai_embed
try:
    from lightrag.llm.azure_openai import azure_openai_complete, azure_openai_embed
except ImportError:
    pass

load_dotenv()

class RagAnythingPipeline:
    def __init__(self, upload_dir: str, output_dir: str, sql_db_path: str, working_dir: str = "./rag_storage", rag_instance=None):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.step1_dir = self.output_dir / "step1_vlm_output"
        self.step2_dir = self.output_dir / "step2_output_granular"
        self.sql_db_path = sql_db_path
        self.working_dir = working_dir

        for p in [self.upload_dir, self.output_dir, self.step1_dir, self.step2_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # 2. Vision Client 改用 global_args 讀取設定
        self.azure_client = self._init_vision_client()
        self._init_db()

        self.rag = rag_instance
        if self.rag:
            logger.info("✅ Pipeline initialized with injected LightRAG instance.")

    def _init_vision_client(self):
        """利用 global_args 建立 Vision Client，減少重複代碼"""
        api_key = global_args.llm_binding_api_key
        endpoint = global_args.llm_binding_host
        # Vision Deployment 可能與 Chat 不同，這裡保留彈性讀取 env，否則 fallback 到預設 model
        self.vision_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", global_args.llm_model)

        if api_key and endpoint:
            try:
                from openai import AzureOpenAI
                if "openai/v1" in endpoint: endpoint = endpoint.split("/openai/v1")[0]
                return AzureOpenAI(api_key=api_key, api_version="2024-02-15-preview", azure_endpoint=endpoint)
            except ImportError:
                logger.warning("⚠️ 'openai' package missing.")
        return None

    def _init_db(self):
        """保留你的專屬 DB Schema"""
        try:
            with sqlite3.connect(self.sql_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS financial_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_code TEXT, report_year INTEGER, metric_name TEXT,
                        metric_value REAL, unit TEXT, source_file TEXT, page_number INTEGER,
                        original_text TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
        except Exception as e:
            logger.error(f"❌ DB Init failed: {e}")

    async def initialize_rag(self):
        """Standalone 模式下，根據 global_args 動態啟動 LightRAG"""
        if self.rag: return

        logger.info("🚀 Initializing Standalone LightRAG (Dynamic Config)...")
        binding = global_args.llm_binding
        
        # 3. 動態決定 LLM Function (不 Hardcode)
        if binding == "azure_openai":
            llm_func = azure_openai_complete
            embed_func_core = azure_openai_embed
        else:
            llm_func = openai_complete
            embed_func_core = openai_embed

        # 4. 動態 Embedding Wrapper
        async def dynamic_embedding_wrapper(texts: list[str]) -> np.ndarray:
            return await embed_func_core.func(
                texts=texts,
                model=global_args.embedding_model, 
                api_key=global_args.embedding_binding_api_key or global_args.llm_binding_api_key,
                base_url=global_args.embedding_binding_host or global_args.llm_binding_host,
            )

        try:
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=llm_func,
                llm_model_name=global_args.llm_model,
                embedding_func=EmbeddingFunc(
                    embedding_dim=global_args.embedding_dim or 1536,
                    max_token_size=8192,
                    func=dynamic_embedding_wrapper
                ),
                chunk_token_size=global_args.chunk_size,
                chunk_overlap_token_size=global_args.chunk_overlap_size,
            )
            await self.rag.initialize_storages()
            logger.success(f"✅ LightRAG Initialized using binding: {binding}")
        except Exception as e:
            logger.error(f"❌ LightRAG Init Failed: {e}")
            raise e

    # === Helpers (精簡邏輯) ===
    def _encode_image(self, image_path: str) -> Optional[str]:
        if not os.path.exists(image_path): return None
        with open(image_path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

    def _get_safe_content(self, item: Dict) -> str:
        for k in ["text", "content", "table_body", "image_caption", "table_caption"]:
            val = item.get(k)
            if isinstance(val, list): val = "".join(val)
            if val and str(val).strip(): return str(val).strip()
        return ""

    def _find_real_image_path(self, file_stem: str, rel_path: str) -> Optional[str]:
        if not rel_path: return None
        img_name = os.path.basename(rel_path)
        for sub in ["auto/images", "images", ""]:
            p = self.step1_dir / file_stem / sub / img_name
            if p.exists(): return str(p)
        return None

    # === Step 1: Mineru OCR ===
    async def run_mineru_extraction(self, file_path: str) -> str:
        fpath = Path(file_path)
        fstem = fpath.stem
        
        possible_json = [
            self.step1_dir / fstem / "auto" / f"{fstem}_content_list.json",
            self.step1_dir / fstem / "auto" / "content_list.json"
        ]
        for p in possible_json:
            if p.exists():
                logger.info(f"⚡ Output exists: {p}")
                return str(p)

        logger.info(f"🚀 Running Mineru on {fpath.name}...")
        cmd = ["magic-pdf", "-p", str(fpath), "-o", str(self.step1_dir), "-m", "auto"]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"❌ Mineru Error: {stderr.decode()}")
            raise Exception("Mineru Failed")

        for p in possible_json:
             if p.exists(): return str(p)
        
        found = list((self.step1_dir / fstem).rglob("*content_list.json"))
        if found: return str(found[0])
        raise FileNotFoundError(f"Mineru generated no JSON for {fstem}")

    # === Step 2: Vision ETL (保留邏輯，簡化調用) ===
    async def _call_vision_llm(self, img_path: str, mode: str = "table", context="" ) -> Optional[str]:
        if not self.azure_client: return None
        b64_img = await asyncio.to_thread(self._encode_image, img_path)
        if not b64_img: return None

        sys_prompt = "You are a Financial Data Extractor." if mode == "table" else "Describe for RAG."
        try:
            resp = await asyncio.to_thread(
                self.azure_client.chat.completions.create,
                model=self.vision_deployment,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Context: {context}\nExtract data."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]}
                ],
                temperature=0.1, max_tokens=2048,
                response_format={"type": "json_object"} if mode == "table" else None
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Vision LLM Error: {e}")
            return None

    async def run_hybrid_etl(self, json_path: str, file_name: str) -> List[Dict[str, Any]]:
        logger.info(f"🚀 Starting Hybrid ETL: {file_name}")
        with open(json_path, 'r', encoding='utf-8') as f: content_list = json.load(f)
        
        fstem = Path(file_name).stem
        chunks = []
        sql_buffer = [] 
        
        for idx, item in enumerate(content_list):
            itype = item.get('type', 'text')
            content = self._get_safe_content(item)
            img_path = self._find_real_image_path(fstem, item.get('img_path', ''))
            prev_ctx = self._get_safe_content(content_list[idx-1])[-200:] if idx > 0 else ""

            if itype in ['table', 'tabular', 'image', 'figure'] and img_path and self.azure_client:
                mode = "table" if "table" in itype else "caption"
                ai_res = await self._call_vision_llm(img_path, mode, prev_ctx)
                if ai_res:
                    content = f"{content}\n\n[AI Analysis]: {ai_res}"
                    if mode == "table":
                        try:
                            metrics = json.loads(ai_res).get("metrics", [])
                            for m in metrics:
                                sql_buffer.append((
                                    "Unknown", 2024, m.get('metric_name'), m.get('value'), 
                                    m.get('unit'), file_name, item.get('page_idx', 0), str(m)
                                ))
                        except: pass

            if content:
                chunks.append({
                    "content": content,
                    "metadata": {"source": file_name, "page": item.get('page_idx', 0), "type": itype, "priority": "NORMAL"}
                })

        # 批量插入 SQL (比舊版更快更簡潔)
        if sql_buffer:
            with sqlite3.connect(self.sql_db_path) as conn:
                conn.executemany("""
                    INSERT INTO financial_metrics (company_code, report_year, metric_name, metric_value, unit, source_file, page_number, original_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, sql_buffer)

        if chunks:
            out_path = self.step2_dir / fstem / "granular_content.json"
            out_path.parent.mkdir(exist_ok=True, parents=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
                
        return chunks

    # === Step 3: Ingest ===
    async def ingest_to_lightrag(self, chunks: List[Dict[str, Any]], doc_id: str):
        if not self.rag: await self.initialize_rag()
        
        logger.info(f"🚀 Ingesting {len(chunks)} chunks...")
        rag_chunks = [{
            "content": c['content'],
            "priority": c['metadata']['priority'],
            "page_info": f"Page {c['metadata']['page']}",
            "file_path": c['metadata']['source']
        } for c in chunks]
        
        full_text = "\n\n".join(c['content'] for c in chunks)
        await self.rag.ainsert_structured_chunks(full_text=full_text, text_chunks=rag_chunks, doc_id=doc_id)
        logger.success(f"✅ Ingested: {doc_id}")

    # === Main Entry ===
    async def process_document(self, file_path: str):
        fname = os.path.basename(file_path)
        try:
            jpath = await self.run_mineru_extraction(file_path)
            chunks = await self.run_hybrid_etl(jpath, fname)
            if chunks: await self.ingest_to_lightrag(chunks, fname)
            return {"status": "success", "doc_id": fname}
        except Exception as e:
            logger.exception(f"🔥 Failure: {fname}")
            return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    async def main():
        # Standalone Test
        p = RagAnythingPipeline("./data/input", "./data/output", "./data/financial.db", "./data/rag_storage")
        logger.info("Pipeline Ready.")
    asyncio.run(main())