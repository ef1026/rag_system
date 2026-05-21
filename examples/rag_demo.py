#!/usr/bin/env python
import os
import argparse
import asyncio
import logging
from pathlib import Path
import sys

# 1. 开启国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def process_with_rag(file_path, output_dir, api_key, base_url=None, working_dir=None, parser=None):
    try:
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # === 【核心修改】增加输出清洗逻辑 ===
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            # 1. 强制在 System Prompt 里要求纯 JSON
            safety_prompt = " Output MUST be raw JSON only. No Markdown. No ```json wrapper."
            if system_prompt:
                system_prompt += safety_prompt
            else:
                system_prompt = "You are a helpful assistant." + safety_prompt

            # 2. 调用模型
            response = await openai_complete_if_cache(
                "deepseek-chat",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
            
            # 3. 【给DeepSeek擦嘴】强制清理 Markdown 标记
            if isinstance(response, str):
                cleaned_response = response.replace("```json", "").replace("```", "").strip()
                return cleaned_response
            return response

        # 视觉函数也做同样处理
        async def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
            # 转为纯文本调用
            return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        print("\n[System] 加载本地 Embedding 模型...")
        local_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        async def _async_embedding_func(texts):
            return await asyncio.to_thread(lambda: local_model.encode(texts))

        embedding_func = EmbeddingFunc(
            embedding_dim=384,
            max_token_size=512,
            func=_async_embedding_func
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        logger.info("\nQuerying processed document:")
        # 换一个更具体的问题，容易命中关键词
        text_queries = ["What is the main content of this document?","What is Fourier Transform?"]

        for query in text_queries:
            logger.info(f"\n[Text Query]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"Answer: {result}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--api-key", default=os.getenv("LLM_BINDING_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BINDING_HOST"))
    
    args = parser.parse_args()
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./rag_storage", exist_ok=True)

    asyncio.run(process_with_rag(
        args.file_path, "./output", args.api_key, args.base_url, "./rag_storage", "mineru"
    ))

if __name__ == "__main__":
    main()