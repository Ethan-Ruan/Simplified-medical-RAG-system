import streamlit as st
import time
import os
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, COLLECTION_NAME, EMBEDDING_DIM,
    INDEX_TYPE, INDEX_PARAMS, INDEX_METRIC_TYPE,
    id_to_doc_map
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from milvus_utils import (
    init_milvus_connection,
    get_or_create_collection,
    index_data_if_needed,
    search_similar_documents
)
from rag_core import generate_answer

# 设置 HF 缓存
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache'

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📄 医疗 RAG 系统 (Milvus Standalone)")
st.markdown(f"使用 Milvus Standalone, `{EMBEDDING_MODEL_NAME}`, 和 `{GENERATION_MODEL_NAME}`。")

# 初始化 Milvus
milvus_client = init_milvus_connection()

if milvus_client:
    # 获取或创建 collection
    collection = get_or_create_collection(
        alias=milvus_client,
        collection_name=COLLECTION_NAME,
        embedding_dim=EMBEDDING_DIM,
        index_type=INDEX_TYPE,
        index_params=INDEX_PARAMS,
        index_metric_type=INDEX_METRIC_TYPE
    )

    # 加载模型
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)

    if collection and embedding_model and generation_model and tokenizer:
        # 加载数据
        data = load_data(DATA_FILE)

        # 插入数据（兼容旧版函数签名）
        indexed = index_data_if_needed(
            collection=collection,
            data=data,
            embedding_model=embedding_model,
            max_articles=MAX_ARTICLES_TO_INDEX
        )

        st.divider()
        if not indexed and not id_to_doc_map:
            st.error("索引失败，且映射为空，RAG 功能禁用。")
        else:
            query = st.text_input("请输入问题：", key="query_input")
            if st.button("获取答案") and query:
                start = time.time()
                with st.spinner("检索中..."):
                    ids, dists = search_similar_documents(collection, query, embedding_model)
                if not ids:
                    st.warning("未检索到相关文档。")
                else:
                    docs = [id_to_doc_map[i] for i in ids if i in id_to_doc_map]
                    for idx, doc in enumerate(docs):
                        dist_str = f" (距离 {dists[idx]:.4f})"
                        with st.expander(f"文档 {idx+1}{dist_str} - {doc['title']}"):
                            st.write(doc['abstract'])
                    st.divider()
                    with st.spinner("生成答案..."):
                        answer = generate_answer(query, docs, generation_model, tokenizer)
                        st.subheader("答案：")
                        st.write(answer)
                st.info(f"耗时 {time.time()-start:.2f} 秒")
    else:
        st.error("模型加载或 collection 准备失败，请检查日志。")
else:
    st.error("Milvus 连接失败，请检查服务。")

# 侧边栏配置
st.sidebar.header("系统配置")
st.sidebar.markdown(f"**Collection:** {COLLECTION_NAME}")
st.sidebar.markdown(f"**数据文件:** {DATA_FILE}")
st.sidebar.markdown(f"**嵌入模型:** {EMBEDDING_MODEL_NAME}")
st.sidebar.markdown(f"**生成模型:** {GENERATION_MODEL_NAME}")
st.sidebar.markdown(f"**最大索引数:** {MAX_ARTICLES_TO_INDEX}")
st.sidebar.markdown(f"**检索 Top K:** {TOP_K}")