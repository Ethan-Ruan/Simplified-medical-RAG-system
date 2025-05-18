import streamlit as st
import re
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from config import (
    COLLECTION_NAME, EMBEDDING_DIM,
    INDEX_TYPE, INDEX_PARAMS, INDEX_METRIC_TYPE,
    SEARCH_PARAMS, TOP_K, MAX_ARTICLES_TO_INDEX,
    id_to_doc_map
)


def init_milvus_connection(host: str = "localhost", port: str = "19530") -> str:
    """
    Connects to a Milvus Standalone instance and returns the connection alias.
    """
    try:
        connections.connect(alias="default", host=host, port=port)
        st.success("Connected to Milvus Standalone!")
        return "default"
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {e}")
        return None


def get_or_create_collection(
    alias: str,
    collection_name: str,
    embedding_dim: int,
    index_type: str,
    index_params: dict,
    index_metric_type: str
) -> Collection:
    """
    Ensures the specified collection exists in Milvus with correct schema.
    Drops and recreates the collection if it exists with incompatible schema.
    Returns the Collection object or None on failure.
    """
    try:
        # List existing collections
        existing = utility.list_collections()

        # If collection exists, drop it to ensure fresh schema
        if collection_name in existing:
            utility.drop_collection(collection_name)
            st.info(f"Dropped existing collection '{collection_name}' to apply updated schema.")

        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=2048),
        ]
        schema = CollectionSchema(fields, description="RAG collection with dim={embedding_dim}")

        # Create collection
        collection = Collection(name=collection_name, schema=schema, using=alias)
        st.success(f"Created collection '{collection_name}' with dim={embedding_dim}.")

        # Create index on vector field
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": index_type,
                "metric_type": index_metric_type,
                "params": index_params
            }
        )
        st.info(f"Created index ({index_type}) on '{collection_name}'.")

        # Load collection into memory for search
        collection.load()
        st.success(f"Collection '{collection_name}' is loaded and ready.")
        return collection

    except Exception as e:
        st.error(f"Error setting up Milvus collection: {e}")
        return None


def index_data_if_needed(
    collection: Collection,
    data: list,
    embedding_model,
    max_articles: int = MAX_ARTICLES_TO_INDEX
) -> bool:
    """
    Indexes documents into Milvus if not already indexed.
    Updates global id_to_doc_map. Returns True on success.
    """
    if collection is None:
        st.error("Milvus collection not available.")
        return False

    # Check current entity count
    try:
        current_count = collection.num_entities
    except Exception:
        current_count = 0
    st.write(f"Currently in collection: {current_count} entities.")

    # Prepare documents for embedding
    to_index = data[:max_articles]
    docs, temp_map = [], {}
    skipped_count = 0  # 新增跳过计数器
    for i, doc in enumerate(to_index):
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        source_file = doc.get('source_file', '')
        chunk_idx = doc.get('chunk_index', 0)
        # 增强内容构造：包含标题+来源+分块信息+摘要
        content = f"Title: {title}\nSource: {source_file} (Chunk {chunk_idx})\nAbstract: {abstract}".strip()
        
        # 新增数据清洗逻辑
        def clean_content(text):
            """清洗微信公众号广告和操作引导"""
            # 定义广告关键词黑名单
            ad_keywords = [
                '点击关注', '扫码领取', '立即注册', '限时优惠',
                '添加客服', '立即购买', '点击阅读原文', '领取福利'
            ]
            # 去除HTML标签
            text = re.sub(r'<[^>]+>', '', text)
            # 检查是否包含广告关键词
            if any(keyword in text for keyword in ad_keywords):
                return None
            return text.strip()
        
        cleaned_content = clean_content(content)
        if not cleaned_content:
            skipped_count += 1
            continue
            
        docs.append(cleaned_content)
        # 确保元数据包含完整字段
        temp_map[i] = {
            'id': i,
            'title': title,
            'abstract': abstract,
            'source_file': source_file,
            'chunk_index': chunk_idx,
            'content': cleaned_content  # 存储清洗后的内容
        }

    # 新增清洗结果提示
    if skipped_count > 0:
        st.info(f"过滤 {skipped_count} 条包含广告/引导的内容")
        
    needed = len(docs)
    if current_count < needed and docs:
        st.warning(f"Indexing {needed - current_count} new documents...")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        ids = list(range(needed))
        previews = [c[:500] for c in docs]

        # Insert into Milvus
        try:
            collection.insert([ids, embeddings, previews])
            collection.flush()
            id_to_doc_map.update(temp_map)
            st.success(f"Inserted {needed} documents into Milvus.")
            return True
        except Exception as e:
            st.error(f"Error inserting data: {e}")
            return False
    else:
        st.write("No new documents to index.")
        if not id_to_doc_map:
            id_to_doc_map.update(temp_map)
        return True


def search_similar_documents(
    collection: Collection,
    query: str,
    embedding_model
) -> tuple:
    """
    Searches the Milvus collection for documents similar to the query.
    Returns two lists: IDs and distances.
    """
    if collection is None:
        st.error("Milvus collection not available.")
        return [], []

    try:
        q_emb = embedding_model.encode([query])[0]
        results = collection.search(
            data=[q_emb],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=TOP_K,
            output_fields=["id"]
        )
        if not results or not results[0]:
            return [], []
        ids = [hit.entity.get("id") for hit in results[0]]
        dists = [hit.distance for hit in results[0]]
        return ids, dists
    except Exception as e:
        st.error(f"Search failed: {e}")
        return [], []

