import streamlit as st
import torch
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P, REPETITION_PENALTY, USE_RERANKER, RERANKER_MODEL
from sentence_transformers import CrossEncoder

def generate_answer(query, context_docs, gen_model, tokenizer, history=None):
    """Generates an answer using the LLM based on query, context and conversation history."""
    if not context_docs:
        return "I couldn't find relevant documents to answer your question."
    if not gen_model or not tokenizer:
         st.error("Generation model or tokenizer not available.")
         return "Error: Generation components not loaded."

    # 新增对话历史处理
    history_str = ""
    if history:
        history_str = "\n\n对话历史：\n" + "\n".join([f"- 用户: {h['query']}\n- 助手: {h['answer']}" for h in history[-3:]])  # 保留最近3轮对话

    # 使用重新排序模块（新增历史感知的查询扩展）
    if USE_RERANKER and len(context_docs) > 1:
        try:
            reranker = CrossEncoder(RERANKER_MODEL)
            # 结合历史信息构建增强查询（添加边界检查）
            enhanced_queries = [query]  # 默认包含当前查询
            if history and len(history) >= 1:  # 添加历史存在性检查
                enhanced_queries.extend([f"{query} [历史上下文: {h['query']}]" for h in history[-1:]])
                
            # 生成文档对时添加空值保护
            if not enhanced_queries or not context_docs:
                st.warning("无法进行重排序：查询或上下文为空")
                return context_docs
                
            pairs = [(eq, doc['content']) for eq in enhanced_queries for doc in context_docs]
            scores = reranker.predict(pairs)
            
            # 添加分数与文档数量一致性检查
            if len(scores) != len(pairs):
                st.error("重排序分数与文档对数量不匹配")
                return context_docs
                
            # 使用安全的排序方式
            sorted_indices = sorted(range(len(context_docs)), 
                                  key=lambda i: max(scores[i::len(context_docs)]),  # 取每个文档的最大分数
                                  reverse=True)
            context_docs = [context_docs[i] for i in sorted_indices]
            st.info(f"✅ 使用{RERANKER_MODEL}模型完成相关性重排序")
        except Exception as e:
            st.error(f"重新排序失败: {e}")

    context = "\n\n---\n\n".join([doc['content'] for doc in context_docs]) # Combine retrieved docs

    # 更新提示模板（新增对话历史和多轮优化说明）
    prompt = f"""基于以下上下文文档和对话历史，逐步分析问题并优化最终回答：
    
{history_str}

上下文文档元数据：
{'\n'.join([f'- 文档[{doc["id"]}] 来源: {doc["source_file"]} (分块: {doc["chunk_index"]})' for doc in context_docs])}

上下文内容：
{context}
用户问题：{query}
请按以下步骤思考：
1. 分析历史对话中的关键信息
2. 识别当前问题与历史问题的关联
3. 结合上下文文档进行多角度验证
4. 整合信息生成最终回答：
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_GEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id # Important for open-end generation
            )
        # Decode only the newly generated tokens, excluding the prompt
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return "Sorry, I encountered an error while generating the answer." 

    # 在问答处理逻辑后更新历史记录（移动到函数内部）
    st.session_state.chat_history.append({
        'query': query,
        'answer': response,
        'context_ids': [doc['id'] for doc in context_docs]
    })
    return response.strip()

# 在Streamlit应用中添加session_state管理（示例）
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

