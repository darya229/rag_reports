import re
import ast
import streamlit as st
import os
import pandas as pd
from qdrant_client.models import models
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from fastembed import SparseTextEmbedding 
import yadisk
API_QDRANT=os.getenv("API_QDRANT")
API_DISK=os.getenv("API_DISK")

y = yadisk.YaDisk(token=API_DISK)
#--------INITIALIZE CONNECTIONS ONCE -----------
@st.cache_resource(ttl=3600)
def initialize_connections():
    """Инициализация всех подключений один раз при запуске - кэшируется"""
    try:
        client_db = QdrantClient(
            url = "http://176.109.105.181:6333/",
            api_key=API_QDRANT
)
        bm25_model = SparseTextEmbedding("Qdrant/bm25")
        dense_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
        
        return client_db, bm25_model, dense_model, cross_encoder_model
    except Exception as e:
        st.error(f"❌ Ошибка инициализации: {e}")
        return None, None, None, None

# Инициализируем подключения через cache_resource
client, bm25_model, dense_model, cross_encoder_model = initialize_connections()

# Сохраняем в session_state для доступа из всех страниц
if client is not None:
    st.session_state.client = client
    st.session_state.bm25_model = bm25_model  
    st.session_state.dense_model = dense_model
    st.session_state.cross_encoder_model = cross_encoder_model

# Проверяем успешность инициализации
if None in [client, bm25_model, dense_model]:
    st.error("Не удалось инициализировать приложение. Пожалуйста, обновите страницу.")
    st.stop()


client = st.session_state.client
dense_embedding_model = st.session_state.dense_model
bm25_embedding_model = st.session_state.bm25_model
cross_encoder_model = st.session_state.cross_encoder_model

def extract_tables(text):
    """
    Извлекает из текста все списки вида ['Company', '2025 Capex', ...]
    
    Args:
        text (str): Входной текст
        
    Returns:
        list or None: Список найденных списков или None, если ничего не найдено
    """
    if not text or not isinstance(text, str):
        return None
    
    # Паттерн для поиска списков Python в тексте
    # Ищем конструкции, которые начинаются с [ и заканчиваются на ]
    pattern = r'\[[^\]]*\]'
    
    # Находим все потенциальные списки
    matches = re.findall(pattern, text)
    
    result = []
    for match in matches:
        try:
            # Пробуем распарсить найденную строку как Python-объект
            parsed_list = ast.literal_eval(match)
            result.append(parsed_list)
            
        except (SyntaxError, ValueError):
            # Если не удалось распарсить, пропускаем
            continue
    
    return result if result else None


def search_separately(query_text, collection_name, dense_embedding_model, bm25_embedding_model, client):
    """Поиск отдельно по dense и sparse векторам"""

    # dense_vector = dense_embedding_model.embed_query(query_text)
    dense_vector = dense_embedding_model.encode(query_text)
    sparse_vector = next(bm25_embedding_model.query_embed(query_text))
    
    # Поиск по dense векторам
    dense_results = client.query_points(
        collection_name=collection_name,
        query=dense_vector.tolist() if hasattr(dense_vector, 'tolist') else dense_vector,
        using="qwen",  # указываем имя вектора отдельно
        with_payload=True,
        with_vectors=False,
        limit=50
    )
    
    #Добавляем тег типа поиска
    for point in dense_results.points:
        point.payload['search_type'] = 'Семантический'

    # Поиск по sparse векторам
    sparse_vector_obj = models.SparseVector(**sparse_vector.as_object())
    sparse_results = client.query_points(
        collection_name=collection_name,
        query=sparse_vector_obj,
        using="bm25",  # указываем имя вектора отдельно
        with_payload=True,
        with_vectors=False,
        limit=50
    )

    #Добавляем тег типа поиска
    for point in sparse_results.points:
        point.payload['search_type'] = 'Ключевые слова'

    #
    return dense_results.points, sparse_results.points


def rerank_snippets(query_text, dense_results, sparse_results, cross_encoder_model):

    # Удаляем дубликаты

    chunks_id = set()
    chunks = []
    for item in sparse_results + dense_results:
        if item.id not in chunks_id:
            chunks_id.add(item.id)
            chunks.append(item)
        else:
            chunks_id_list = list(chunks_id)
            dublicate_chunk_index = chunks_id_list.index(item.id)
            dublicate_chunk = chunks[dublicate_chunk_index]
            dublicate_chunk.payload['search_type'] = str(dublicate_chunk.payload['search_type']) + ',' + item.payload['search_type']


    #Создаем входные данные для кросс энкодера

    inputs = [(query_text, chunk.payload['page_content']) for chunk in chunks]
    # softmax_fn = Softmax(dim=0)
    rerank_scores = cross_encoder_model.predict(inputs)
    # rerank_scores = cross_encoder_model.compute_score(inputs, normalize=True)

    #Добавляем к сниппетам rerank_score

    for i in range(len(chunks)):
        chunks[i].payload['rerank_score'] = rerank_scores[i]

    # Ранжируем результаты

    reranked_chunks = []

    rerank_scores_sorted_desc = np.sort(rerank_scores)[::-1]
    for score in rerank_scores_sorted_desc:
        for chunk in chunks:
            if chunk.payload['rerank_score'] == score:
                reranked_chunks.append(chunk)
            else:
                pass

    return reranked_chunks


def hybrid_rerank_search(query_text, dense_embedding_model, bm25_embedding_model, client,  cross_encoder_model, collection_name):

    dense_results, sparse_results = search_separately(query_text=query_text, 
                                                      dense_embedding_model=dense_embedding_model, 
                                                      bm25_embedding_model=bm25_embedding_model, 
                                                      client=client, 
                                                      collection_name=collection_name)

    reranked_snippets = rerank_snippets(query_text, dense_results, sparse_results, cross_encoder_model)

    return reranked_snippets


def retriev_chunks(query: str):
    all_snippets = []
    df = pd.DataFrame()
    reranked_snippets = hybrid_rerank_search(query_text=query,
                                            dense_embedding_model=dense_embedding_model, bm25_embedding_model=bm25_embedding_model, cross_encoder_model=cross_encoder_model, client=client,
                                            collection_name='jan2026_reports_with_metadata')
    
    ##### подставляем таблицы ########

    files = os.listdir("documents_elements_paddle_tables_jan") #для таблиц
    for snippet in reranked_snippets:

        tables = extract_tables(snippet.payload["page_content"])
        if tables:
            doc_filename=snippet.payload["metadata"]["file_name"].replace(".pdf", ".feather")
            if doc_filename in files:
                doc = pd.read_feather(f"documents_elements_paddle_tables_jan/{doc_filename}")
                tables_head_content = doc["table_head_content"].to_list()
                tables_full_content = doc["element_content"].to_list()
                for table in tables:
                    init_content = snippet.payload["page_content"]
                    try:
                        tables_head_content_index = tables_head_content.index(str(table))
                        full_table = tables_full_content[tables_head_content_index]
                        snippet.payload["page_content"] = init_content.replace(str(table), str(full_table))
                    except:
                        pass

    ##### подставляем таблицы ########
    rerank_snippets_active = reranked_snippets[:15]
    all_snippets.append(reranked_snippets)
    context = "\n".join([
        f"<snippet {idx + 1}> {item.payload['page_content']} \n doc_title: {item.payload["metadata"]["file_name"]} \n page: {item.payload["metadata"]["page"]}</snippet {idx + 1}>"
        for idx, item in enumerate(rerank_snippets_active)
        if item.payload and 'page_content' in item.payload and item.payload['page_content']
    ])
    chunks_id = {}
    for k, snippet in enumerate(reranked_snippets):
        chunks_id[k+1] = snippet.id

    prompt = f"""You are a precise information retrieval agent. Your task is to answer the user's question using ONLY the information provided in the numbered snippets below. You must not use any prior knowledge or external information.

    Instructions:

    Read and analyze the provided snippets carefully.

    If the information needed to answer the question is present in the snippets, synthesize an answer based solely on that information.

    Crucially, after every sentence or distinct claim in your answer, you MUST cite the source by appending the relevant snippet reference in angle brackets in this tempalete <sup>[5]</sup>. For example: Answers text <sup>[8]</sup>.

    If the information to answer the question is not found in any snippet, state clearly: "I cannot answer the question based on the provided snippets."

    Do not add any interpretations, conclusions, or information not explicitly supported by the snippets.

    Snippets:
    {context}
"""
    df.loc[0, "Вопрос"] = query
    df.loc[0, 'Контекст'] = context
    df.loc[0, 'Промпт'] = prompt
    df.loc[0, 'retrieved_chunks_id'] = str(chunks_id)
    df.loc[0, 'Кол-во извлеченных сниппетов'] = len(reranked_snippets[:15])


#таблица для отрисовки
    reranked_snippets_df = pd.DataFrame()
    for k, snippet in enumerate(rerank_snippets_active):
        # print(k)
        reranked_snippets_df.loc[k, 'Позиция чанка'] = k+1
        reranked_snippets_df.loc[k, 'id'] = snippet.id
        # reranked_snippets_df.loc[k, 'rerank_score'] = snippet.payload['rerank_score']
        reranked_snippets_df.loc[k, 'page_content'] = snippet.payload['page_content']
        reranked_snippets_df.loc[k, 'file_name'] = snippet.payload['metadata']['file_name']
        reranked_snippets_df.loc[k, 'page'] = snippet.payload['metadata']['page']
        reranked_snippets_df.loc[k, 'prime_category'] = snippet.payload['metadata']['category']
        reranked_snippets_df.loc[k, 'region'] = snippet.payload['metadata']['doc_region']
        reranked_snippets_df.loc[k, 'countries'] = str(snippet.payload['metadata']['doc_countries'])
        reranked_snippets_df.loc[k, 'keywords'] = str(snippet.payload['metadata']['doc_keywords'])
        reranked_snippets_df.loc[k, 'page'] = snippet.payload['metadata']['page']
        try:
            reranked_snippets_df.loc[k, 'download_link'] = y.get_meta(f"/Reports 2026 YTD (sep)/{snippet.payload['metadata']['file_name']}").file 
        except Exception as e:
            st.toast(f"Ошибка при поиске в хранилище | {snippet.payload['metadata']['file_name']}")
            reranked_snippets_df.loc[k, 'download_link'] = "#"

    return df, reranked_snippets_df

