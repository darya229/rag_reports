import re
import ast
import streamlit as st
import os
from langchain_core.tools import tool
import pandas as pd
from qdrant_client.models import models
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from fastembed import SparseTextEmbedding 
import yadisk
from qdrant_client.http.models import Filter, FieldCondition, Range, DatetimeRange, MatchText
from datetime import datetime
API_QDRANT=os.getenv("API_QDRANT")
API_DISK=os.getenv("MY_YA_DISK")
API_QDRANT =  os.getenv("API_QDRANT")

y = yadisk.YaDisk(token=API_DISK)
#--------INITIALIZE CONNECTIONS ONCE -----------
@st.cache_resource(ttl=3600)
def initialize_connections():
    """Инициализация всех подключений один раз при запуске - кэшируется"""
    try:
        client_db = QdrantClient(
            url = "http://176.109.105.181:6333/",
            api_key=API_QDRANT,
            timeout=300,
            check_compatibility=False
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
if "work_on_query" not in st.session_state:
    st.session_state.work_on_query = 0

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


def search_separately(query_text, 
                      collection_name, 
                      dense_embedding_model, 
                      bm25_embedding_model, 
                      client,
                      quartel_filter:str | None=None,
                      month_filter: str | None=None,
                      year_filter: str | None=None,
                      category_filter: str | None=None,
                      company_filter: str | None=None,
                      filename_filter:str | None=None):
                      

    # print(f"Включилась функция search_separately")

    #Парсим фильтры
    must = []
    should = []
    # Определяем начальную и конечную даты квартала

    available_categories = ["Transport and Mechanical Engineering", "Retail Trade", "Pharmaceuticals", "Cryptocurrency Market", "TMT",
                            "Electric Power Industry", "Oil and Gas", "Currency Market", "Real Estate", "Food Industry", "Chemical Industry",
                            "Metals", "Infrastructure and Utilities", "Economy", "Geopolitics", "Other", "Agriculture", "Coal", "Finance",
                            "Fintech", "Automotive industry"]
    if quartel_filter:
        pattern_quartel = r'^([1-4])Q(\d{4})$'
        match = re.match(pattern_quartel, quartel_filter)

        quarter = int(match.group(1))
        year = int(match.group(2))

        quarter_months = {
            1: ["01", "02", "03"],
            2: ["04", "05", "06"],
            3: ["07", "08", "09"],
            4: ["10", "11", "12"]
        }

        months = quarter_months[quarter]

        month_conditions = [
            FieldCondition(
                key="metadata.month",
                match={"value": month}
            )
            for month in months
        ]

        must.append(
            Filter(
                must=[
                    FieldCondition(
                        key="metadata.year",
                        match={"value": str(year)}
                    )
                ],
                should=month_conditions
            )
        )
    
    if month_filter:
        pattern_month = r'^([1-9]|1[0-2])-(\d{4})$'
        match_month = re.match(pattern_month, month_filter)
        month = int(match_month.group(1))

        year_month_filter = int(match_month.group(2))

  
        if month not in [10, 11, 12]:
            month = "0" + str(month)

        must.append(
            Filter(
                must=[
                    FieldCondition(
                        key="metadata.year",
                        match={"value": str(year_month_filter)}
                    ),
                    FieldCondition(
                        key="metadata.month",
                        match={"value": str(month)}
                    ),
                ]
            )
        )

    if year_filter:
        pattern_year = r'^\d{4}$'
        year_match = re.match(pattern_year, str(year_filter))
        
        #проверяем, что год в диапазоне 2024-2026
        year_year_filter = int(year_match.group(0))

        must.append(
            Filter(
                must=[
                    FieldCondition(
                        key="metadata.year",
                        match={"value": str(year_year_filter)}
                    ),
                ]
            )
        )

    if category_filter:
      
        should.append(FieldCondition(
            key="metadata.category",
            match={"value": category_filter}  # точное совпадение
        ))

    if company_filter:
        should.append(FieldCondition(
            key="metadata.doc_company",
            match=MatchText(text=company_filter)  # поиск по не точному совпадению
        ))

    if filename_filter:
        should.append(FieldCondition(
            key="metadata.file_name",
            match={"value": filename_filter}  # точное совпадение
        ))

    date_filter = Filter(
        must=must,
        should=should
    )
 
    dense_vector = dense_embedding_model.encode(query_text)
    sparse_vector = next(bm25_embedding_model.query_embed(query_text))
    
    # Поиск по dense векторам
    dense_results = client.query_points(
        collection_name=collection_name,
        query=dense_vector.tolist() if hasattr(dense_vector, 'tolist') else dense_vector,
        using="qwen",  # указываем имя вектора отдельно
        with_payload=True,
        with_vectors=False,
        limit=50,
        query_filter=date_filter
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
        limit=50,
        query_filter=date_filter
    )

    #Добавляем тег типа поиска
    for point in sparse_results.points:
        point.payload['search_type'] = 'Ключевые слова'

    # print(f"len chunks search_separately\ndense_results.points: {len(dense_results.points)}\nsparse_results.points:{len(sparse_results.points)}")
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

    # print(f"rerank_snippets len(chunks): {len(chunks)}")
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


def hybrid_rerank_search(query_text, 
                         dense_embedding_model, 
                         bm25_embedding_model, 
                         client,  
                         cross_encoder_model, 
                         collection_name,
                         quartel_filter:str | None=None,
                         month_filter: str | None=None,
                         year_filter: str | None=None,
                         category_filter: str | None=None,
                         company_filter: str | None=None,
                         filename_filter:str | None=None):
    # print(f"Функция hybrid_rerank_search")

    dense_results, sparse_results = search_separately(query_text=query_text, 
                                                      dense_embedding_model=dense_embedding_model, 
                                                      bm25_embedding_model=bm25_embedding_model, 
                                                      client=client, 
                                                      collection_name=collection_name,
                                                      quartel_filter=quartel_filter,
                                                      month_filter=month_filter,
                                                      year_filter=year_filter,
                                                      category_filter=category_filter,
                                                      company_filter=company_filter,
                                                      filename_filter=filename_filter)
    if (len(dense_results) + len(sparse_results))==0:
        return "Nothing found"
    # print(f"len(dense_results) = {len(dense_results)}\nlen(sparse_results) = {len(sparse_results)}")

    reranked_snippets = rerank_snippets(query_text, dense_results, sparse_results, cross_encoder_model)

    return reranked_snippets


reranked_snippets_df = pd.DataFrame()
current_reranked_idx = 0
def retrieve_chunks(query: str,
                    quartel_filter:str | None=None,
                    month_filter: str | None=None,
                    year_filter: str | None=None,
                    category_filter: str | None=None,company_filter: str | None=None,
                    filename_filter:str | None=None):
    global reranked_snippets_df, current_reranked_idx
    # print("Включилась функция retrieve_chunks")
    all_snippets = []
    df = pd.DataFrame()
    reranked_snippets = hybrid_rerank_search(query_text=query,
                                            dense_embedding_model=dense_embedding_model, 
                                            bm25_embedding_model=bm25_embedding_model, 
                                            cross_encoder_model=cross_encoder_model, 
                                            client=client,
                                            collection_name='reports_database_v2',
                                            quartel_filter=quartel_filter,
                                            month_filter=month_filter,
                                            year_filter=year_filter,
                                            category_filter=category_filter,
                                            filename_filter=filename_filter)
    # print(f"len reranked_snippets in retrieve_chunks: {len(reranked_snippets)} | {type(reranked_snippets)}")
    if reranked_snippets == "Nothing found":
        return "Nothing found"
    else:
    
        ##### подставляем таблицы ########

        files = os.listdir("C:/Users/Chill Out/Documents/SSS/ТестированиеRAG/documents_elements_paddle_tables") #для таблиц
        for snippet in reranked_snippets:

            tables = extract_tables(snippet.payload["page_content"])
            if tables:
                doc_filename=snippet.payload["metadata"]["file_name"].replace(".pdf", ".feather")
                if doc_filename in files:
                    doc = pd.read_feather(f"C:/Users/Chill Out/Documents/SSS/ТестированиеRAG/documents_elements_paddle_tables/{doc_filename}")
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
        rerank_snippets_active = reranked_snippets[:10]
        all_snippets.append(reranked_snippets)
        context = "\n".join([
            f"<snippet {idx + current_reranked_idx  + 1}> {item.payload['page_content']} \n doc_title: {item.payload["metadata"]["file_name"]} \n page: {item.payload["metadata"]["page"]}</snippet {idx + current_reranked_idx + 1}>"
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
        # print(prompt)
        df.loc[0, "Вопрос"] = query
        df.loc[0, 'Контекст'] = context
        df.loc[0, 'Промпт'] = prompt
        df.loc[0, 'retrieved_chunks_id'] = str(chunks_id)
        df.loc[0, 'Кол-во извлеченных сниппетов'] = len(reranked_snippets[:15])

    #таблица для отрисовки
        
        for k, snippet in enumerate(rerank_snippets_active):
            # print(k)
            reranked_snippets_df.loc[current_reranked_idx, 'Позиция чанка'] = current_reranked_idx +1
            reranked_snippets_df.loc[current_reranked_idx , 'id'] = snippet.id
            # reranked_snippets_df.loc[k, 'rerank_score'] = snippet.payload['rerank_score']
            reranked_snippets_df.loc[current_reranked_idx , 'page_content'] = snippet.payload['page_content']
            reranked_snippets_df.loc[current_reranked_idx , 'file_name'] = snippet.payload['metadata']['file_name']
            reranked_snippets_df.loc[current_reranked_idx , 'page'] = snippet.payload['metadata']['page']
            reranked_snippets_df.loc[current_reranked_idx , 'prime_category'] = snippet.payload['metadata']['category']
            reranked_snippets_df.loc[current_reranked_idx , 'region'] = snippet.payload['metadata']['doc_region']
            reranked_snippets_df.loc[current_reranked_idx , 'countries'] = str(snippet.payload['metadata']['doc_countries'])
            reranked_snippets_df.loc[current_reranked_idx , 'keywords'] = str(snippet.payload['metadata']['doc_keywords'])
            reranked_snippets_df.loc[current_reranked_idx , 'page'] = snippet.payload['metadata']['page']
            # reranked_snippets_df.loc[current_reranked_idx , 'datetime'] = datetime(snippet.payload['metadata']['datetime']).strftime("%Y-%m")
            try:
                reranked_snippets_df.loc[current_reranked_idx, 'download_link'] = y.get_meta(f"/Reports 2026 YTD (sep)/{snippet.payload['metadata']['file_name']}").file 
            except Exception as e:
                print(f"file: {snippet.payload['metadata']['file_name']} | \n\nerror: {e}")
                reranked_snippets_df.loc[current_reranked_idx , 'download_link'] = "#"
            current_reranked_idx +=1

        return df

def reset_data():
    global reranked_snippets_df, current_reranked_idx
    # print("Сброс переменных")
    reranked_snippets_df = pd.DataFrame()
    current_reranked_idx = 0
    return reranked_snippets_df, current_reranked_idx



