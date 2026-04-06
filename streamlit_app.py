import streamlit as st
import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from fastembed import SparseTextEmbedding 
import asyncio
import nest_asyncio
from googletrans import Translator
import yadisk
import re
from langfuse.langchain import CallbackHandler
from langchain.agents.middleware import ToolCallLimitMiddleware
from dotenv import load_dotenv
load_dotenv()
from forms.show_chunks import show_chunks
API_QDRANT =  os.getenv("API_QDRANT")
API_DEEPSEEK=os.getenv("API_DEEPSEEK")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")
API_DISK=os.getenv("MY_YA_DISK")

from RAG.retrieve import *

langfuse_handler = CallbackHandler()

deepseek_llm = ChatDeepSeek(
    model="deepseek-reasoner",
    api_key = API_DEEPSEEK,
    temperature=1,
    max_tokens=8000,
    reasoning_effort="medium",
)


deepseek_llm_assistant = ChatDeepSeek(
    model="deepseek-chat",
    api_key=API_DEEPSEEK,
    temperature=0.5,
)

@st.dialog('Найденные фрагменты')
def show_chunks_form():
   show_chunks()

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

#Переведод на англ
async def translate_text(text):
     async with Translator() as translator:
         result = await translator.translate(text)
         return result.text

st.set_page_config(layout='wide')


LOGO_URL_LARGE = "logo_edited.png"
st.logo(
    LOGO_URL_LARGE,
    size="large"
)

# Добавляем CSS для фиксации chat_input внизу
st.markdown("""
<style>
 
    .tooltip-link {
        color: green;
        text-decoration: underline;
        cursor: pointer;
    }
    .st-emotion-cache-yfw52f a {
        color: rgb(29 255 69);
            

</style>
""", unsafe_allow_html=True)


def process_text_with_refs(text, df):
    """
    Преобразует текст с ссылками вида [N] или [N,M] в HTML-ссылки
    
    Параметры:
    text (str): Исходный текст с ссылками
    df (pd.DataFrame): DataFrame с колонками 'Позиция чанка', 'file_name', 'page', 'href'
    
    Возвращает:
    str: Текст с замененными ссылками
    """
    
    # Создаем словарь для быстрого поиска данных по номеру ссылки
    ref_dict = {}
    for _, row in df.iterrows():
        ref_dict[str(int(row["Позиция чанка"]))] = {
            'href': row["download_link"],
            'file_name': row['file_name'],
            'page': int(row['page'])+1
        }
    sources = []
    check_added_sources = set()
        
    
    def replace_ref(match):
        """Заменяет одну ссылку или несколько ссылок через запятую"""
        refs_text = match.group(1)  # содержимое внутри [ ]
        
        # Разделяем ссылки, если их несколько через запятую
        ref_numbers = [num.strip() for num in refs_text.split(',')]
        
        # Создаем HTML для каждой ссылки
        html_parts = []
        for ref_num in ref_numbers:
            if ref_num in ref_dict:
                ref_data = ref_dict[ref_num]
                html = f'<a href="{ref_data["href"]}" class="tooltip-link" title="{ref_data["file_name"]} \n page: {ref_data["page"]} "><sup>[{ref_num}]</sup></a>'
                html_parts.append(html)
                check_added_sources.add(int(ref_num))

            else:
                # Если ссылка не найдена в DataFrame, оставляем как есть
                html_parts.append(f'<sup>[{ref_num}]</sup>')

        
        # Объединяем все части
        if len(html_parts) > 1:
            return ' '.join(html_parts)
        else:
            return html_parts[0]
    
    # Ищем все вхождения [число] или [число,число]
    pattern = r'\[([0-9,\s]+)\]'
    
    # Заменяем все найденные ссылки
    processed_text = re.sub(pattern, replace_ref, text)

    check_added_sources_list = list(check_added_sources)
    check_added_sources_list.sort()
    doc_to_refs = {}
    for item in check_added_sources_list:
        ref_data = ref_dict[str(item)]
        doc_key = ref_data["file_name"]
        if doc_key not in doc_to_refs:
            doc_to_refs[doc_key] = []
        doc_to_refs[doc_key].append(item)
    
    # Формируем строки источников с группировкой
    for doc_name, ref_nums in doc_to_refs.items():
        ref_nums.sort()
        refs_str = '], ['.join(str(num) for num in ref_nums)
        add_source = f"[{refs_str}] — {doc_name}\n\n"
        sources.append(add_source)
    
    return processed_text + f"\n\n______\n\n**Источники** \n\n{'\n'.join(sources)}"

# Инициализация session_state
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'query_info' not in st.session_state:
    st.session_state.query_info = {}
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "text": "Введите ваш вопрос"}]
if 'current_retrieved_chunks' not in st.session_state:
    st.session_state.current_retrieved_chunks = pd.DataFrame()
if 'current_query_text' not in st.session_state:
    st.session_state.current_query_text = ""
if 'show_dialog' not in st.session_state:
    st.session_state.show_dialog = False
if 'dialog_query_id' not in st.session_state:
    st.session_state.dialog_query_id = None
@tool
def rag(user_query: str,
        quartel_filter:str | None=None,
        month_filter: str | None=None,
        year_filter: str | None=None,
        category_filter: str | None=None,
        company_filter: str | None=None,
        filename_filter:str | None=None):
    """RAG. Searching for information in a vector database and preparing a response using LLM. RAG is best suited for searching one specific question for one entity.
        Perform semantic (dense) and keyword (sparse) search with filtering support.
        
        This function executes two types of search simultaneously:
        1. Semantic search using dense embeddings (Qwen model) - finds documents by meaning
        2. Keyword search using sparse embeddings (BM25) - finds documents by exact/partial word matching
        
        The filters use logical AND for date/time conditions (must) and logical OR for 
        category/company/filename conditions (should). This means:
        - Date filters (quarter, month, year) are combined with AND
        - Category, company, and filename filters are combined with OR
        - Date filters AND (category OR company OR filename)
        
        Parameters
        ----------
        user_query : str
            Search query text
        quartel_filter : str | None, optional
            Filter by quarter in format: 'XQYYYY' where X=1-4 (quarter number), YYYY=4-digit year
            Example: '1Q2026' (first quarter of 2026), '2Q2025' (second quarter of 2025)
        month_filter : str | None, optional
            Filter by month in format: 'M-YYYY' where M=1-12 (month number without leading zero), 
            YYYY=4-digit year
            Example: '1-2026' (January 2026), '12-2025' (December 2025)
            Note: Month should be 1-12, year must be between 2025-2026
        year_filter : str | None, optional
            Filter by year in format: 'YYYY' where YYYY=4-digit year
            Example: '2025', '2026'
            Note: Only years 2025-2026 are supported
        category_filter : str | None, optional
            Filter by category (exact match). Available categories:
            'Transport and Mechanical Engineering', 'Retail Trade', 'Pharmaceuticals', 
            'Cryptocurrency Market', 'TMT', 'Electric Power Industry', 'Oil and Gas', 
            'Currency Market', 'Real Estate', 'Food Industry', 'Chemical Industry',
            'Metals', 'Infrastructure and Utilities', 'Economy', 'Geopolitics', 'Other', 
            'Agriculture', 'Coal', 'Finance', 'Fintech', 'Automotive industry'
        company_filter : str | None, optional
            Filter by company name (partial/text match, not exact). Uses fuzzy matching.
            Example: 'нефть' will match 'Роснефть', 'Транснефть', etc.
        filename_filter : str | None, optional
            Filter by filename (exact match)
    
    
    """
    def check_year(year:int):
        if year <= 2023 or year >= 2027:
            return f"Значение года выходит за диапазон базы. Ожидается год в промежутке от 2025 до 2026 включительно Получено: {year}"
        

    #Проверяем фильтры
    available_categories = ["Transport and Mechanical Engineering", "Retail Trade", "Pharmaceuticals", "Cryptocurrency Market", "TMT",
                            "Electric Power Industry", "Oil and Gas", "Currency Market", "Real Estate", "Food Industry", "Chemical Industry",
                            "Metals", "Infrastructure and Utilities", "Economy", "Geopolitics", "Other", "Agriculture", "Coal", "Finance",
                            "Fintech", "Automotive industry"]
    if quartel_filter:
        pattern_quartel = r'^([1-4])Q(\d{4})$'
        match = re.match(pattern_quartel, quartel_filter)
        
        if not match:
            return f"Неверный формат квартала. Ожидается 'XQYYYY', где X=1-4, YYYY=год. Получено: {quartel_filter}"
        
        quarter = int(match.group(1))
        year = int(match.group(2))
        check_year(year)

    
    if month_filter:
        pattern_month = r'^([1-9]|1[0-2])-(\d{4})$'
        match_month = re.match(pattern_month, month_filter)
        if not match_month:
            return f"Неверный формат: '{month_filter}'. Ожидается формат 'M-YYYY', где M = 1-12, YYYY = 4-значный год (например, '1-2026')"
        month = int(match_month.group(1))

        year_month_filter = int(match_month.group(2))

        check_year(year_month_filter)
        if month not in [10, 11, 12]:
            month = "0" + str(month)

    if year_filter:
        pattern_year = r'^\d{4}$'
        year_match = re.match(pattern_year, str(year_filter))
        
        if not year_match:
            return f"Неверный формат года. Ожидается 'YYYY', YYYY=год. Получено: {year_filter}"
        #проверяем, что год в диапазоне 2024-2026
        year_year_filter = int(year_match.group(0))
        check_year(year_year_filter)

    if category_filter:
        if category_filter not in available_categories:
            return f"""Такой категории нет в базе. Ожидаются категории: ["Transport and Mechanical Engineering", "Retail Trade", "Pharmaceuticals", "Cryptocurrency Market", "TMT", "Electric Power Industry", "Oil and Gas", "Currency Market", "Real Estate", "Food Industry", "Chemical Industry", Metals", "Infrastructure and Utilities", "Economy", "Geopolitics", "Other", "Agriculture", "Coal", "Finance", "Fintech", "Automotive industry"]. Получено: {category_filter}"""

    retrieve_result = retrieve_chunks(query = user_query,
                                      quartel_filter=quartel_filter,
                                      month_filter=month_filter,
                                      year_filter=year_filter,
                                      category_filter=category_filter,
                                      company_filter=company_filter,
                                      filename_filter=filename_filter)
    if not isinstance(retrieve_result, pd.DataFrame):
        return "Nothing found"
    
    else:
        llm_response= deepseek_llm.invoke(
        [
            HumanMessage(content=retrieve_result.loc[0, "Промпт"])
        ])
        return llm_response.content

@tool
def current_date():
    """This function shows a current date in format &Y-&m-&d"""
    return datetime.now().strftime("%Y-%m-%d")

tool_limit_middleware = ToolCallLimitMiddleware(
    tool_name="rag",  # Укажите имя нужного инструмента
    run_limit=5,  # Не более 3 вызовов за один запуск
    exit_behavior="continue"  # Как себя вести при превышении лимита
)

st.subheader("❗️ база данных содержит 6275 документов за период 01.01.2026 — 01.03.2026")

# Выводим историю сообщений
for i, message in enumerate(st.session_state.messages):
    # Определяем аватар в зависимости от роли :material/android:
    if message["role"] == "assistant" and message.get("is_system", False):
        avatar = ":material/android:"  # Для системных сообщений ассистента
    elif message["role"] == "assistant":
        avatar = ":material/android:"  # Для обычных сообщений ассистента
    elif message["role"] == "user":
        avatar = ":material/person_pin:"
    else:
        avatar = ":material/android:"
    
    with st.chat_message(message["role"], avatar=avatar):
        if 'text' in message:
            st.markdown(message['text'], unsafe_allow_html=True)
        

# Обработка диалога
if st.session_state.show_dialog and st.session_state.dialog_query_id is not None:
    query_key = f"query_{st.session_state.dialog_query_id}"
    if query_key in st.session_state.query_info:
        st.session_state.current_retrieved_chunks = st.session_state.query_info[query_key]
        # Получаем текст запроса из истории сообщений
        for msg in st.session_state.messages:
            if msg.get('query_id') == st.session_state.dialog_query_id and msg.get('role') == 'user':
                st.session_state.current_query_text = msg.get('text', '')
                break

        # Сбрасываем флаги после показа диалога
        # Важно: это должно быть после вызова show_chunks_form()
        st.session_state.show_dialog = False
        st.session_state.dialog_query_id = None
        # Не используем rerun() здесь

# chat_input будет внизу
user_input = st.chat_input('Ваш запрос')
if user_input:
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append({"role": "user", "text": user_input})
    st.session_state.query_count += 1
    current_query_id = st.session_state.query_count
    
    with st.chat_message("user", avatar=":material/person_pin:"):
        st.write(user_input)
    
    try:
        nest_asyncio.apply()
        user_input_en = asyncio.run(translate_text(user_input))
    except:
        st.error("Ошибка при переводе запроса")
        raise
    with st.chat_message("user", avatar=":material/person_pin:"):
        st.write(user_input_en)
    # Сообщение о поиске
    with st.chat_message("assistant", avatar=":material/android:"):
        temp_message = st.empty()
        temp_message.write("🔨 Работаю над запросом...")    

    system_prompt = """
    You are a smart assistant who provides answers only by using the RAG system. The RAG system is designed to answer simple, specific questions about a single topic or fact.

    Key rules for using the RAG system:

    1. **Decompose complex queries** — When a user asks a question that involves multiple entities, comparisons, or distinct topics, break it down into separate, simple questions. Each question should be answerable independently by the RAG system.
    
    Example: User asks: "Who had higher revenue in 2025: NVIDIA or AMD?"
    Decompose into:
    - "Revenue 2025 NVIDIA"
    - "Revenue 2025 AMD"

    2. **Formulate queries for vector search** — RAG queries should be concise, factual, and structured as simple noun phrases or short questions that focus on the key entities and attributes. Avoid conversational language, pronouns, or complex syntax. Use a consistent format: [Attribute] + [Entity] + [Year/Context].
    
    Example: User asks: "What was NVIDIA's revenue in 2024?"
    RAG query: "Revenue NVIDIA 2024"
    
    Example: User asks: "What happened in Venezuela in January 2026? Tell me a gold forecast for 2026 year."
    Decompose into:
    - "What happened in Venezuela January 2026"
    - "Gold forecast 2026"

    3. **Preserve citations** — Each RAG response contains citations formatted as <sup>#</sup> (e.g., <sup>1</sup>, <sup>2</sup>) that reference source documents. When synthesizing information from multiple RAG queries into your final answer, you MUST preserve all citations exactly as they appear. Never remove, alter, or separate citations from the facts they support. Keep them in the same <sup>#</sup> format.
    
    Critical: If a piece of information comes with a citation, that citation must remain attached to that information in your final answer. Citations are essential for traceability and credibility.

    4. **One fact per query** — Each RAG query should seek a single piece of information. Do not combine multiple facts or comparisons into one query.

    5. **Synthesize with citations intact** — After retrieving results from all decomposed queries, combine the information into a cohesive answer. Ensure that every fact retains its original citation markers. If information is missing from any query, state what could not be found while preserving citations from the information you did receive.

    Remember: The RAG system performs best with clear, atomic, fact-oriented queries. Citations are non-negotiable — they must survive your summarization unchanged.
    If user's query depend on time, but you don't know a current date (for example he asks about last month events), you can use current_date tool, that shows a current date.
    """
    tools = [rag,current_date]
    try:
        agent = create_agent(
            model = deepseek_llm_assistant,
            tools = tools,
            system_prompt=system_prompt,
            middleware=[tool_limit_middleware]
            # state_schema=State
    )


        answer_raw = agent.invoke(input={
            "messages": [HumanMessage(user_input_en)]
            }, config={"callbacks": [langfuse_handler]})
        with st.chat_message("assistant", avatar=":material/android:"):
            temp_message = st.empty()
            temp_message.write("⏳ Добавляю ссылки...")

        answer = process_text_with_refs(answer_raw["messages"][-1].content, reranked_snippets_df)

        with st.chat_message("assistant", avatar=":material/android:"):
            temp_message = st.empty()
            temp_message.write("✅ Ответ готов")
        # Показываем ответ
        with st.chat_message("assistant", avatar=":material/android:"):
            st.markdown(answer, unsafe_allow_html=True)
        reset_data()


        # Сохраняем ответ в историю с меткой, что это ответ с кнопкой
        st.session_state.messages.append({
            "role": "assistant", 
            "text": answer,
            "has_answer": True,  # Отмечаем, что это полноценный ответ с кнопкой
            "query_id": current_query_id,
            "is_system": False
        })
        
    except Exception as e:
        st.error(f"Ошибка при работе LLM: {e}")
        raise