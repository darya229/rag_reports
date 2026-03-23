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
API_DEEPSEEK=os.getenv("API_DEEPSEEK")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")

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
            timeout=300
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
    for item in check_added_sources_list:
        ref_data = ref_dict[str(item)]
        add_source = f"[{str(item)}] — {ref_data["file_name"]} \n\n"
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
def rag(user_query: str):
    """RAG. Поиск информации в векторной базе данных и подготовка ответа с помощью LLM. RAG лучше всего ищет один конкретный вопрос для одной сущности."""
    print("Включилась функция RAG.")
    print(f"QUERY: {user_query} \n")
    st.toast(f"🔎 Ищу информацию по запросу: {user_query}")

    retrieve_result = retrieve_chunks(query = user_query)
    llm_response= deepseek_llm_assistant.invoke(
    [
        HumanMessage(content=retrieve_result[0].loc[0, "Промпт"])
    ])
    print(f"QUERY: {user_query} \n\nRESPONSE: {llm_response.content}\n\n")
    return llm_response.content

tool_limit_middleware = ToolCallLimitMiddleware(
    tool_name="rag",  # Укажите имя нужного инструмента
    run_limit=3,  # Не более 3 вызовов за один запуск
    exit_behavior="continue"  # Как себя вести при превышении лимита
)

st.title('📚 База знаний')
st.subheader("Ответ LLM")

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
    """
    tools = [rag]
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