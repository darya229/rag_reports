import streamlit as st
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from fastembed import SparseTextEmbedding 
import asyncio
import nest_asyncio
from googletrans import Translator
from dotenv import load_dotenv
load_dotenv()
API_DEEPSEEK=os.getenv("API_DEEPSEEK")
API_QDRANT=os.getenv("API_QDRANT")

from retrieve import *

deepseek_llm = ChatDeepSeek(
    model="deepseek-reasoner",
    api_key = API_DEEPSEEK,
    temperature=1,
    max_tokens=8000,
    reasoning_effort="medium",
)

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

#Переведод на англ
async def translate_text(text):
     async with Translator() as translator:
         result = await translator.translate(text)
         return result.text


st.set_page_config(layout='wide')

# Добавляем CSS для фиксации chat_input внизу
st.markdown("""
<style>
    .chat-input-container {
        position: fixed;
        bottom: 20px;
        width: 50%; /* примерная ширина правой колонки */
        background-color: white;
        padding: 10px;
        z-index: 100;
    }
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 45%;
    }
    /* Добавляем отступ снизу для контента, чтобы не перекрывался */
    .chat-messages-container {
        margin-bottom: 100px;
    }
            
    .tooltip-link {
        color: green;
        text-decoration: underline;
        cursor: pointer;
    }
    .st-emotion-cache-yfw52f a {
        color: rgb(29 255 69);
            
    /* Скролл для левой колонки - используем более надежные селекторы */
    div[data-testid="column"]:nth-of-type(1) {
        height: calc(100vh - 200px) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 10px !important;
    }

    /* Скролл для правой колонки */
    div[data-testid="column"]:nth-of-type(2) {
        height: calc(100vh - 200px) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
    }

    /* Чтобы контент внутри колонок тоже правильно отображался */
    div[data-testid="column"] > div {
        height: 100% !important;
        overflow-y: auto !important;
    }

    /* Фикс для контейнера сообщений в левой колонке */
    .chat-messages-container {
        margin-bottom: 100px;
        height: auto !important;
    }

    /* Убедимся, что chat_input остается внизу */
    .stChatInput {
        position: relative !important;
        bottom: 0 !important;
        width: 100% !important;
        margin-top: 20px !important;
    }

    /* Стилизация полосы прокрутки */
    div[data-testid="column"]::-webkit-scrollbar {
        width: 8px !important;
    }

    div[data-testid="column"]::-webkit-scrollbar-track {
        background: #f1f1f1 !important;
        border-radius: 10px !important;
    }

    div[data-testid="column"]::-webkit-scrollbar-thumb {
        background: #888 !important;
        border-radius: 10px !important;
    }

    div[data-testid="column"]::-webkit-scrollbar-thumb:hover {
        background: #555 !important;
    }

</style>
""", unsafe_allow_html=True)
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'query_info' not in st.session_state:
    st.session_state.query_info = {}


st.title('📚 База знаний')
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Ответ LLM")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "text": "Введите ваш вопрос"}]
    
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    
    if 'query_info' not in st.session_state:
        st.session_state.query_info = {}
    
    if 'current_retrieved_chunks' not in st.session_state:
        st.session_state.current_retrieved_chunks = pd.DataFrame()
    
    if 'current_query_text' not in st.session_state:
        st.session_state.current_query_text = ""
    
    # Контейнер для сообщений с отступом снизу
    messages_container = st.container()
    with messages_container:
        for i, message in enumerate(st.session_state.messages):
            # Определяем аватар в зависимости от роли
            if message["role"] == "assistant" and message.get("is_system", False):
                avatar = ":material/android:"  # Для системных сообщений ассистента
            elif message["role"] == "assistant":
                avatar = ":material/priority_high:"  # Для обычных сообщений ассистента
            elif message["role"] == "user":
                avatar = ":material/person_pin:"
            else:
                avatar = ":material/android:"
            
            with st.chat_message(message["role"], avatar=avatar):
                if 'text' in message:
                    st.markdown(message['text'], unsafe_allow_html=True)
                
                # Если у сообщения есть кнопка для показа фрагментов
                if message.get("has_button", False):
                    if st.button("Показать найденные фрагменты", key=f"query_btn_{message['query_id']}"):
                        st.session_state.current_retrieved_chunks = st.session_state.query_info[f"query_{message['query_id']}"]
                        # Сохраняем текст запроса для отображения
                        st.session_state.current_query_text = message.get("query_text", "")
                        st.rerun()
    
    # chat_input будет внизу
    user_input = st.chat_input('Ваш запрос')
    if user_input:
        # Добавляем сообщение пользователя в историю
        st.session_state.messages.append({"role": "user", "text": user_input})
        st.session_state.query_count += 1
        
        with st.chat_message("user", avatar=":material/person_pin:"):
            st.write(user_input)
        try:
            nest_asyncio.apply()
            user_input_en = asyncio.run(translate_text(user_input))
            temp_message = st.empty()
        except:
            st.error("Ошибка при переводе запроса")
            raise

        with st.chat_message("user", avatar=":material/person_pin:"):
            st.write(user_input_en)

        # Сообщение о поиске
        with st.chat_message("assistant", avatar=":material/android:"):
            temp_message = st.empty()
            temp_message.write("⏳ Поиск информации в базе")
        
        # Здесь ваша функция retriev_chunks
        try:
            df, reranked_snippets_df = retriev_chunks(query=user_input_en)
        except:
            st.error("Ошибка при работе базы данных")
            raise

        # Сохраняем информацию о запросе
        current_query_id = st.session_state.query_count
        st.session_state.query_info[f"query_{current_query_id}"] = reranked_snippets_df

        # Добавляем сообщение о найденных фрагментах с кнопкой
        with st.chat_message("assistant", avatar=":material/android:"):
            st.write(f"📑 Найдено {len(reranked_snippets_df)} фрагментов")
            
            # Кнопка для этого конкретного запроса
            if st.button("Показать найденные фрагменты", key=f"query_btn_{current_query_id}"):
                st.session_state.current_retrieved_chunks = reranked_snippets_df
                st.session_state.current_query_text = user_input  # Сохраняем текст запроса
                st.rerun()
        
        # Сохраняем это сообщение в историю с меткой о кнопке
        st.session_state.messages.append({
            "role": "assistant", 
            "text": f"📑 Найдено {len(reranked_snippets_df)} фрагментов",
            "has_button": True,
            "query_id": current_query_id,
            "query_text": user_input,  # Сохраняем текст запроса
            "is_system": True
        })
        
        # Сообщение о подготовке ответа
        with st.chat_message("assistant", avatar=":material/android:"):
            temp_message = st.empty()
            temp_message.write("⏳ Подготовка ответа...")
        
        # Сохраняем сообщение о подготовке ответа
        st.session_state.messages.append({
            "role": "assistant", 
            "text": "⏳ Подготовка ответа...",
            "is_system": True
        })

        try:
            answer = deepseek_llm.invoke(
                [
                    SystemMessage(content=df.loc[0, 'Промпт']),
                    HumanMessage(content=df.loc[0, 'Вопрос'])
                ]
            )
            with st.chat_message("assistant", avatar=":material/android:"):
                temp_message = st.empty()
                temp_message.write("✅ Ответ готов")
                st.markdown(answer.content, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant", 
                "text": answer.content,
                "has_button": False,
                "query_id": current_query_id,
                "query_text": user_input,  # Сохраняем текст запроса
                "is_system": True
            })
        except:
            st.error("Ошибка при работе LLM")
            raise


with col2:
    st.subheader("Просмотр источников")
    
    # Отображаем текст запроса, если есть выбранные фрагменты
    if "current_retrieved_chunks" in st.session_state and not st.session_state.current_retrieved_chunks.empty:
        if st.session_state.current_query_text:
            st.write(f"📝 **Найденные фрагменты для запроса:**")
            st.info(st.session_state.current_query_text)
            st.divider()  # Разделитель для визуального отделения
        st.dataframe(st.session_state.current_retrieved_chunks)
        num_rows = len(st.session_state.current_retrieved_chunks)
            
            # Создаем список вариантов от 1 до num_rows
        options = [str(i) for i in range(1, num_rows + 1)]
            
            # Создаем selectbox с динамическим количеством вариантов
        option = st.selectbox(
            "Выберите номер фрагмента",
            options=options,
            key="fragment_selector"
        )
            
        # Можно добавить отображение выбранного фрагмента
        if option:
            selected_idx = int(option) - 1  # Преобразуем в индекс (0-based)
            st.write(f"Выбран фрагмент №{option}")
            st.markdown(f'Скачать документ: <a href="https://downloader.disk.yandex.ru/disk/50f0f0e5a200da653ceaf3ce15df383ee72fab7ee685fa161dba8c4278e1b0ff/69b97cde/pcuE8BFM5tD6h2lj86pHazD42oYzeGUJCarCTKVRZxh1837V0t5DDlen2wZctd467cOjf5l6ltpZvlfUX0ZKRA%3D%3D?uid=297821915&filename=20-Jan-BOFA_morning_notes.pdf&disposition=attachment&hash=&limit=0&content_type=application%2Fpdf&owner_uid=297821915&fsize=295333&hid=849866c3d7b1d556a89d05d878b26c7b&media_type=document&tknv=v3&etag=06ca087e87274a2d65b284239c573bba">здесь</a>', unsafe_allow_html=True)
            st.write("**документ:**")
            st.write(st.session_state.current_retrieved_chunks.loc[selected_idx, "file_name"])
            st.write(f"{int(st.session_state.current_retrieved_chunks.loc[selected_idx, "page"])} стр")
            st.write(f"**содержание:**")
            st.write(st.session_state.current_retrieved_chunks.loc[selected_idx, "page_content"])
    else:
        st.write("Выберите запрос, чтобы увидеть найденные фрагменты")
