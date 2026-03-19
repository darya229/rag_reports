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
import yadisk
import re
from dotenv import load_dotenv
load_dotenv()
from forms.show_chunks import show_chunks
API_DEEPSEEK=os.getenv("API_DEEPSEEK")
API_QDRANT=os.getenv("API_QDRANT")
API_DISK = os.getenv("API_DISK")

from RAG.retrieve import *

y = yadisk.YaDisk(token=API_DISK)


deepseek_llm = ChatDeepSeek(
    model="deepseek-reasoner",
    api_key = API_DEEPSEEK,
    temperature=1,
    max_tokens=8000,
    reasoning_effort="medium",
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
            'page': int(row['page'])
        }
        
    
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
    
    return processed_text

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

st.title('📚 База знаний')
st.subheader("Ответ LLM")

# Выводим историю сообщений
for i, message in enumerate(st.session_state.messages):
    # Определяем аватар в зависимости от роли :material/android:
    if message["role"] == "assistant" and message.get("is_system", False):
        avatar = ":material/priority_high:"  # Для системных сообщений ассистента
    elif message["role"] == "assistant":
        avatar = ":material/android:"  # Для обычных сообщений ассистента
    elif message["role"] == "user":
        avatar = ":material/person_pin:"
    else:
        avatar = ":material/android:"
    
    with st.chat_message(message["role"], avatar=avatar):
        if 'text' in message:
            st.markdown(message['text'], unsafe_allow_html=True)
        
        # Добавляем кнопку для каждого сообщения ассистента с ответом
        if message['role'] == 'assistant' and message.get('has_answer', False):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                button_key = f"show_chunks_btn_{message.get('query_id', i)}"
                if st.button('📑 Найденные фрагменты', key=button_key, use_container_width=True):
                    # Устанавливаем флаги для показа диалога
                    st.session_state.show_dialog = True
                    st.session_state.dialog_query_id = message.get('query_id')
                    # Не используем rerun() здесь

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
        
        # Показываем диалог
        show_chunks_form()
        
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
        temp_message.write("⏳ Поиск информации в базе")
    
    # Здесь ваша функция retriev_chunks
    try:
        df, reranked_snippets_df = retriev_chunks(query=user_input_en)
    except:
        st.error("Ошибка при работе базы данных")
        raise

    # Сохраняем информацию о запросе
    st.session_state.query_info[f"query_{current_query_id}"] = reranked_snippets_df

    # Сообщение о найденных фрагментах
    with st.chat_message("assistant", avatar=":material/android:"):
        st.write(f"📑 Найдено {len(reranked_snippets_df)} фрагментов")
    
    # Сохраняем системное сообщение о найденных фрагментах
    st.session_state.messages.append({
        "role": "assistant", 
        "text": f"📑 Найдено {len(reranked_snippets_df)} фрагментов",
        "is_system": True,
        "query_id": current_query_id
    })
    
    # Сообщение о подготовке ответа
    with st.chat_message("assistant", avatar=":material/android:"):
        temp_message = st.empty()
        temp_message.write("⏳ Подготовка ответа...")
    
    # Сохраняем сообщение о подготовке ответа
    st.session_state.messages.append({
        "role": "assistant", 
        "text": "⏳ Подготовка ответа...",
        "is_system": True,
        "query_id": current_query_id
    })

    try:
        answer_raw = deepseek_llm.invoke(
            [
                SystemMessage(content=df.loc[0, 'Промпт']),
                HumanMessage(content=df.loc[0, 'Вопрос'])
            ]
        )
        with st.chat_message("assistant", avatar=":material/android:"):
            temp_message = st.empty()
            temp_message.write("⏳ Добавляю ссылки...")

        answer = process_text_with_refs(answer_raw.content, reranked_snippets_df)

        with st.chat_message("assistant", avatar=":material/android:"):
            temp_message = st.empty()
            temp_message.write("✅ Ответ готов")
        # Показываем ответ
        with st.chat_message("assistant", avatar=":material/android:"):
            st.markdown(answer, unsafe_allow_html=True)

            
            # Добавляем кнопку прямо под ответом
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button('📑 Найденные фрагменты', key=f"current_answer_btn_{current_query_id}", use_container_width=True):
                    # Устанавливаем флаги для показа диалога
                    st.session_state.show_dialog = True
                    st.session_state.dialog_query_id = current_query_id
                    # Не используем rerun() здесь

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