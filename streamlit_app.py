import streamlit as st
import os
import asyncio
import requests
import nest_asyncio
from googletrans import Translator
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
API_RAG = st.secrets["API_RAG"]
RAG_URL = st.secrets["RAG_URL"]



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
hide_streamlit_style = """
                <style>
                footer {
                visibility: hidden;
                height: 0%;
                }
                div[data-testid="stToolbar"] button {
                    display: none;
                }
                # div[data-testid="stToolbar"] {
                # visibility: hidden;
                # height: 0%;
                # position: fixed;
                # }
                # div[data-testid="stDecoration"] {
                # visibility: hidden;
                # height: 0%;
                # position: fixed;
                # }
                # #MainMenu {
                # visibility: hidden;
                # height: 0%;
                # }

                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


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
if 'reranked_snippets_df' not in st.session_state:
    print("add reranked_snippets_df in session state")
    st.session_state.reranked_snippets_df = pd.DataFrame()
if 'current_reranked_idx' not in st.session_state:
    print("add rcurrent_reranked_idx in session state")
    st.session_state.current_reranked_idx = 0


# st.subheader("❗️ база данных содержит 6275 документов за период 01.01.2026 — 01.03.2026")
st.subheader("RAG search agent (demo)")
with st.expander("Информация о базе данных"):
    st.write("Количесво документов в базе: 6275")
    st.write("Дата выпуска документов: 01.01.2026 — 01.03.2026")
    st.write("Тематика документов:")
    categories_list = """categories_list = [
    'Автопром',
    'Валютный рынок',
    'Геополитика',
    'Инфраструктура и ЖКХ',
    'Криптовалютный рынок',
    'Металлы',
    'Недвижимость',
    'Нефтегаз',
    'Пищевая промышленность',
    'Прочие',
    'Розничная торговля',
    'Сельское хозяйство',
    'ТМТ',
    'Транспорт и машиностроение',
    'Уголь',
    'Фармацевтика',
    'Финансы',
    'Финтех',
    'Химическая промышленность',
    'Экономика',
    'Электроэнергетика'    
] """
    st.code(categories_list, language="python")

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
        if 'steps_logic' in message:
            with st.expander("Нажми сюда, чтобы узнать мою логику построения ответа"):
                st.json(message['steps_logic'], expanded=3)
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
    st.write("User input")
    st.code(st.session_state)
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
        temp_message.write("🔨 Работаю над запросом. Это займет несколько минут. Пожалуйста, не закрывайте окно приложения.")    


    try:
        url = f"http://{RAG_URL}:3000/api/rag"
        token = API_RAG
        params = {
            "query": user_input
        }
        headers = {
            "Authorization": API_RAG
        }
        rag_agent_response = requests.request("GET", url, headers=headers, params=params)
        llm_response = rag_agent_response.json()

        with st.chat_message("assistant", avatar=":material/android:"):
            temp_message = st.empty()
            temp_message.write("✅Ответ готов")  
        with st.chat_message("assistant", avatar=":material/android:"):
            with st.expander("Нажми сюда, чтобы узнать мою логику построения ответа"):
                st.json(llm_response["processing_info"], expanded=3)
            
            st.markdown(llm_response["agent_answer"], unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant", 
            "text": llm_response["agent_answer"],
            "steps_logic": llm_response["processing_info"],
            "has_answer": True,  # Отмечаем, что это полноценный ответ с кнопкой
            "query_id": current_query_id,
            "is_system": False
        })
        
    except Exception as e:
        st.error(f"Ошибка при работе LLM: {e}")
        raise
