import streamlit as st
import sys
import os

# while not cookie_manager.get_all():
#     print("загрузка куки")
#     time.sleep(2)
# st.toast('🚀 Приложение запущено!', icon='🎉')
st.set_page_config(layout='centered')
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)



# Комбинированный стиль для кнопки - RGB обводка + свечение
st.markdown("""
    <style>
    /* Основной стиль кнопки - черное тело с RGB обводкой */
    div.stButton > button {
        background-color: #000000 !important;
        color: white !important;
        border: 2px solid transparent !important;
        border-radius: 10px !important;
        padding: 12px 40px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        cursor: pointer !important;
        transition: all 0.5s ease !important;
        position: relative !important;
        overflow: hidden !important;
        z-index: 1 !important;
        letter-spacing: 1px !important;
    }
    
    /* RGB обводка (постоянная) */
    div.stButton > button::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        border: 5px solid transparent;
        border-radius: 12px !important;
        background: transparent !important;
        background-clip: padding-box !important;
        z-index: -1 !important;
        animation: rgb-border 3s linear infinite !important;
        filter: blur(2px) brightness(1.2) !important;
        opacity: 1 !important;
        
        /* Градиентная рамка */
        border-image: linear-gradient(
            45deg,
            #ff0000, #ff7300, #fffb00, #48ff00,
            #00ffd5, #002bff, #7a00ff, #ff00c8, #ff0000
        ) 1 !important;
        border-image-slice: 1 !important;
    }
    
    @keyframes rgb-border {
        0% { 
            background-position: 0% 50%; 
        }
        50% { 
            background-position: 100% 50%; 
        }
        100% { 
            background-position: 0% 50%; 
        }
    }
    
    /* Эффект при наведении - вся кнопка становится RGB */
    div.stButton > button:hover {
        color: white !important;
        background: linear-gradient(
            45deg,
            rgba(255, 0, 0, 0.9),
            rgba(255, 115, 0, 0.9),
            rgba(255, 251, 0, 0.9),
            rgba(72, 255, 0, 0.9),
            rgba(0, 255, 213, 0.9),
            rgba(0, 43, 255, 0.9),
            rgba(122, 0, 255, 0.9),
            rgba(255, 0, 200, 0.9)
        ) !important;
        background-size: 400% 400% !important;
        animation: rgb-hover 3s linear infinite !important;
        
        /* Усиленная обводка при наведении */
        box-shadow: 
            0 0 15px rgba(255, 0, 0, 0.6),
            0 0 25px rgba(255, 115, 0, 0.5),
            0 0 35px rgba(0, 255, 255, 0.4) !important;
    }
    
    @keyframes rgb-hover {
        0% { 
            background-position: 0% 50%; 
        }
        50% { 
            background-position: 100% 50%; 
        }
        100% { 
            background-position: 0% 50%; 
        }
    }
    
    /* Дополнительный RGB слой при наведении для большего эффекта */
    div.stButton > button:hover::before {
        filter: blur(8px) !important;
        opacity: 0.9 !important;
        animation: rgb-border 1.5s linear infinite !important;
    }
    
    /* Эффект при нажатии */
    div.stButton > button:active {
        transform: scale(0.98) !important;
        transition: transform 0.1s ease !important;
        
        /* Яркое свечение при нажатии */
        box-shadow: 
            0 0 20px rgba(255, 0, 0, 0.8),
            0 0 35px rgba(0, 255, 255, 0.7),
            0 0 50px rgba(255, 0, 200, 0.6) !important;
    }
    
    /* Анимация текста при наведении */
    div.stButton > button:hover span {
        display: inline-block;
        animation: text-glow 1.5s ease-in-out infinite alternate;
    }
    
    @keyframes text-glow {
        from {
            text-shadow: 0 0 5px #fff,
                         0 0 10px #fff,
                         0 0 15px #00ffea,
                         0 0 20px #00ffea;
        }
        to {
            text-shadow: 0 0 10px #fff,
                         0 0 20px #ff0080,
                         0 0 30px #ff0080,
                         0 0 40px #ff0080;
        }
    }
    
    /* Убираем стандартные стили фокуса */
    div.stButton > button:focus {
        outline: none !important;
        box-shadow: 
            0 0 20px rgba(255, 0, 0, 0.7),
            0 0 30px rgba(0, 255, 255, 0.6),
            0 0 40px rgba(255, 0, 200, 0.5) !important;
    }
    
    /* Эффект для неактивной кнопки */
    div.stButton > button:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }
    
    /* Контейнер для кнопки */
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px 0;
    }
    </style>
""", unsafe_allow_html=True)

if "logout_button" not in st.session_state:
    st.session_state.logout_button = False

st.session_state["logout_button"] = False
    

# if st.session_state["logout_button"] and cookie_manager.get("logged_in_user"):
    # st.session_state.logout_button = False
    # st.session_state.authenticated = False
    # st.session_state.role = ""
    # st.session_state.user = ""
    # cookie_manager.delete("logged_in_user")

print(f"FRONT PAGE STATE: {st.session_state}")
with st.container():
    col1, col2 = st.columns(2, vertical_alignment='center')
    with col1: 
        st.image("pages/search_logo.gif")
    with col2:
        st.markdown(

            '<span style="font-size: 62px; font-weight: semibold; color: #000000">RAG SEARCH</span>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<span style="font-size: 16px; color: #000000; font-weight: 160">AI Agent</span>',
            unsafe_allow_html=True
        )

with st.container():
    if st.button('Login', use_container_width='stretch'):
        st.switch_page('pages/login.py')



