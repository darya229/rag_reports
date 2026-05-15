import streamlit as st
import paramiko
import sys
import os
import time
import json
from datetime import datetime, timedelta
import requests
from streamlit_cookies_manager import EncryptedCookieManager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import extra_streamlit_components as stx
from streamlit_extras.customize_running import center_running
from helpers.helpers import setup_logo, hide_running_man_animation, server_connect
# from helpers.helpers import verification

setup_logo()
hide_running_man_animation()

@st.fragment
def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()
cookie_manager.get_all()
print("Загрузка куки 2 сек")
time.sleep(2)
if st.session_state.get("logout_button") == True:
    print("logout button turn on")
    st.session_state.authenticated = False
    st.session_state.role = ""
    st.session_state.user = ""
    print("delete logged_in_user")
    try:
        cookie_manager.delete("logged_in_user")
    except:
        print('pass')
    print(f"cookie: {cookie_manager.get("logged_in_user")}")
    print(st.session_state)
    print("2 sec rest")
    time.sleep(2)
    st.switch_page('pages/front_page.py')


# Инициализация session_state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role = ""
    st.session_state.user = ""


def verification(username, password):
    print("def verification")
    params = {
    "user":username,
    "password":password
    }
    response_verify = requests.get(url="http://103.54.16.74:3000/auth", params=params) 
    print(response_verify)
    resposne_verify_json = json.loads(response_verify.content)
    # print(f"verification: {cookie_manager.get_all()}")
    # Текущая дата и время
    current_date = datetime.now()
    # Дата через 7 дней
    date_in_7_days = current_date + timedelta(days=7)

    # Читаем результат

    if resposne_verify_json['auth_status'] == 'error: list index out of range':
        st.error("Пользователь не найден")
    else:
        # st.write(result)
        if resposne_verify_json['auth_status'] == True and username == 'admin':
            st.session_state.authenticated = True
            st.session_state.role = 'admin'
            st.session_state.user = username

            cookie_manager.set(cookie="logged_in_user", val="admin", expires_at=date_in_7_days, key="1")
            print("set cookie, 2 sec rest")
            time.sleep(2)

            st.rerun()
        elif resposne_verify_json['auth_status'] == True:
            st.session_state.authenticated = True
            st.session_state.role = 'admin'
            st.session_state.user = username
            cookie_manager.set(cookie="logged_in_user", val="user", expires_at=date_in_7_days, key="1")
            st.rerun()
        else:
            st.session_state.authenticated = False
            st.error("Invalid Username/Password")

def authenticate_user():
    print("authenticate_user")
    print(f"cookie_manager.get(logged_in_user) == {cookie_manager.get("logged_in_user")}")
    # Если уже аутентифицирован
    if (cookie_manager.get("logged_in_user")=="admin") or (st.session_state.get("authenticated", False)):
        return True
    # if st.session_state.get("authenticated", False):
    #     return True
    with st.container(vertical_alignment="center"):
    # Форма входа
        with st.form("login_form"):
            username = st.text_input(label="Username:")
            password = st.text_input(label="Password:", type="password")
            
            # Кнопка в форме
            if st.form_submit_button("Login", width="stretch"):
                verification(username, password)
                

    
    return False

# Основной код

if authenticate_user():
    print(" check authenticate_user")
    if (cookie_manager.get("logged_in_user") == "admin") or (st.session_state.get("role") == 'admin'):
        print(f"cookie_manager.get(logged_in_user) == {cookie_manager.get("logged_in_user")}")
        print(f"st.session_state.get(role) == {st.session_state.get("role")}")
        st.session_state.authenticated = True
        st.session_state.role = 'admin'
        st.session_state.user = 'admin'
        # print(st.session_state)
        st.switch_page('pages/admin_profile_main.py')

    elif (st.session_state["role"] == 'user'):
        st.session_state.authenticated = True
        st.session_state.role = 'user'
        st.session_state.user = 'user'
        st.switch_page('pages/user_profile_main.py') 


