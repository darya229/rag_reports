import streamlit as st
import sys
import os
from streamlit_js_eval import streamlit_js_eval
import json
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.helpers import add_user_type_lable, check_admin, server_connect, completed_session_button, admin_panel, setup_logo, hide_running_man_animation

setup_logo()
hide_running_man_animation()
def window_height():
    if 'screen_height' not in st.session_state:
        st.session_state["screen_height"] = streamlit_js_eval(js_expressions='screen.height', key = 'SCR')
        print(st.session_state["screen_height"])
        return st.session_state["screen_height"]
    
@st.cache_resource
def load_session_data(role):
    params = {
        "mode":"get_sessions_data",

    }
    try:
        response = requests.get(url="http://103.54.16.74:3000/data", params=params)
        response_json = json.loads(response.content)
        result_dict = response_json["data"]
        # print(f"type: {type(result_dict)} | {result_dict}")
        return result_dict
    except:
        st.error("500")

if check_admin(st.session_state) == True:
    st.set_page_config(layout='wide')
    add_user_type_lable(st.session_state)
    st.title('Результаты')
    sessions_data = load_session_data(role=st.session_state["role"])
    st.session_state[f"sessions_data_{st.session_state["role"]}"] = sessions_data
    admin_panel(state="results")

    st.header('Заверешенные запросы')
    
    params = {
        "mode":"comlpeted_sessions_list",

    }

    response = requests.get(url="http://103.54.16.74:3000/data", params=params)
    response_json = json.loads(response.content)
    result = response_json["data"]


    with st.container(height=450):
        for session in result:
            completed_session_button(session)

        if 'show_session_data' not in st.session_state:
            st.session_state.show_session_data = None  

        if st.session_state.get('show_session_data') is not None:
            st.switch_page('pages/show_results.py')






else:
    st.switch_page('pages/login.py')