import streamlit as st
from functools import partial

from helpers.helpers import add_user_type_lable, check_admin, server_connect, admin_panel, setup_logo, hide_running_man_animation

setup_logo()
hide_running_man_animation()




def set_up_report_type(button_label):
    st.session_state["report_type"] = button_label


if check_admin(st.session_state) == True:
    st.set_page_config(layout='wide')
    add_user_type_lable(st.session_state)
    if st.session_state['show_session_data'] is None:
        print("I AM GOING TO SWITCH PAGE")
        st.switch_page('pages/results.py')
    session_name = st.session_state['show_session_data']
    st.title(session_name)
    result_dict_show = st.session_state[f"sessions_data_{st.session_state["role"]}"][session_name]
    admin_panel(state="all_buttons_available")
    if result_dict_show:
        st.header('Ответ агента')
        with st.container(height=450):
            st.markdown(result_dict_show)
    else:

        st.write(f"❌ Ошибка 500")




        
    # st.write(result)
    # st.session_state['show_session_data'] = "None"
else:
    st.switch_page('pages/login.py')