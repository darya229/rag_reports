import streamlit as st 
import paramiko
import os
from functools import partial
from streamlit_cookies_manager import EncryptedCookieManager

def logout():
    print('def logout()')
    st.session_state.logout_button = True


    # cookies['logged_in_user'] = ''
    # cookies.save()

def check_auth(session_state):
    if st.session_state.get("authenticated", False):
        return True
    else:
        st.switch_page('pages/login.py')

def check_admin(session_state):
    print('check admin')
    # print(st.session_state)
    if session_state.get("authenticated", False):
        if session_state["role"] == 'admin':
            if "show_session_data" in st.session_state:
                print(f'check admin true: {st.session_state.show_session_data}')
            return True
        else:
            print(f'check admin false: {st.session_state.show_session_data}')
            return False
        
    else:
        st.switch_page('pages/login.py')

def add_user_type_lable(session_state):

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            role_text = session_state.get('role', '')
            if role_text == 'admin':
                st.markdown(
                    f"""
                    <div style="
                        background-color: #1bb523;
                        color: white;
                        padding: 4px 16px;
                        border-radius: 15px;
                        display: inline-block;
                        font-size: 11px;
                        font-family: sans-serif;
                        margin: 1px 0;
                        margin-bottom: 1px
                    ">
                        user
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #12e316;
                        color: white;
                        padding: 4px 8px;
                        border-radius: 4px;
                        display: inline-block;
                        font-size: 11px;
                        font-family: sans-serif;
                        margin: 1px 0;
                        margin-bottom: 1px
                    ">
                        {role_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        with col2:
            if st.button('Logout', on_click=partial(logout)):
                st.switch_page('pages/login.py')




# def verification(username, password, cookies):
#     ssh = paramiko.SSHClient()
#     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     ssh.connect('103.54.16.74', username='rdp_squirrel', password='P0stRCjBVSwN')
    
#     command = f"""cd /home/rdp_squirrel/Documents/SSS/AI-Scenario/ && source /home/rdp_squirrel/Documents/SSS/AI-Scenario/.venv312/bin/activate && python -c \"import sys; sys.path.append('/home/rdp_squirrel/Documents/SSS/AI-Scenario'); from db_functions import user_authentication; print(user_authentication(user_name='{username}', passwd='{password}'))\""""
#     stdin, stdout, stderr = ssh.exec_command(command)
#     # Читаем результат
#     result = stdout.read().decode().strip()
#     error = stderr.read().decode().strip()
#     if error:
#         st.error(error)

#     else:
#         st.write(result)
#         if eval(result)==True and username == 'admin':
#             st.session_state.authenticated = True
#             st.session_state.role = 'admin'
#             st.session_state.user = username
#             cookies['logged_in_user'] = 'admin'
#             cookies.save()
#             st.rerun()
#         elif eval(result)==True:
#             st.session_state.authenticated = True
#             st.session_state.role = 'user'
#             st.session_state.user = username
#             st.rerun()
#         else:
#             st.session_state.authenticated = False
#             st.error("Invalid Username/Password")
#     ssh.close()
#     return cookies

def server_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='103.54.16.74', 
                username='rdp_squirrel', 
                password='P0stRCjBVSwN', 
                # timeout=30, 
                # banner_timeout=30
                )
    return ssh

def current_session_to_show(button_label):
    st.session_state['show_session_data'] = button_label
    # st.switch_page("pages/show_results.py")

def completed_session_button(session_id):
    return st.button(label=session_id, width="stretch", on_click=partial(current_session_to_show, session_id))

def reset_show_session_state():
    if 'show_session_data' in st.session_state:
        st.session_state.show_session_data = None


def admin_panel(state: str):
    if state == 'request':
        with st.container():
            col1, col2 = st.columns(2, vertical_alignment='center')
            with col1:
                st.button('Запрос', use_container_width='stretch', disabled=True)
                
            with col2:
                if st.button('Результаты', use_container_width='stretch', on_click=partial(reset_show_session_state, )):
                    st.switch_page('pages/results.py')

    if state == "results":
        with st.container():
            col1, col2 = st.columns(2, vertical_alignment='center')
            with col1:
                if st.button('Запрос', use_container_width='stretch'):
                    st.switch_page('pages/admin_profile_main.py')
                
            with col2:
                st.button('Результаты', use_container_width='stretch', disabled=True)

        
    if state == "all_buttons_available":
        with st.container():
            col1, col2 = st.columns(2, vertical_alignment='center')
            with col1:
                if st.button('Запрос', use_container_width='stretch'):
                    st.switch_page('pages/admin_profile_main.py') 
                
            with col2:
                if st.button('Результаты', use_container_width='stretch', on_click=partial(reset_show_session_state, )):
                    st.switch_page('pages/results.py')


def user_panel(state: str):
    if state == 'request':
        pass

    if state == "results":
        pass


def setup_logo():
    LOGO_URL_LARGE = "pages/logo_edited.png"
    st.logo(
        LOGO_URL_LARGE,
        size="large"
    )

def hide_running_man_animation():
    hide_streamlit_style = """
                    <style>

                    div[data-testid="stStatusWidget"] {
                    visibility: hidden;
                    height: 0%;
                    position: fixed;
                    }

                    </style>
                    """
    return st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# hide_streamlit_style = """
#                 <style>
#                 div[data-testid="stToolbar"] {
#                 visibility: hidden;
#                 height: 0%;
#                 position: fixed;
#                 }
#                 div[data-testid="stDecoration"] {
#                 visibility: hidden;
#                 height: 0%;
#                 position: fixed;
#                 }
#                 div[data-testid="stStatusWidget"] {
#                 visibility: hidden;
#                 height: 0%;
#                 position: fixed;
#                 }
#                 #MainMenu {
#                 visibility: hidden;
#                 height: 0%;
#                 }
#                 header {
#                 visibility: hidden;
#                 height: 0%;
#                 }
#                 footer {
#                 visibility: hidden;
#                 height: 0%;
#                 }
#                 </style>
#                 """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)