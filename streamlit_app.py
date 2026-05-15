import streamlit as st

import os
import sys

front_page = st.Page(
    page = 'pages/front_page.py',
    title = 'RAG Search',
    default=True
)



login = st.Page(
    page='pages/login.py',
    title='Login',
    default=False
)

admin_page = st.Page(
    page = 'pages/admin_profile_main.py',
    title = 'Главная',
    default = False
)

results = st.Page(
    page='pages/results.py',
    title='Результаты',
    default=False
)
show_results = st.Page(
    page='pages/show_results.py',
    title='Результаты',
    default=False
)

pg = st.navigation(pages=[front_page, login, admin_page, results, show_results], position='hidden')
pg.run()