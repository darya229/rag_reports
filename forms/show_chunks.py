import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

def show_chunks():
    # Используем контейнер с настройками ширины
    with st.container():

        st.markdown(
            """
            <style>
            .st-d5  {
                width: 900px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

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
            st.markdown(f'Скачать документ: <a href="{st.session_state.current_retrieved_chunks.loc[selected_idx, "download_link"]}">здесь</a>', unsafe_allow_html=True)
            st.write("**документ:**")
            st.write(st.session_state.current_retrieved_chunks.loc[selected_idx, "file_name"])
            st.write(f"{int(st.session_state.current_retrieved_chunks.loc[selected_idx, "page"])} стр")
            st.write(f"**содержание:**")
            st.write(st.session_state.current_retrieved_chunks.loc[selected_idx, "page_content"])
    else:
        st.write("Выберите запрос, чтобы увидеть найденные фрагменты")