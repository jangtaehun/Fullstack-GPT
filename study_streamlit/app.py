import streamlit as st

# from datetime import datetime
# today = datetime.today().strftime("%H:%M:%S")
# st.title(today)


# model = st.selectbox("Choose your model", ("GPT-3", "GPT-4"))
# if model == "GPT-3":
#     st.write("cheap")
# else:
#     st.write("expensive")


# value = st.slider(
#     "temperature",
#     min_value=0.1,
#     max_value=1.0,
# )
# st.write(value)


# st.title("title")
# # st.sidebar.title("sidebar title")
# # st.sidebar.text_input("xxx")
# with st.sidebar:
#     st.title("sidebar title")
#     st.text_input("xxx")


# tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])
# with tab_one:
#     st.write("a")
# with tab_two:
#     st.write("b")
# with tab_three:
#     st.write("c")


st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🐈",
)

st.title("FullstackGPT Home")

with st.chat_message("human"):
    st.write("Hello!")

with st.chat_message("Ai"):
    st.write("Hi")

with st.status("Embedding file...", expanded=True) as status:
    time.sleep(2)
    st.write("Getting the file")
    time.sleep(2)
    st.write("Embedding the file")
    time.sleep(2)
    st.write("Caching the file")
    status.update(label="Error", state="error")

st.chat_input("Send a message to the Ai")
