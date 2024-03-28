import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📑",
)
st.title("DocumentGPT")

# streamlit의 session state는 여러 번 재실행을 해도 data가 보존될 수 있게 해준다.
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    # 만약 st.session_state가 messages라는 key를 가지고 있지 않다면 빈 list로 initialize 한다.
    # 가지고 있다면 그 messages를 유지하고 싶기 때문에 아무것도 하지 않는다.


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)


message = st.chat_input("Send a message to the Ai")

if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)
