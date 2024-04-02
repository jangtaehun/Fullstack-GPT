import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader


# 파일에 대한 질문을 하는 대신에 LLM이 우리에게 파일의 내용과 관련된 질문을 하게끔 명령할 것이다.
# Wikipedia Retriever를 사용
# OutputParser를 배우는 것에 포인트를 둬야 한다.

# Function Calling
# 우리가 만든 함수가 어떻게 생겼는지, 어떤 parameter 값을 원하는지 LLM에게 설명할 수 있다.
# 그런 뒤엔 우리가 LLM에게 질문을 했을 때, LLM이 text로 답하는 게 아니라 우리가 작성한 함수들을 호출한다.
# agent에게 LLM을 설명 -> LLM은 함수를 호출 -> 호출에 필요한 인자값들을 함수에 넣어준다.

st.title("QuizGPT")


# 단지 tex file을 넣어준다. -> embed (X)
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader("./files/chapter_one.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


# with st.sidebar:
choice = st.selectbox(
    "Choose what you want to use.",
    (
        "File",
        "Wikipedia Article",
    ),
)
if choice == "File":
    file = st.file_uploader(
        "Upload a .docx, .txt or .pdf file", type=["docx", "txt", "pdf"]
    )
    if file:
        docs = split_file(file)
else:
    topic = st.text_input("Search Wikipedia...")
    if topic:
        retriever = WikipediaRetriever()
        # top_k_results=1 : 첫 번째 결과만 사용
        with st.status("Searching Wikipedia..."):
            docs = retriever.get_relevant_documents(topic)
