import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📑",
)

llm = ChatOpenAI(temperature=0.1)

# 이 코드를 사용하면 파일을 삭제하고 다시 첨부해도 기록이 남는다.
# streamlit의 session state는 여러 번 재실행을 해도 data가 보존될 수 있게 해준다.
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
# 만약 st.session_state가 messages라는 key를 가지고 있지 않다면 빈 list로 initialize 한다.
# 가지고 있다면 그 messages를 유지하고 싶기 때문에 아무것도 하지 않는다.


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    # print(file_content) : 첨부한 file의 내용

    file_path = f"./.cache/files/{file.name}"
    # print(file_path) : ./.cache/files/첨부한 file의 이름

    with open(file_path, "wb") as f:
        f.write(file_content)
    # file_path에 첨부한 file 이름의 file을 만든다.

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader("./files/chapter_one.txt")
    print(loader)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


# message, role 저장
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


# message 기록 보이기, 저장X
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# template
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    {context}만을 사용해 질문에 답변을 해주세요. 모른다면 모른다고 하세요. 꾸며내지 마세요.
    """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")
# st.markdown(
#     """
# Welcome!
# Use this chatbot to ask questions to an AI about your files!
# """
# )

# with st.sidebar:
file = st.file_uploader(
    """
Use this chatbot to ask questions to an AI about your files! \n
You can upload a .txt .pdf or .docx file
                        """,
    type=["pdf", "txt", "docx"],
)


# file이 존재하면 실행
if file:
    retriever = embed_file(file)
    send_message("나는 준비됐어 물어봐!!", "ai", save=False)
    paint_history()
    message = st.chat_input("첨부한 파일에 대해 어떤 것이든 물어봐!!")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | template
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")
        # docs = retriever.invoke(message) -> retriever

        # 사용자가 보내는 message는 RunnablePassthrough()로, langchain은 사용자의 message를 가지고 retriever을 호출
        # 그리고 LangChain은 많은 양의 documents인 retriever의 출력물을 준다. 그게 format docs function으로 주어지고 변환해준다.
else:
    st.session_state["messages"] = []
    # 초기화
