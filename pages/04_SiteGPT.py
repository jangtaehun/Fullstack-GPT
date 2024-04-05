import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# https://openai.com/sitemap.xml#####

llm = ChatOpenAI(
    temperature=0.1,
)
answers_prompt = ChatPromptTemplate.from_template(
    """
    답변할 수 없다면 아무말이나 지어내지말고 모른다고 하세요.
    그리고 각 답변을 0부터 5까지의 점수로 평가해주세요.
    0점은 사용자에게 쓸모없음, 5점은 사용자에게 매우 유용함을 의미합니다.
    사용자 question에 대한 예제입니다.
    
    Make sure to inclide the answer's score.
    
    Context: {context}
    
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    # st.write(answers)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(input):
    answers = input["answers"]
    question = input["question"]
    choose_chain = choose_prompt | llm
    condence = "\n\n".join(
        f"Answer: {answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke({"question": question, "answers": condence})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " '")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
# load_website가 한 번 실행된 후 같은 url로 다시 호출되면, 아무 일도 하지않게 설정
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/blog\/).*",
        ],
        parsing_function=parse_page,
        # beatiful soup의 동작을 커스텀할 수 있는 function을 작성할 수 있다.
    )
    loader.requests_per_second = 5  # 속도는 느려지지만 차단은 안 당한다
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.set_page_config(
    page_title="Site GPT",
    page_icon="🖥️",
)

# from langchain.document_transformers import Html2TextTransformer
# html2text_transformer = Html2TextTransformer()

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.

"""
)

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    # async chromium loader
    # loader = AsyncChromiumLoader([url])
    # docs = loader.load()
    # transformed = html2text_transformer.transform_documents(docs)
    # st.write(docs)

    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap url")
    else:
        retriever = load_website(url)

        query = st.text_input("Ask a question to the website.")
        if query:
            # chain 2개: 1. 모든 개별 document에 대한 답변 생성 및 채점, 2. 모든 답변을 가진 마지막 시점에 실행 -> 점수가 제일 높고, 가장 최신 정보를 담고 있는 답변 고르기
            chain = (
                {"docs": retriever, "question": RunnablePassthrough()}
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            # get_answer이 무엇을 반환하든 choose_answer에 전달된다. / get_answer의 출력 -> choose_answer의 입력값

            result = chain.invoke(query)
            # 질문이 retriever로 전달 -> docs 반환 -> RunnablePassthrough는 question 값으로 교체된다.
            # dictionary가 get_answers function의 입력값으로 전달되고, 그 function은 dictionary에서 documents와 question 값을 추출

            st.write(result.content.replace("$", "\$"))


# code challenge
# streaming, 사용자 및 assistant message, chat history
# text 실시간 생성,

# 추가적으로!! 사용자 question을 cache -> 똑같은 질문을 받으면 LLM에게 반복적으로 질문하는 것보다 이미 한 번 얻은 답변을 반환
# 처음 question을 받았을 때 응답을 보내면서 어딘가에 저장, 그 응답을 생성한 question도 어딘가에 보관
# retrieve와 map re-rank를 작동시키기 전에 LLM에게 먼저 물어본다
