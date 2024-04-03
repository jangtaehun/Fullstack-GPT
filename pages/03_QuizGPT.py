import streamlit as st
import json
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser

# 파일에 대한 질문을 하는 대신에 LLM이 우리에게 파일의 내용과 관련된 질문을 하게끔 명령할 것이다.
# Wikipedia Retriever를 사용
# OutputParser를 배우는 것에 포인트를 둬야 한다.

# Function Calling
# 우리가 만든 함수가 어떻게 생겼는지, 어떤 parameter 값을 원하는지 LLM에게 설명할 수 있다.
# 그런 뒤엔 우리가 LLM에게 질문을 했을 때, LLM이 text로 답하는 게 아니라 우리가 작성한 함수들을 호출한다.
# agent에게 LLM을 설명 -> LLM은 함수를 호출 -> 호출에 필요한 인자값들을 함수에 넣어준다.


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        AI는 선생님 역할을 해주는 훌륭한 조수입니다.
        
        아래의 내용에 따라 유저의 텍스트에 대한 지식을 판단하는 문제 10개를 만드세요
        
        각각의 질문들은 모두 4개의 선택지가 있어야 하고 그 중 3개는 오답 1개는 정답이어야 합니다.
        
        (O)문자를 통해 정답을 표시하도록 합니다.
        
        질문 예제:
        
        
        질문: 바다는 어떤 색입니까?
        
        선택지s: 빨강 | 노랑 | 초록 | 파랑(O)
        
        
        질문: 대한민국의 수도는 어디입니까?
        
        선택지s: 평양 | 도쿄 | 베이징 | 서울(O)
        
        
        질문: 아바타1은 언제 개봉했나요?
        
        선택지s: 2009(O) | 2001 | 1998 | 2007
        
        
        질문: 줄리어스 시저는 누구인가요?
        
        선택지s: 화가 | 로마의 황제(O) | 배우 | 의사
        
        
        너의 차례야!!
        
        context: {context}
        """,
        )
    ]
)
questions_chain = {"context": format_docs} | question_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    당신은 강력한 formatting 알고리즘입니다.
    
    당신은 시험 문제를 json 형식으로 바꿔줍니다.
    (O) 표시가 있는 보기가 정답입니다.
    
    질문 예제:
        
        질문: 바다는 어떤 색입니까?
        선택지s: 빨강 | 노랑 | 초록 | 파랑(O)
        
        질문: 대한민국의 수도는 어디입니까?
        선택지s: 평양 | 도쿄 | 베이징 | 서울(O)
        
        질문: 아바타1은 언제 개봉했나요?
        선택지s: 2009(O) | 2001 | 1998 | 2007
        
        질문: 줄리어스 시저는 누구인가요?
        선택지s: 화가 | 로마의 황제(O) | 배우 | 의사
        
        
        결과 예시:
        
        '''json
        {{ "questions": [
            {{
                "질문": "바다는 어떤 색입니까?",
                "선택지s": [
                    {{
                        "선택지": "빨강",
                        "정답": false
                    }},
                    {{
                        "선택지": "노랑",
                        "정답": false
                    }},
                    {{
                        "선택지": "초록",
                        "정답": false
                    }},
                    {{
                        "선택지": "파랑",
                        "정답": true
                    }},
                ]
            }},
            {{
                "질문": "대한민국 수도는 어디입니까?",
                "선택지s": [
                    {{
                        "선택지": "평양",
                        "정답": false
                    }},
                    {{
                        "선택지": "도쿄",
                        "정답": false
                    }},
                    {{
                        "선택지": "배이징",
                        "정답": false
                    }},
                    {{
                        "선택지": "서울",
                        "정답": true
                    }},
                ]
            }},
            {{
                "질문": "줄리어스 시저는 누구입니까?",
                "선택지s": [
                    {{
                        "선택지": "화가",
                        "정답":false
                    }},
                    {{
                        "선택지": "로마의 황제",
                        "정답": true
                    }},
                    {{
                        "선택지": "배우",
                        "정답": false
                    }},
                    {{
                        "선택지": "의사",
                        "정답": false
                    }},
                ]
            }},
            {{
                "질문": "바다는 어떤 색입니까?",
                "선택지s": [
                    {{
                        "선택지": "빨강",
                        "정답": false
                    }},
                    {{
                        "선택지": "노랑",
                        "정답": false
                    }},
                    {{
                        "선택지": "초록",
                        "정답": false
                    }},
                    {{
                        "선택지": "파랑",
                        "정답": true
                    }},
                ]
            }},
            
        ]
            
        }}
        '''
        Your turn!

        Questions: {context}
    """,
        )
    ]
)
# {{}} langchain이 이 부분을 format하지 않기를 바리기 때문
# {}: 어떤 값을 주입하고 싶을 때 중괄호 사용

formatting_chain = formatting_prompt | llm


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


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    # hash: 들어오는 데이터의 서명을 생성한다는 것
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    # top_k_results=1 : 첫 번째 결과만 사용
    docs = retriever.get_relevant_documents(term)
    return docs


# with st.sidebar:
docs = None
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
        docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    QuizeGPT는 Wikipedia와 사용자가 제공한 파일을 이용해 퀴즈를 만들어 제공합니다.
    
    사용을 원하신다면 파일을 업로드하거나 Wikipedia을 선택해 검색어를 입력해주세요.
    """
    )
else:

    # questions_response = questions_chain.invoke(docs)
    # st.write(questions_response.content)
    # # 문서를 chain을 불러올 때 넘겨주고 -> format_docs로 넘어간 뒤 -> string을 반환하고 -> context의 값이 된다. -> prompt 안으로 들어간다.
    # formatting_response = formatting_chain.invoke(
    #     {"context": questions_response.content}
    # )
    response = run_quiz_chain(docs, topic if topic else file.name)

    # 접었다가 펴기 가능하게
    st.write(response)

    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["질문"])
            value = st.radio(
                "Select an option.",
                [선택지["선택지"] for 선택지 in question["선택지s"]],
                index=None,
            )
            if {"선택지": value, "정답": True} in question["선택지s"]:
                st.success("정답")
            elif value is not None:
                st.error("오답")
        button = st.form_submit_button()
