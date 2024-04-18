import streamlit as st
from openai import OpenAI
import glob
import subprocess
from pydub import AudioSegment
import math
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser

# Q&A
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore

llm = ChatOpenAI(temperature=0.1)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


#
# @st.cache_data()
def embed_file(file_path, file_name):
    # file_content = file.read()
    # file_path = f"files/{file.name}"
    # with open(file_path, "wb") as f:
    #     f.write(file_content)
    # 이미 파일이 있기 때문에 생략이 가능하다.

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    print(retriever)
    return retriever


####

client = OpenAI()


def transcribe_chunks(chunk_folder, destination):
    # destination: text file 저장될 위치
    # glob은 순서대로 가져오지 않는다. -> 정렬 필요
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(
            destination, "a", encoding="UTF-8"
        ) as text_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            print(transcript)
            text_file.write(transcript.text)


def extract_audio_from_video(video_path):
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",  # 항상 overwrite(덮어쓰기)
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder, name):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000  # 첫 시간 선택, *1000: pydub은 밀리초단위로 작동
    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len

        chunk = track[start_time:end_time]

        chunk.export(f"{chunks_folder}/{name}_{i}.mp3", format="mp3")


######


def start_video():
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)

        status.update(label="Extracting audio...")
        # 비디오 -> mp3로 변환
        extract_audio_from_video(video_path)

        status.update(label="Cutting audio segments...")
        # mp3를 잘게 나눠서 저장
        cut_audio_in_chunks(audio_path, 10, chunks_folder, f"{video.name}")

        status.update(label="Transcribing audio...")
        # mp3에서 축출해 txt 파일로 저장
        transcribe_chunks(chunks_folder, transcript_path)


def show_tab(video_name):
    # 경로 지정
    video_path = f"./.cache/{video_name}"
    # audio_path = video_path.replace("mp4", "mp3")
    transcript_path = video_path.replace("mp4", "txt")

    transcribe_tab, summary_tab, qna_tab = st.tabs(["Transcript", "Summary", "Q&A"])
    with transcribe_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())
    with summary_tab:
        start = st.button("Generate summary")

        if start:
            # 두 개의 chain을 만든다.
            # 1. 요약
            # 2. 다른 모든 document를 요약 (이전의 요약과 새 context를 사용해 새로운 요약 만들기)
            file_name = video.name.replace("mp4", "txt")
            loader = TextLoader(f"./.cache/{file_name}")

            docs = loader.load_and_split(text_splitter=splitter)
            st.write(docs)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                다음 내용을 간결하게 요약하세요.
                "{text}"
                간결한 요약:
            """
            )
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()
            summary = first_summary_chain.invoke({"text": docs[0].page_content})
            # summary = first_summary_chain.invoke({"text": docs[0].page_content}).content -> StrOutputParser()이 모든 것을 string으로 변환

            # test_summary_chain = first_summary_prompt | llm
            # test_summary = test_summary_chain.invoke({"text": docs[0].page_content})
            # st.write(test_summary.content)
            # st.write(test_summary) -> -> content='다양한 음식과 제품 소개를 통해 자산관리와 건강에 대한 정보를 제공하는 광고이다.'

            refine_prompt = ChatPromptTemplate.from_template(
                """
                당신의 역할은 마지막 요약본을 생성하는 것입니다.
                우리는 이미 존재하는 요약본을 제공할 것입니다.
                요약본: {existing_summary}
                우리는 필요한 경우에만 이미 존재하는 요약본을 가다듬을 수 있습니다.
                -----------------
                {context}
                -----------------
                주어진 새로운 text를 통해, 기존의 요약본을 가다듬으세요.
                만약 context가 유용하지 않다면, 기존의 요약본을 return 하세요.
                """
            )
            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[:]):
                    status.update(label=f"Processing document {i}/{len(docs)-1}")
                    summary = refine_chain.invoke(
                        {"existing_summary": summary, "context": doc.page_content}
                    )
                    st.write(summary)
            st.write(summary)

    with qna_tab:
        retriever = embed_file(transcript_path, video_name)
        docs = retriever.invoke("데이터에 대해 어떤 내용을 이야기하고 있나요?")
        st.write(docs)


####

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🖥️",
)

st.markdown(
    """
        # MeetingGPT
    """
)

with st.sidebar:
    video = st.file_uploader("video", type=["mp4", "avi", "mkv", "mov"])


if video:
    has_transcript = os.path.exists(f"./.cache/{video.name}")
    if has_transcript:
        show_tab(video.name)
    else:
        start_video()
        show_tab(video.name)
else:
    st.markdown(
        """
        Welcome to MeetingGPT, upload a video and i will give you a transcript, a summary and a chat bot to ask any questions about it.
        
        Get started by uploading a video file in the sidebar.s
        """
    )
