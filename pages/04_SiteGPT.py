import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# https://openai.com/sitemap.xml#####


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
    return docs


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
        docs = load_website(url)
        st.write(docs)
