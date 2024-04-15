import os
import requests
from typing import Type
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import SystemMessage
from langchain.retrievers import WikipediaRetriever

# llm = ChatOpenAI(temperature=0.1, model_name="gpt-4-turbo")
# alpha_vantage_api_key = os.environ.get("GI7UXI89QUWWJFK5")


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query you will search for.Example query: Stock Market Symbol for Apple Company"
    )


class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company.
    It takes a query as an argument.
    Example query: Stock Market Symbol for apple Company
    """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = (
        StockMarketSymbolSearchToolArgsSchema
    )

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


class CompanyOverviewArgsSchema(BaseModel):
    symbol: str = Field(description="Stock symbol of the company.Example: AAPL,TSLA")


class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        print(requests)
        r = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()


class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()["annualReports"]


class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockperformance"
    description = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        response = r.json()
        return list(response["Weekly Time Series"].items())[:200]


class WikiArgsSchema(BaseModel):
    term: str = Field(description="The search of the company.Example: Apple company")


class WikiSearchTool(BaseTool):
    name = "WikiSearchTool"
    description = """
    Use this tool to find the company's information.
    Example term: Apple company
    """
    args_schema: Type[WikiArgsSchema] = WikiArgsSchema

    def _run(self, term):
        retriever = WikipediaRetriever(top_k_results=4, lang="ko")
        docs = retriever.get_relevant_documents(term)
        return docs


st.set_page_config(
    page_title="InvestorGPT",
    page_icon="💼",
)

with st.sidebar:
    openaikey = None
    openaikey = st.text_input("Write Your OpenAI API key: ", type="password")
    os.environ["OPENAI_API_KEY"] = openaikey

llm = ChatOpenAI(temperature=0.1, model_name="gpt-4-turbo")
alpha_vantage_api_key = os.environ.get("GI7UXI89QUWWJFK5")

agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
        # 주식 심볼을 아는 데 사용
        CompanyOverviewTool(),
        # 정보를 얻는데 사용
        CompanyIncomeStatementTool(),
        # llm이 사용 안 할 수도 있다. llm이 더 많은 정보가 필요하다고 생각할 때만 사용
        CompanyStockPerformanceTool(),
        WikiSearchTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are a hedge fund manager.
                
            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
                
            Consider the performance of a stock, the company overview and the income statement.
                
            Be assertive in your judgement and recommend the stock or advise the user against it.                                
            """
        )
    },
)

# streamlit

st.markdown(
    """
        # InvestorGPT
                
        Welcome to InvestorGPT.
                
        Write down the name of a company and our Agent will do the research for you.
    """
)

company = st.text_input("Write the name of the company you are interested on.")

if company:
    result = agent.invoke(company)
    st.write(result["output"].replace("$", "\$"))
