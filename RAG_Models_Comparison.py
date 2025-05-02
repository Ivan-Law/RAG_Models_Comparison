from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import Sequence
from typing_extensions import Annotated, TypedDict
import wget
import datetime

filename = 'professor.txt'
url = 'https://www.biosciences-labs.bham.ac.uk/filatov/cv.txt'
myQueries = [{"input": "Which university did he or she get Ph.D.?"},
             {"input": "Which countries did he or she work afterwards?"},
             {"input": "Apart from English, what other languages does he or she speak?"},
             ]

################

class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def myLLM_loop(start_time, local_llm):

    def call_model(state: State):
        response = rag_chain.invoke(state)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }

    wget.download(url, out=filename)
    loader = TextLoader(filename)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings)
    retriever = docsearch.as_retriever()
    llm = ChatOllama(model=local_llm, temperature=0.5, num_predict=256)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are a helpful assistant to answer questions. "
        "Answer the questions based on the pieces of retrieved context. "
        "If the question cannot be answered using the information provided, answer with 'I don't know.'"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "42"}}

    print('By using ' + local_llm + ', the LLM generates :-')
    for i in myQueries:
        result = app.invoke(i, config=config)
        print(result["answer"])
    print('=========================================================================')
    print('By using ' + local_llm + ', it takes ' + str((datetime.datetime.now().replace(microsecond=0) - start_time)) + ' to finish.')
    print('=========================================================================')

myLocal_llm = ["gemma3:1b",
               "qwen3:1.7b",
               "llama3.1",
               "gemma3:4b",
               "cogito:3b",
               "cogito:8b",
               "deepseek-r1:1.5b",
               "deepseek-r1:8b"]
for j in myLocal_llm:
    myLLM_loop(datetime.datetime.now().replace(microsecond=0), j)