from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, AIMessage
import getpass
import os
import warnings

warnings.filterwarnings("ignore")

def rag_chain_with_history(llm, retriever):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

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

    qa_system_prompt = """You are a chatbot that acts as the person described in the resume. 
Answer all questions as if you are this person. Use the information in the provided resume 
to guide your responses. If the question goes beyond the information in the resume, respond with:
"That information is not in my Database."

{context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def setup_google_api_key():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDvohaTjjuwIYrFxL7XRmc205rWMC3-2hM"

def setup_llm():
    setup_google_api_key()  

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",  
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    return llm

def load_documents(file_paths):
    documents = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            continue
        documents.extend(loader.load_and_split())
    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_embeddings(model_name="all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

def store_in_vectorstore(documents, embedding):
    return Chroma.from_documents(documents=documents, embedding=embedding)

def setup_conversational_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    ai_chain = rag_chain_with_history(llm, retriever)
    
    return ai_chain, memory

# Initialize ai_chain and memory at the module level
llm = setup_llm()
file_paths = ["/Users/bramhabajannavar/Desktop/Portfolio/Backend/portfolio_api/RESUME.pdf"]
documents = load_documents(file_paths)
knowledge_base = split_documents(documents)
embedding = create_embeddings()
vectorstore = store_in_vectorstore(knowledge_base, embedding)
ai_chain, memory = setup_conversational_chain(llm, vectorstore)

def process_input(user_input, chat_history):
    # Invoke RAG model with chat history
    response = ai_chain.invoke({"chat_history": chat_history, "input": user_input})

    # Extract model output
    answer = response.get("output") or response.get("answer", "Unexpected result format.")

    # Update chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})

    # Print the output
    print(f"User Input: {user_input}")
    print(f"Model Response: {answer}")

    return answer

def main():
    chat_history = []

    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the application.")
            break
        try:
            result = process_input(query, chat_history)
        except Exception as e:
            print(f"Error during query processing: {e}")

if __name__ == "__main__":
    main()