import os
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from embed import get_retriever

def get_llm():
    ollama_host = os.getenv('OLLAMA_HOST', 'http://axora-ollama:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'phi3:mini')
    
    print(f"Initializing LLM:")
    print(f"  - Model: {ollama_model}")
    print(f"  - Host: {ollama_host}\n")
    
    llm = OllamaLLM(
        base_url=ollama_host,
        model=ollama_model,
        temperature=0.7,
    )
    
    return llm


def get_rag_chain():    
    llm = get_llm()
    retriever = get_retriever(
        search_type="similarity", 
        search_kwargs={"k": 4}
    )

    prompt = PromptTemplate.from_template(
        """You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question:
{input}

Answer clearly and concisely based only on the given context. If the context doesn't contain relevant information, say so.
"""
    )
    

    def log_prompt(inputs):
        print("\n" + "=" * 60)
        print("PROMPT SENT TO LLM")
        print("=" * 60)
        # inputs here has 'context' and 'input' keys already populated
        rendered = prompt.format(**inputs)
        print(rendered)
        return rendered

    rag_chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough(),
        }
        | RunnableLambda(log_prompt)
        | llm
        | StrOutputParser()
    )

    return rag_chain