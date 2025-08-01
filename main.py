from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from FaissVector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering question about a pizza restaurant

Here are some relevant reviews: {reviews}
Here is the question to answer: {question}"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

# while True:
#     print("\n\n-------------------------------------")
#     question = input("Ask your question (q to quit)")
#     print("\n\n")
#     if question == "q":
#         break
    
#     reviews = retriever.invoke(question)
#     result=chain.invoke({"reviews": reviews, "question": question})
#     print(result)


def response(question):
    reviews = retriever.invoke(question)
    result=chain.invoke({"reviews": reviews, "question": question})
    return result
    
