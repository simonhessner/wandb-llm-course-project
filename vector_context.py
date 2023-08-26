import os
import argparse
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Example questions
# How do I use the command to show the total size of the current directory in human readable form?
# How do I use the command to show the total size of the parent directory in human readable form?
# How can I use the terminal and du to write the size of the current directory to a new file which should be on my first external SD card?

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("question")
  args = parser.parse_args()

  assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY!"

  # turn on wandb logging for langchain
  os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
  os.environ["WANDB_PROJECT"] = "wandb-course-project"

  # Load documentation file that has been exported like this (see create_doc.sh)
  # man du | col -bx > du.man
  document_loader = TextLoader("du.man")
  documents = document_loader.load()
  assert len(documents) == 1
  document = documents[0]
  document.page_content = document.page_content.replace('    ', ' ')

  # Create chunks
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 50,
    length_function = len,
    is_separator_regex = False,
  )
  chunks = text_splitter.create_documents([document.page_content])

  # Create vector DB with OpenAI embeddings
  embeddings = OpenAIEmbeddings()
  if os.path.isdir("chroma_db"):
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    print("loaded chroma_db vector store")
  else:
    db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    print("saved chroma_db vector store")
  retriever = db.as_retriever(search_kwargs=dict(k=3))

  prompt_template = """Use the following documentation of a linux command line tool to answer questions about it.
  Give concise answers. If you don't know the answer, say "I don't know". The docs are between --- marks.

  ---
  {context}
  ---

  Question: {question}
  Answer: """
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )

  qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT})
  result = qa.run(args.question)
  print(result)