import os
import tiktoken
import argparse
from rich.markdown import Markdown
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Example questions
# How do I use the command to show the total size of the current directory in human readable form?
# How do I use the command to show the total size of the parent directory in human readable form?
# How can I use the terminal and du to write the size of the current directory to a new file which should be on my first external SD card?
# Summarize the documentation

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("question")
  args = parser.parse_args()

  assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY!"

  # turn on wandb logging for langchain
  os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

  # optionally set your wandb settings or configs
  os.environ["WANDB_PROJECT"] = "wandb-course-project"

  # Load documentation file that has been exported like this (see create_doc.sh)
  # man du | col -bx > du.man
  document_loader = TextLoader("du.man")
  documents = document_loader.load()

  # In this version there is no chunking done because the documentation for
  # du is relatively short. Therfore no vector DB is needed
  tokenizer = tiktoken.encoding_for_model("text-davinci-003")
  tokens = len(tokenizer.encode(documents[0].page_content))
  assert tokens < 2000, f"Currently max 2000 tokens supported, {tokens} given"

  prompt_template = """Use the following documentation of a linux command line tool to answer questions about it.
  Give concise answers. The docs are between --- marks.

  ---
  {doc}
  ---

  Question: {question}
  Answer: """
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["doc", "question"]
  )
  doc = documents[0].page_content

  prompt = PROMPT.format(doc=doc, question=args.question)

  llm = OpenAI()
  response = llm.predict(prompt)
  print(response)