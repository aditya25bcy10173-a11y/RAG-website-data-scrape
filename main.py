
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Load PDF
loader = PyPDFLoader("document_loaders/GRU.pdf")
docs = loader.load()

print(f"Total pages extracted: {len(docs)}")

# Create prompt
template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that extracts and summarizes information from PDF pages."),
    ("human", "Analyze the following PDF page and provide a clear summary:\n\n{data}")
])

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

# Create chain
chain = template | model

# Extract and summarize each page
for i, doc in enumerate(docs):

    print(f"\n--- PAGE {i + 1} ---")

    result = chain.invoke({
        "data": doc.page_content
    })

    print(result.content)
