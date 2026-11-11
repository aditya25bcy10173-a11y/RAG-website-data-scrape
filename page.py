
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

os.environ.setdefault("USER_AGENT", "MyRAGApplication/1.0")

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Check API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# Website URL
url = "https://www.python.org/"

# Load website
loader = WebBaseLoader(
    web_path=url,
    raise_for_status=True
)

docs = loader.load()

print("Website loaded successfully!")
print("Total documents:", len(docs))

# Extract website content
website_content = "\n\n".join(
    doc.page_content for doc in docs
)

print("\nWebsite content extracted!")

# Create prompt
template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant that extracts and summarizes website content. "
        "Give clear and accurate answers based only on the provided content."
    ),
    (
        "human",
        "Summarize this website content:\n\n{data}"
    )
])

# Initialize Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0
)

# Create chain
chain = template | model

# Generate summary
result = chain.invoke({
    "data": website_content
})

print("\n========== WEBSITE SUMMARY ==========\n")
print(result.content)
