import requests
import io
import PyPDF2
import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from datetime import datetime


# Load environment variables from .env file
load_dotenv()

port = int(os.getenv("PORT", 8000))

# Gemini 2.0 Flash API via LangChain
class GeminiApi:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # For embeddings
        # Initialize Gemini 2.0 Flash via LangChain using GEMINI_API_KEY from .env
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key
        )

    def parse_query(self, query: str) -> Dict:
        """Extract structured data from natural language query using Gemini 2.0 Flash."""
        try:
            prompt_template = PromptTemplate(
                input_variables=["query"],
                template="""
                Parse the following query into a JSON object with fields 'raw_query', 'topic', and 'query_type':
                Query: {query}
                Example output: {{"raw_query": "{query}", "topic": "maternity", "query_type": "cover"}}
                Ensure the output is valid JSON.
                """
            )
            prompt = prompt_template.format(query=query)
            response = self.llm.invoke(prompt)

            cleaned = self.clean_response(response.content)
            parsed = json.loads(cleaned)
            return parsed
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API parsing error: {str(e)}")
        
    def clean_response(self, response_text: str) -> str:
        """Removes the first and last lines (e.g., ```json wrappers) from multi-line LLM responses."""
        lines = response_text.strip().splitlines()
        if len(lines) <= 2:
            return response_text
        return "\n".join(lines[1:-1])



    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts using sentence-transformers."""
        # Note: If Gemini provides embeddings, replace with LangChain integration
        return self.embedder.encode(texts, convert_to_numpy=True)

    def evaluate_answer(self, query: Dict, clauses: List[Tuple[str, str, float]]) -> Dict:
        """Generate answer and justification using Gemini 2.0 Flash."""
        try:
            prompt_template = PromptTemplate(
                input_variables=["query", "clauses"],
                template="""
                Given the query: {query}
                And relevant clauses: {clauses}
                Provide a concise answer and justification in JSON format:
                {{
                    "answer": "string",
                    "justification": [{{"clause_id": "string", "text": "string", "relevance_score": float, "reason": "string"}}]
                }}
                Ensure the answer addresses the query topic and type (e.g., coverage, conditions).
                Ensure the output is valid JSON.
                """
            )
            prompt = prompt_template.format(
                query=query['raw_query'],
                clauses=json.dumps([{'clause_id': c[0], 'text': c[1], 'relevance_score': c[2]} for c in clauses])
            )
            response = self.llm.invoke(prompt)
            cleaned = self.clean_response(response.content)
            result = json.loads(cleaned)

            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API evaluation error: {str(e)}")

@dataclass
class Document:
    """Class to handle document loading from PDF Blob URL."""
    blob_url: str

    def extract_text(self) -> str:
        """Extract text from PDF Blob URL."""
        try:
            response = requests.get(self.blob_url)
            response.raise_for_status()
            pdf_file = io.BytesIO(response.content)
            reader = PyPDF2.PdfReader(pdf_file)
            return ''.join(page.extract_text() or '' for page in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error extracting text from {self.blob_url}: {str(e)}")

class QueryRetrievalSystem:
    """Main class for query processing and document retrieval."""
    def __init__(self, llm_api: GeminiApi):
        self.llm_api = llm_api
        self.index = None
        self.clauses = []

    def index_document(self, document: Document):
        """Index document clauses using FAISS."""
        text = document.extract_text()
        self.clauses = [(f"clause_{i}", clause) for i, clause in enumerate(text.split('\n\n')) if clause.strip()]
        clause_texts = [clause[1] for clause in self.clauses]
        embeddings = self.llm_api.generate_embeddings(clause_texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search_clauses(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Retrieve top-k relevant clauses using FAISS semantic search."""
        query_embedding = self.llm_api.generate_embeddings([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [(self.clauses[i][0], self.clauses[i][1], 1.0 - d/2.0) for i, d in zip(indices[0], distances[0])]

    def process_query(self, query: str, document: Document) -> Dict:
        """Process a single query and generate answer with justification."""
        parsed_query = self.llm_api.parse_query(query)
        if not self.index:
            self.index_document(document)
        matched_clauses = self.search_clauses(query, top_k=3)
        result = self.llm_api.evaluate_answer(parsed_query, matched_clauses)
        return {
            'query': parsed_query['raw_query'],
            'answer': result['answer'],
            'justification': result['justification']
        }

    def process_questions(self, questions: List[str], document: Document) -> List[Dict]:
        """Process multiple questions and return answers."""
        if not self.index:
            self.index_document(document)
        return [self.process_query(question, document) for question in questions]

# FastAPI setup
app = FastAPI(
    title="HackRX Query Retrieval API",
    description="API for processing queries against policy documents using Gemini 2.0 Flash via LangChain",
    version="1.0",
    openapi_url="/api/v1/openapi.json"
)

# Authentication
API_KEY = "5567e44f0baf1876abcb15031a2c1f25dcf5268280928d3c94cc955c52b8f99c"
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# Request and Response Models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Global system instance
llm_api = GeminiApi()
processor = QueryRetrievalSystem(llm_api)

@app.post("/hackrx/run", response_model=QueryResponse, tags=["Query Processing"])
async def run_query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
    """Process multiple questions against a PDF document and save the response."""
    try:
        document = Document(blob_url=request.documents)
        results = processor.process_questions(request.questions, document)
        
        # Save the full results to a JSON file before returning
        filename = f"query_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        return {"answers": [r['answer'] for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)