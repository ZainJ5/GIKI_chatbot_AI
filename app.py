from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import logging
import time
import random
import requests
from datetime import datetime
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChatbotApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_cors()
        self.setup_config()
        self.retriever = None
        self.setup_routes()
        
        if not self.test_llm():
            logger.error("LLM test failed - application may not function correctly")
        else:
            logger.info("LLM test successful")
            
        if not self.test_embeddings():
            logger.error("Embeddings test failed - application may not function correctly")
        else:
            logger.info("Embeddings test successful")
        
        self.load_data()

    def setup_cors(self):
        CORS(self.app, resources={
            r"/*": {
                "origins": ["http://localhost:5173"],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })

    def test_llm(self):
        """Test the LLM connection and functionality"""
        url = f"{self.AZURE_OPENAI_ENDPOINT}/openai/deployments/{self.AZURE_CHAT_DEPLOYMENT}/chat/completions?api-version={self.AZURE_OPENAI_API_VERSION}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.AZURE_OPENAI_CHAT_API_KEY
        }
        payload = {
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        try:
            logger.info(f"Testing LLM connection to {url}")
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"LLM test response: {content[:50]}...")
                return True
            else:
                logger.error(f"LLM test failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error testing LLM connection: {str(e)}")
            return False

    def test_embeddings(self):
        """Test the embeddings connection and functionality"""
        url = f"{self.AZURE_OPENAI_EMBEDDINGS_ENDPOINT}/openai/deployments/{self.AZURE_EMBEDDING_DEPLOYMENT}/embeddings?api-version={self.AZURE_OPENAI_EMBEDDINGS_API_VERSION}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.AZURE_OPENAI_EMBEDDINGS_API_KEY
        }
        payload = {
            "input": ["test connection"]
        }
        
        try:
            logger.info(f"Testing embeddings connection to {url}")
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                embedding = response_data.get("data", [{}])[0].get("embedding", [])
                logger.info(f"Embeddings test successful - received vector of length {len(embedding)}")
                return True
            else:
                logger.error(f"Embeddings test failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error testing embeddings connection: {str(e)}")
            return False

    def setup_config(self):
        self.AZURE_OPENAI_CHAT_API_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://u2023-m9d0ac83-eastus2.cognitiveservices.azure.com")
        self.AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        
        self.AZURE_OPENAI_EMBEDDINGS_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY")
        self.AZURE_OPENAI_EMBEDDINGS_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT", "https://zainjamshaidai.openai.azure.com")
        self.AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.AZURE_OPENAI_EMBEDDINGS_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION", "2023-05-15")
        
        if not self.AZURE_OPENAI_CHAT_API_KEY:
            logger.error("AZURE_OPENAI_CHAT_API_KEY environment variable is not set")
            raise ValueError("AZURE_OPENAI_CHAT_API_KEY environment variable is not set")
            
        if not self.AZURE_OPENAI_EMBEDDINGS_API_KEY:
            logger.error("AZURE_OPENAI_EMBEDDINGS_API_KEY environment variable is not set")
            raise ValueError("AZURE_OPENAI_EMBEDDINGS_API_KEY environment variable is not set")
        
        self.JSON_PATH = os.getenv("JSON_DATA_PATH", os.path.join(os.path.dirname(__file__), "data.json"))
        if not os.path.exists(self.JSON_PATH):
            logger.error(f"Data file not found at {self.JSON_PATH}")
            raise FileNotFoundError(f"Data file not found at {self.JSON_PATH}")

    def create_document(self, source, post):
        document_mappings = {
            "Instagram": lambda: Document(
                page_content=f"Instagram Post: {post.get('caption', '')}",
                metadata={
                    "source": "Instagram",
                    "date": post.get("date"),
                    "type": post.get("type"),
                    "image_url": post.get("image_url"),
                    "extracted_text": post.get("extracted_text")
                }
            ),
            "Reddit": lambda: Document(
                page_content=f"Reddit Post: {post.get('title', '')} - {post.get('content', '')}",
                metadata={
                    "source": "Reddit",
                    "date": post.get("date"),
                    "type": post.get("type"),
                    "author": post.get("author"),
                    "url": post.get("url"),
                    "comments": post.get("comments", [])
                }
            ),
            
        }
        return document_mappings.get(source, lambda: None)()

    def process_documents(self, documents, chunk_size=500, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = []
        for doc in documents:
            if len(doc.page_content) > chunk_size:
                chunks.extend(text_splitter.split_documents([doc]))
            else:
                chunks.append(doc)
        return chunks

    def create_embeddings_with_retry(self, texts, max_retries=5):
        """Create embeddings with retry logic and custom implementation if needed"""
        retries = 0
        delay = 1
        
        try:
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=self.AZURE_EMBEDDING_DEPLOYMENT,
                openai_api_key=self.AZURE_OPENAI_EMBEDDINGS_API_KEY,
                azure_endpoint=self.AZURE_OPENAI_EMBEDDINGS_ENDPOINT,
                api_version=self.AZURE_OPENAI_EMBEDDINGS_API_VERSION,
                chunk_size=10,  
                max_retries=6,   
                timeout=120     
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error with LangChain embeddings: {str(e)}")
            logger.info("Falling back to custom implementation...")
            
        class CustomEmbeddings:
            def __init__(self, app):
                self.app = app
                
            def embed_documents(self, texts):
                all_embeddings = []
                for i in range(0, len(texts), 5): 
                    batch = texts[i:i+5]
                    url = f"{self.app.AZURE_OPENAI_EMBEDDINGS_ENDPOINT}/openai/deployments/{self.app.AZURE_EMBEDDING_DEPLOYMENT}/embeddings?api-version={self.app.AZURE_OPENAI_EMBEDDINGS_API_VERSION}"
                    headers = {
                        "Content-Type": "application/json",
                        "api-key": self.app.AZURE_OPENAI_EMBEDDINGS_API_KEY
                    }
                    payload = {"input": batch}
                    
                    for attempt in range(max_retries):
                        try:
                            response = requests.post(url, headers=headers, json=payload, timeout=30)
                            if response.status_code == 200:
                                data = response.json()
                                batch_embeddings = [item["embedding"] for item in data["data"]]
                                all_embeddings.extend(batch_embeddings)
                                time.sleep(1)
                                break
                            elif response.status_code in [429, 503]:
                                wait_time = min(delay * (2 ** attempt) + random.uniform(0, 1), 60)
                                logger.info(f"Rate limited. Waiting {wait_time:.2f} seconds before retry...")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"Error: {response.status_code}, {response.text}")
                                raise Exception(f"API Error: {response.status_code}")
                        except Exception as e:
                            if attempt == max_retries - 1:
                                raise
                            wait_time = min(delay * (2 ** attempt) + random.uniform(0, 1), 60)
                            logger.info(f"Error, retrying in {wait_time:.2f} seconds: {str(e)}")
                            time.sleep(wait_time)
                
                return all_embeddings
                
            def embed_query(self, text):
                return self.embed_documents([text])[0]
        
        return CustomEmbeddings(self)

    def load_data(self):
        try:
            logger.info("Loading data from JSON file")
            with open(self.JSON_PATH, 'r', encoding='utf8') as f:
                data = json.load(f)

            documents = []
            for source_obj in data:
                source = source_obj.get("source")
                for post in source_obj.get("data", []):
                    if doc := self.create_document(source, post):
                        documents.append(doc)

            processed_chunks = self.process_documents(documents)
            
            embeddings = self.create_embeddings_with_retry(processed_chunks)
            
            if len(processed_chunks) > 50:
                logger.info(f"Processing {len(processed_chunks)} documents in smaller batches")
                batch_size = 50
                
                first_batch = processed_chunks[:batch_size]
                vectorstore = FAISS.from_documents(first_batch, embeddings)
                
                for i in range(batch_size, len(processed_chunks), batch_size):
                    batch = processed_chunks[i:i+batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1} of {(len(processed_chunks) + batch_size - 1) // batch_size}")
                    try:
                        temp_vs = FAISS.from_documents(batch, embeddings)
                        vectorstore.merge_from(temp_vs)
                        time.sleep(2) 
                    except Exception as e:
                        logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            else:
                vectorstore = FAISS.from_documents(processed_chunks, embeddings)
            
            self.retriever = vectorstore.as_retriever()
            
            logger.info(f"Data loaded successfully. Total chunks: {len(processed_chunks)}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def setup_routes(self):
        @self.app.route('/health')
        def health_check():
            return jsonify({
                "status": "healthy",
                "retriever_status": "initialized" if self.retriever else "not initialized",
                "timestamp": datetime.utcnow().isoformat()
            })

        @self.app.route('/ask', methods=['POST'])
        def ask_question():
            try:
                data = request.json
                question = data.get("question")

                if not question:
                    return jsonify({"error": "Question is required"}), 400

                template = """
                Question: {question}
                Context: {context}
                
                Instructions:
                1. Comprehensive Analysis:
                   - Carefully analyze the provided context
                   - Identify key information directly relevant to the question
                   - Extract precise, factual details
                
                2. Answer Evaluation:
                   - If context provides sufficient information:
                     * Construct a concise, accurate answer
                     * Directly cite source information
                     * Use clear, precise language
                   
                   - If context is insufficient or irrelevant:
                     * Use your dataset to answer the question
                     * Answer that question on your own
                     * Use web to get the most rellevant information 
                     * Clearly state "Insufficient contextual information"
                     * Provide a disclaimer about potential limitations
                     * Offer guidance on seeking additional sources
                
                3. Response Formatting:
                   - Begin with a direct, clear answer
                   - Use bullet points or numbered lists if explaining complex information
                   - Include source credibility indicators when possible
                   - Maintain objectivity and neutrality
                
                4. Additional Guidance:
                   - Recommend verification for time-sensitive or rapidly changing information
                   - Suggest consulting domain experts or primary sources for critical decisions
                
                Answer Template:
                [Concise Direct Answer]
                            
                Detailed Explanation:
                [Optional expanded context and reasoning]

                Don't start the answer with Answer heading just give answer that I can directly show on UI
                """

                prompt = ChatPromptTemplate.from_template(template)
                
                try:
                    llm = AzureChatOpenAI(
                        azure_deployment=self.AZURE_CHAT_DEPLOYMENT,
                        openai_api_key=self.AZURE_OPENAI_CHAT_API_KEY,
                        azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
                        api_version=self.AZURE_OPENAI_API_VERSION,
                        max_retries=6,
                        timeout=120
                    )
                    
                    rag_chain = (
                        {"context": self.retriever, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                    )

                    response = rag_chain.invoke(question)
                    answer = str(response)
                except Exception as e:
                    logger.error(f"Error getting response from LLM: {str(e)}")
                    answer = "I'm sorry, I encountered an issue processing your question. Please try again in a few moments."

                return jsonify({
                    "answer": answer,
                    "timestamp": datetime.utcnow().isoformat()
                })

            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                return jsonify({"error": str(e)}), 500

    def run(self):
        port = int(os.getenv("PORT", 5000))
        self.app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    chatbot = ChatbotApp()
    chatbot.run()