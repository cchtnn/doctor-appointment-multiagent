import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class LLMModel:
    def __init__(self, model_name="llama-3.1-8b-instant", provider="groq"):
        """
        Initialize LLM Model
        
        Args:
            model_name: Name of the model to use
            provider: Either 'groq' or 'openai'
        """
        if not model_name:
            raise ValueError("Model is not defined.")
        
        self.model_name = model_name
        self.provider = provider.lower()
        
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
            
            self.model = ChatGroq(
                model=self.model_name,
                api_key=api_key,
                temperature=0,
                max_tokens=4192,
                timeout=60,
                max_retries=2,
            )
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            
            os.environ["OPENAI_API_KEY"] = api_key
            self.model = ChatOpenAI(model=self.model_name)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Choose 'groq' or 'openai'.")
        
    def get_model(self):
        return self.model


if __name__ == "__main__":
    # Default: Uses Groq's Llama model
    llm_instance = LLMModel()  
    llm_model = llm_instance.get_model()
    
    # Test the model
    response = llm_model.invoke("Hi, how are you?")
    print(response.content)
    
    # Optional: Use OpenAI instead
    # llm_instance = LLMModel(model_name="gpt-4o", provider="openai")
    # llm_model = llm_instance.get_model()