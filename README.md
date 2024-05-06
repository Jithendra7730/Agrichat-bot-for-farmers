!pip install langchain
!pip install openai
!pip install gradio
!pip install huggingface_hub
import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

# Set OpenAI API key
OPENAI_API_KEY = "OPENAI_API_KEY"

# Set OpenAI API key as environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Template for conversation prompt
template = """
You are a helpful assistant providing agriculture advice.
{chat_history}
User: {user_message}
Chatbot:
"""

prompt = PromptTemplate(input_variables=["chat_history", "user_message"], template=template)

# Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize language model chain
llm_chain = LLMChain(
    llm=ChatOpenAI(temperature='0.5', model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
    prompt=prompt,
    verbose=True,
    memory=memory
)

def is_agriculture_related(query):
    # Agriculture-related keywords
    agriculture_keywords = [
        "crop", "plant", "harvest", "yield", "crop rotation", "crop variety",
        "crop management", "crop disease", "crop pest", "crop growth", "crop nutrition",
        "soil", "soil health", "soil fertility", "soil moisture", "soil pH",
        "soil erosion", "soil conservation", "soil management", "soil amendments", "soil testing",
        "irrigation", "water management", "drip irrigation", "sprinkler irrigation",
        "irrigation scheduling", "water efficiency", "water quality", "water conservation", "drainage",
        "pest", "pest control", "integrated pest management", "pesticide",
        "insecticide", "herbicide", "fungicide", "biological control", "pest monitoring", "pest detection",
        "weather", "climate", "temperature", "rainfall", "humidity",
        "wind", "frost", "heat stress", "drought", "flood",
        "season", "growing season", "planting season", "harvest season",
        "off-season", "crop cycle", "seasonal crops", "seasonal changes", "seasonal variations",
        "farming", "agriculture", "farm management", "sustainable agriculture",
        "organic farming", "precision agriculture", "agribusiness", "farm economics", "farm equipment", "farm technology",
        "livestock", "cattle", "dairy", "poultry",
        "swine", "sheep", "goat", "animal health", "animal nutrition", "livestock management"
    ]
    for keyword in agriculture_keywords:
        if keyword in query:
            return True
    return False

def get_agriculture_response(user_message, chat_history):
    if is_agriculture_related(user_message):
        # Provide agriculture-related response
        response = llm_chain.predict(user_message=user_message)
    else:
        response = "Sorry, I can only answer agriculture-related questions."
    return response

# Create chat interface
demo = gr.Interface(get_agriculture_response,
                    inputs=["text", "text"],
                    outputs="text",
                    title="Agriculture Chatbot",
                    description="Ask questions related to agriculture using keywords provided",
                    examples=[["What crops should I plant in sandy soil?", ""],
                              ["How can I prevent pests in my tomato garden?", ""],
                              ["What irrigation method is best for dry climates?", ""]])

if _name_ == "_main_":
    demo.launch()
