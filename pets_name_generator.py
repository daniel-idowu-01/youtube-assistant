import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

huggingface_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def generate_pet_name(animal_type, pet_color):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct",  # Free model
        model_kwargs={"temperature": 0.7, "max_new_tokens": 100},
        huggingfacehub_api_token=huggingface_key
    )

    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a {animal_type} pet and I want a cool name for it. It is {pet_color} in color. Suggest me 5 cool names for my pet."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='pet_name')
    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})
    
    return response

def langchain_agent():
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct",
        model_kwargs={"temperature": 0.7, "max_new_tokens": 100},
        huggingfacehub_api_token=huggingface_key
    )

    tools = load_tools(['wikipedia', 'llm-math'], llm=llm)

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    result = agent.run("What is the average age of a dog? Multiply the age by 3")

    return result

if __name__ == "__main__":
    print(generate_pet_name("dog", "black"))
