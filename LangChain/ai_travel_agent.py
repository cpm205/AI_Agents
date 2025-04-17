import os
import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import AzureOpenAI

# Set environment variables for Azure OpenAI configuration
os.environ['OPENAI_API_KEY'] = ""
os.environ['OPENAI_API_BASE'] = "https://devdemo.openai.azure.com/"

# Define a prompt template to gather user preferences
class GatherPreferences(PromptTemplate):
    def __init__(self):
        super().__init__(template="What are your preferences for hotels, activities, and restaurants in {city}?")

# Define a chain to handle the travel recommendation process
class TravelAgentChain(LLMChain):
    def __init__(self):
        llm = AzureOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            api_base=os.environ['OPENAI_API_BASE'],
            model_name="gpt-35-turbo-16k"
        )
        super().__init__(llm=llm, prompt=GatherPreferences())

    def run(self, city):
        # Run the chain with the specified city
        preferences = self.execute(city=city)
        # Use preferences to fetch and generate recommendations
        recommendations = self.generate_recommendations(preferences)
        return recommendations

    def generate_recommendations(self, preferences):
        # Define the prompt for the AI model
        prompt = f"Based on the following preferences: {preferences}, provide a list of recommended hotels, activities, restaurants, and events in the city."

        # Call the Azure OpenAI API
        response = openai.Completion.create(
            engine="gpt-35-turbo-16k",  # Specify the model engine
            prompt=prompt,
            max_tokens=150,  # Adjust the token limit as needed
            n=1,
            stop=None,
            temperature=0.7  # Adjust the creativity level
        )

        # Extract and return the recommendations
        recommendations = response.choices[0].text.strip()
        return recommendations

# Example usage
if __name__ == "__main__":
    travel_agent = TravelAgentChain()
    recommendations = travel_agent.run("New York")
    print(recommendations) 