import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
os.environ["ANONYMIZED_TELEMETRY"] = "false"

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY') 
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

# Configure browser settings
browser_config = BrowserConfig(
    headless=False,  # Run with UI for debugging
    disable_security=True,  # Allow certain browser behaviors
    extra_chromium_args=["--start-maximized"],  # Maximize window
)

# Configure browser context settings
context_config = BrowserContextConfig(
    wait_for_network_idle_page_load_time=3.0,  # Ensure pages fully load before proceeding
    browser_window_size={"width": 1280, "height": 1100},
    locale="en-US",
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36",
    highlight_elements=True,  # Enable visual debugging
)


def get_next_task_folder():
    base_path = os.path.join('extracted_data')
    # Create extracted_data folder if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
 
    counter = 1
    while True:
        folder_name = f"workflow{counter}"
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path
        counter += 1
        

async def run_search():
    browser = Browser(config=browser_config)
    context = BrowserContext(browser=browser, config=context_config)

    try:
        agent = Agent(
            task=(
                """Find three apartments in New York City priced under $300,000. 
                Each apartment must have at least two bedrooms and a total area of at least 120 square meters. 
                Return the results in a structured JSON format, including the following details for each apartment:
                - Price
                - Location1
                - Number of bedrooms
                - Total area
                - Any additional relevant features."""
            ),
            llm=llm,
            max_actions_per_step=25,
            browser_context=context,
        )

        # Run the agent and capture the output
        output = await agent.run(max_steps=50)
        
        # Convert the AgentHistoryList to a string representation
        output_str = str(output)

        # Get the next available task folder
        task_folder = get_next_task_folder()
        
        # Create output.txt in the task folder
        output_file = os.path.join(task_folder, 'output.txt')
        
        # Write the string output to a text file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(output_str)
    
    finally:
        # Properly close the browser context
        await context.close()
        await browser.close()
        
if __name__ == '__main__':
    asyncio.run(run_search())