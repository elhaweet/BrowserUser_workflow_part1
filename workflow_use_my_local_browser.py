import asyncio
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')

# Initialize the language model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

# Define browser configuration
browser_config = BrowserConfig(
    chrome_instance_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe"
)

# Define browser context configuration
context_config = BrowserContextConfig(
    browser_window_size={'width': 1280, 'height': 1100},
    locale='en-US',
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    highlight_elements=True,
    # allowed_domains=['example.com'],  # Replace with actual domains as needed
)

# Initialize browser and context
browser = Browser(config=browser_config)
context = BrowserContext(browser=browser, config=context_config)

async def run_search():
    agent = Agent(
        task=(
            'I want you to find 5 apartments in New Cairo with 3 bedrooms and 2 bathrooms priced under 3 million EGP, posted in the last 3 months.'
        ),
        llm=llm,
        max_actions_per_step=4,
        browser_context=context,
    )
    await agent.run(max_steps=25)

if __name__ == '__main__':
    asyncio.run(run_search())
    
    

