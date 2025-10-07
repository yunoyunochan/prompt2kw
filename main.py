#!/usr/bin/env python3
"""
Thai Keyword Extraction Agent
Usage: python version.py "your Thai text here"
"""

import argparse
from openai import AzureOpenAI
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
import dotenv;
import os;

dotenv.load_dotenv(dotenv.find_dotenv(),override=True);

@tool
def thai_prompt_to_keywords_tool(thai_prompt: str) -> list:
    """
    Extract keywords from Thai text prompts. Analyzes Thai text and returns important keywords,
    removing filler words and particles.
    
    Args:
        thai_prompt: The Thai text to extract keywords from
    
    Returns:
        A list of extracted keywords
    """
    from openai import AzureOpenAI
    
    system_prompt = """You are an expert at extracting keywords from Thai text prompts. 
Your task is to:
1. Analyze the Thai prompt carefully
2. Extract the most important keywords and concepts
3. Return keywords in Thai
4. Focus on: main topics, actions, objects, and key descriptive terms
5. Remove filler words, particles (อะ, หน่อย, ด้วย, etc.), and unnecessary connectors
6. Keep keywords concise and relevant

Format: Return ONLY the keywords separated by commas, nothing else."""
    
    try:
        # Use Azure OpenAI instead of Ollama
        client = AzureOpenAI(
            api_key=os.getenv("API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("azure_endpoint")
        )
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": thai_prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        keywords_text = response.choices[0].message.content.strip()
        keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
        return keywords
        
    except Exception as e:
        return [f"Error: {e}"]


def create_agent():
    """Create and return the agent executor"""
    
    # Azure OpenAI LLM
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("azure_endpoint"),
        api_key=os.getenv("API_KEY"),
        deployment_name="gpt-4.1",
        api_version="2024-02-15-preview",
        temperature=0.4
    )
    
    # System prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent Thai text assistant that automatically detects and processes Thai content.

**Your Role:**
Automatically analyze user input and extract keywords when Thai text is present.

**When to USE thai_prompt_to_keywords_tool (automatically):**
- User message contains Thai text/sentences (even without explicit request)
- Thai content looks like it needs analysis (documents, descriptions, queries)
- User provides Thai text in any context

**When NOT to use the tool:**
- Only greetings with no substantive Thai content (e.g., just "สวัสดี", "สวัสดีครับ")
- Pure questions about capabilities
- English-only messages
- Very short Thai phrases that are just pleasantries

**Response Format:**

1. **When you CALL the tool (keyword extraction):**
   - Return ONLY the keywords exactly as received
   - DO NOT add any text like "The key terms are:", "Keywords:", or explanations
   - Just return: "เอกสาร, การจัดซื้อ, สัญญา"

2. **When you DON'T call the tool (greetings, questions):**
   - Respond naturally and politely in Thai or English as appropriate
   - For Thai greetings, reply in Thai: "สวัสดีครับ! ผมสามารถช่วยวิเคราะห์คีย์เวิร์ดจากข้อความภาษาไทยได้ครับ"
   - For English, reply in English: "Hello! I can help extract keywords from Thai text."

**Examples:**
✅ CORRECT (extraction): "เอกสาร, การจัดซื้อ, สัญญา"
✅ CORRECT (greeting): "สวัสดีครับ! ผมสามารถช่วยวิเคราะห์คีย์เวิร์ดจากข้อความภาษาไทยได้ครับ"
❌ WRONG: "The key terms from your Thai text are: เอกสาร, การจัดซื้อ, สัญญา"

Be proactive with extraction, but friendly and conversational when tools aren't needed."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create agent
    tools = [thai_prompt_to_keywords_tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )
    
    return agent_executor


def main():
    parser = argparse.ArgumentParser(description='Extract keywords from Thai text using AI agent')
    parser.add_argument('--prompt', type=str, help='Thai text prompt to analyze')
    args = parser.parse_args()
    
    # Create agent
    agent_executor = create_agent()
    
    # Process input
    result = agent_executor.invoke({"input": args.prompt})
    
    # Output only the result
    print(result['output'])


if __name__ == "__main__":
    main()

# /bin/python3 /mnt/c/fusion/prompt2kw/version.py --prompt "มีข้อมูลอะไรบ้าง"
# /bin/python3 /mnt/c/fusion/prompt2kw/version.py --prompt 'ต้องการไฟล์สัญญาหน่อย และก็เอกสารทั้งหมดด้วย ที่ที่เกี่ยวข้องการซื้อขายกับผลิตภัณฑ์อะ ของบริษัท Biosidus อะ'


# # Unload a specific model
# curl http://localhost:11434/api/generate -d '{
#   "model": "llama2",
#   "keep_alive": 0
# }'