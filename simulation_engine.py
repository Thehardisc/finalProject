import asyncio
import json
import uuid
import time
import os
import random

try:
    import redis.asyncio as aioredis
except ImportError:
    print("Please install redis: pip install redis")
    exit(1)

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# We simulate a conversation between these two IDs
USER_A_ID = "user_sim_A"
USER_B_ID = "user_sim_B"
CONVERSATION_ID = "sim_chat_001"

# Predefined fallback scenarios if no OpenAI API key is present
MOCK_SCENARIOS = [
    [
        (USER_A_ID, "Hey, are you free to talk?"),
        (USER_B_ID, "Yeah what's up?"),
        (USER_A_ID, "I'm just really frustrated with work today... everything went wrong."),
        (USER_B_ID, "Oh no! 😔 What happened?"),
        (USER_A_ID, "The server crashed and I lost all my progress. I'm so angry right now! 😡"),
        (USER_B_ID, "That is awful. Take a deep breath. Can you recover it?"),
        (USER_A_ID, "No, it's gone. I'm just going to sleep, I can't deal with this tonight."),
        (USER_B_ID, "Alright, get some rest. We'll figure it out tomorrow. ❤️"),
        (USER_A_ID, "Goodnight.")
    ],
    [
        (USER_B_ID, "Morning! Did you see the news?! ✨"),
        (USER_A_ID, "No, what happened?"),
        (USER_B_ID, "We got the funding! The project is approved! 🎉😍"),
        (USER_A_ID, "OMG REALLY?! That is amazing!!"),
        (USER_B_ID, "I know!! I'm so excited!"),
        (USER_A_ID, "We should celebrate tonight! Drinks on me 🍻")
    ]
]

async def generate_llm_chat(scenario_prompt: str, num_messages: int = 8) -> list:
    """Uses OpenAI to generate a synthetic chat if API key is available."""
    if not HAS_OPENAI or not OPENAI_API_KEY:
        print("No OpenAI API key found or package missing. Using Mock Data.")
        return random.choice(MOCK_SCENARIOS)
    
    print(f"Generating LLM chat for scenario: {scenario_prompt}")
    openai.api_key = OPENAI_API_KEY
    
    system_prompt = f"""
    You are simulating a WhatsApp chat between User_A and User_B.
    Scenario: {scenario_prompt}
    Length: Exactly {num_messages} messages.
    Format your response EXACTLY as a JSON array of arrays, like this:
    [
      ["user_sim_A", "message text here"],
      ["user_sim_B", "response here..."]
    ]
    Include appropriate emojis and emotional escalation/de-escalation.
    DO NOT output markdown formatting like ```json, ONLY output the raw valid JSON array.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.8
        )
        content = response.choices[0].message.content.strip()
        chat_data = json.loads(content)
        return chat_data
    except Exception as e:
        print(f"LLM Generation failed: {e}. Falling back to mock data.")
        return random.choice(MOCK_SCENARIOS)

async def run_simulation(scenario: str = "A late night argument that ends in reconciliation."):
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    print(f"Connected to Redis at {REDIS_URL}")
    
    chat_sequence = await generate_llm_chat(scenario)
    
    print(f"Starting simulation of {len(chat_sequence)} messages...")
    
    # We will simulate the messages coming in, but we will artificially manipulate the timestamps
    # to test the Context Engine's volatility and time calculations.
    # We'll start the conversation 1 hour ago.
    current_simulated_time = time.time() - 3600 
    
    for actor, text in chat_sequence:
        # Add latency (random between 5 seconds and 3 minutes)
        latency = random.randint(5, 180)
        current_simulated_time += latency
        
        event = {
            "text": text,
            "conversation_id": CONVERSATION_ID,
            "user_id": actor,
            "timestamp": current_simulated_time,
            "message_id": str(uuid.uuid4()),
            "metadata": {"source": "simulation_engine"}
        }
        
        # Inject to Redis exactly how the API Gateway does it
        await redis.xadd("message_stream", {"payload": json.dumps(event)})
        
        print(f"[{time.strftime('%H:%M:%S', time.localtime(current_simulated_time))}] {actor}: {text}")
        
        # Real-world pause so you can see it processing in the logs (optional)
        await asyncio.sleep(2)
        
    print("Simulation complete! Messages injected to message_stream.")
    print("The system is now computing 115-dim vectors and clusters for these messages.")
    
    await redis.close()

if __name__ == "__main__":
    print("--- INNERLINK SIMULATION ENGINE ---")
    scenarios = [
        "A late night argument about washing the dishes.",
        "Excited planning for a surprise birthday party.",
        "Technical support chat where the customer is very annoyed.",
        "Deep emotional conversation about feeling lost in life."
    ]
    
    selected_scenario = random.choice(scenarios)
    asyncio.run(run_simulation(selected_scenario))
