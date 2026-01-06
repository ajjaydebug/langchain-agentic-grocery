import re
import pandas as pd
from langchain_community.llms import Ollama
from langchain.tools import tool

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("GroceryDataset.csv")
df["Title"] = df["Title"].astype(str).str.lower()
df["Sub Category"] = df["Sub Category"].astype(str).str.lower()

# -----------------------------
# STOP WORDS
# -----------------------------
STOP_WORDS = {
    "find", "show", "list", "what", "are", "is", "the",
    "in", "of", "for", "tell", "me", "than", "cheaper"
}


def extract_keywords(text: str):
    tokens = re.findall(r"\w+", text.lower())
    return [w for w in tokens if w not in STOP_WORDS]


# -----------------------------
# HELPER: STRICT PRODUCT MATCH
# -----------------------------
def find_product_price(product):
    product = product.lower()

    matches = df[
        df["Title"].str.contains(rf"\b{re.escape(product)}\b", regex=True, na=False)
    ]

    if matches.empty:
        matches = df[
            df["Sub Category"].str.contains(rf"\b{re.escape(product)}\b", regex=True, na=False)
        ]

    if matches.empty:
        return None

    return matches.iloc[0]["Price"]


# -----------------------------
# TOOLS
# -----------------------------
@tool
def search_product(query: str) -> str:
    """
    Search products by name or category.
    """
    keywords = extract_keywords(query)
    if not keywords:
        return "NO_RESULT"

    pattern = "|".join(re.escape(k) for k in keywords)
    matches = df[
        df["Title"].str.contains(pattern, na=False, regex=True)
        | df["Sub Category"].str.contains(pattern, na=False, regex=True)
    ]

    if matches.empty:
        return "NO_RESULT"

    row = matches.iloc[0]
    return f"{row['Title']} | ₹{row['Price']} | {row['Sub Category']}"


@tool
def get_price(query: str) -> str:
    """
    Get price of a product.
    """
    keywords = extract_keywords(query)
    if not keywords:
        return "NO_RESULT"

    pattern = "|".join(re.escape(k) for k in keywords)
    matches = df[df["Title"].str.contains(pattern, na=False, regex=True)]

    if matches.empty:
        return "NO_RESULT"

    row = matches.iloc[0]
    return f"The price of {row['Title']} is ₹{row['Price']}."


@tool
def compare_prices(query: str) -> str:
    """
    Compare prices of two products and say which is cheaper.
    """
    words = extract_keywords(query)

    if len(words) < 2:
        return "NO_RESULT"

    product1, product2 = words[0], words[1]

    price1 = find_product_price(product1)
    price2 = find_product_price(product2)

    if price1 is None or price2 is None:
        return "Comparison not possible."

    if price1 < price2:
        return f"{product1} is cheaper than {product2}."
    elif price1 > price2:
        return f"{product2} is cheaper than {product1}."
    else:
        return f"{product1} and {product2} cost the same."


# -----------------------------
# LOAD LLM
# -----------------------------
llm = Ollama(model="tinyllama", temperature=0.2)


# -----------------------------
# MEMORY
# -----------------------------
agent_memory = {
    "observations": [],
    "failed_actions": set()
}

conversation_memory = []


# -----------------------------
# AGENT PLANNER
# -----------------------------
def agent_plan(user_input, agent_memory, conversation_memory):
    prompt = f"""
You are an AI agent.

Conversation history:
{conversation_memory}

Current user request:
"{user_input}"

Agent memory (current goal):
Observations: {agent_memory['observations']}
Failed actions: {list(agent_memory['failed_actions'])}

Choose next action:
- USE_COMPARE_TOOL
- USE_PRICE_TOOL
- USE_SEARCH_TOOL
- FINISH

Reply with only one option.
"""
    raw = llm.invoke(prompt)
    raw_text = raw.strip().lower() if isinstance(raw, str) else str(raw).lower()

    # Use conservative, word-boundary checks
    if re.search(r"\buse_compare_tool\b", raw_text) or re.search(r"\bcompare\b", raw_text):
        return "USE_COMPARE_TOOL"
    if re.search(r"\buse_price_tool\b", raw_text) or re.search(r"\bprice\b", raw_text):
        return "USE_PRICE_TOOL"
    if re.search(r"\buse_search_tool\b", raw_text) or re.search(r"\bsearch\b", raw_text) or re.search(r"\bfind\b", raw_text):
        return "USE_SEARCH_TOOL"
    if re.search(r"\bfinish\b", raw_text) or re.search(r"\bdone\b", raw_text) or re.search(r"\bno action\b", raw_text):
        return "FINISH"

    # Fallback to search to maximize chance of returning something useful
    return "USE_SEARCH_TOOL"


# -----------------------------
# AGENT LOOP
# -----------------------------
print("\n🤖 Agentic Grocery AI (Comparison Enabled)")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    agent_memory["observations"].clear()
    agent_memory["failed_actions"].clear()

    for _ in range(3):
        action = agent_plan(user_input, agent_memory, conversation_memory)

        # If agent wants to finish but nothing has been tried yet, force a search
        if action == "FINISH":
            if not agent_memory["observations"] and not agent_memory["failed_actions"]:
                action = "USE_SEARCH_TOOL"
            else:
                break
        
        if action == "USE_COMPARE_TOOL":
            result = compare_prices.run(user_input)

        elif action == "USE_PRICE_TOOL":
            result = get_price.run(user_input)

        elif action == "USE_SEARCH_TOOL":
            result = search_product.run(user_input)

        else:
            break

        agent_memory["observations"].append(result)

        if result == "NO_RESULT":
            agent_memory["failed_actions"].add(action)
        else:
            break

    if agent_memory["observations"]:
        last = agent_memory["observations"][-1]
        final_answer = last if last != "NO_RESULT" else "Sorry, I couldn't find an answer."
    else:
        final_answer = "Sorry, I couldn't find an answer."

    conversation_memory.append(f"User: {user_input}")
    conversation_memory.append(f"Agent: {final_answer}")

    print(f"\nAgent: {final_answer}\n")
        
