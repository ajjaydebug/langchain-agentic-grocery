import pandas as pd
from langchain_community.llms import Ollama
from langchain.tools import tool

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("GroceryDataset.csv")

# Normalize text columns
df["Title"] = df["Title"].astype(str).str.lower()
df["Sub Category"] = df["Sub Category"].astype(str).str.lower()

# -----------------------------
# HELPER: EXTRACT KEYWORDS
# -----------------------------
STOP_WORDS = {
    "find", "show", "list", "what", "are", "is", "the",
    "in", "of", "for", "price", "cost", "tell", "me"
}

def extract_keywords(text: str):
    words = text.lower().split()
    return [w for w in words if w not in STOP_WORDS]


# -----------------------------
# TOOLS
# -----------------------------
@tool
def search_product(query: str) -> str:
    """Search products in the grocery dataset"""

    keywords = extract_keywords(query)

    if not keywords:
        return "Please mention a product or category."

    pattern = "|".join(keywords)

    matches = df[
        df["Title"].str.contains(pattern, na=False)
        | df["Sub Category"].str.contains(pattern, na=False)
    ]

    if matches.empty:
        categories = df["Sub Category"].value_counts().head(5).index.tolist()
        return (
            "No matching products found.\n"
            "Try categories like:\n"
            + ", ".join(categories)
        )

    results = []
    for _, row in matches.head(5).iterrows():
        results.append(
            f"Product: {row['Title']} | "
            f"Category: {row['Sub Category']} | "
            f"Price: {row['Price']} | "
            f"Rating: {row['Rating']}"
        )

    return "\n".join(results)


@tool
def get_price(query: str) -> str:
    """Get the price of a product"""

    keywords = extract_keywords(query)

    if not keywords:
        return "Please mention the product name."

    pattern = "|".join(keywords)
    matches = df[df["Title"].str.contains(pattern, na=False)]

    if matches.empty:
        return "Price not available for that product."

    row = matches.iloc[0]
    return f"The price of {row['Title']} is {row['Price']}."


# -----------------------------
# LOAD MODEL (LOW RAM SAFE)
# -----------------------------
llm = Ollama(
    model="tinyllama",
    temperature=0.2
)

# -----------------------------
# INTERACTIVE CHAT (CONTROLLER-BASED AGENT)
# -----------------------------
print("\n🛒 Grocery Store AI Assistant")
print("Ask about products, categories, or prices.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    # Controller logic (reliable for local models)
    if "price" in user_input.lower() or "cost" in user_input.lower():
        response = get_price.run(user_input)
    else:
        response = search_product.run(user_input)

    print(f"\nAgent: {response}\n")
