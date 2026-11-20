import pandas as pd
import tiktoken

encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
df = pd.read_csv("chunked_wiki_txt.csv")

df["tokens"] = df["passage"].apply(lambda x: len(encoding.encode(str(x))))

total_tokens = df["tokens"].sum()
average_tokens = df["tokens"].mean()

print(f"Gesamt: {total_tokens:,} Tokens")
print(f"Durchschnitt: {average_tokens:.1f} Tokens/Satz")
print(f"Minimum: {df['tokens'].min()}")
print(f"Maximum: {df['tokens'].max()}")
