import pandas as pd
import os

INPUT_FILE = "data/telegram_posts.csv"
OUTPUT_FILE = "data/telegram_posts_labelled.csv"

df = pd.read_csv(INPUT_FILE)

# Resume from existing labels if present
if os.path.exists(OUTPUT_FILE):
    labelled_df = pd.read_csv(OUTPUT_FILE)
    labelled_ids = set(labelled_df["message_id"].astype(str) + labelled_df["channel"])
else:
    labelled_df = pd.DataFrame(columns=list(df.columns) + ["label"])
    labelled_ids = set()

print("LABELLING GUIDE:")
print("0 = Factual/Neutral - market data, no buy/sell call (e.g. 'NIFTY data positive')")
print("1 = Open Trading Tip - public call WITH target+stoploss shown (e.g. 'Buy LUPIN 2145, Target 2170, SL 2121')")
print("2 = Suspicious - private group invite links, guaranteed returns, no stop-loss, urgency+secrecy")
print()
print("Type 's' to skip, 'q' to quit and save\n")

new_labels = []

for idx, row in df.iterrows():
    key = str(row["message_id"]) + row["channel"]
    if key in labelled_ids:
        continue

    print("-" * 60)
    print(f"[{row['channel']}]")
    print(row["text"][:400])
    print("-" * 60)

    label = input("Label (0/1/2, s=skip, q=quit): ").strip()

    if label == "q":
        break
    if label == "s":
        continue
    if label not in ["0", "1", "2"]:
        print("Invalid input, skipping this one.")
        continue

    row_dict = row.to_dict()
    row_dict["label"] = int(label)
    new_labels.append(row_dict)

if new_labels:
    new_df = pd.DataFrame(new_labels)
    combined = pd.concat([labelled_df, new_df], ignore_index=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(new_labels)} new labels. Total labelled: {len(combined)}")
else:
    print("\nNo new labels added.")
