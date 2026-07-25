
import pandas as pd
import os

INPUT_FILE = "data/telegram_posts.csv"
OUTPUT_FILE = "data/telegram_posts_labelled.csv"

df = pd.read_csv(INPUT_FILE)

# Only show unlabelled posts
if os.path.exists(OUTPUT_FILE):
    labelled_df = pd.read_csv(OUTPUT_FILE)
    labelled_ids = set(zip(labelled_df["channel"].astype(str), labelled_df["message_id"].astype(str)))
    df["_key"] = list(zip(df["channel"].astype(str), df["message_id"].astype(str)))
    df = df[~df["_key"].apply(lambda x: x in labelled_ids)].drop(columns=["_key"])
    print(f"Unlabelled posts available: {len(df)}")
else:
    labelled_df = pd.DataFrame(columns=list(df.columns) + ["label"])
    labelled_ids = set()

print("Labels: 0 = Genuine | 1 = Promotional/Hype | 2 = Suspicious/Pump-Dump")
print("LABELLING GUIDE:")
print("0 = Factual/Neutral - market data, no buy/sell call")
print("1 = Open Trading Tip - public call WITH target+stoploss shown")
print("2 = Suspicious - private group invite links, guaranteed returns, no stop-loss")
print("\nType 's' to skip, 'q' to quit and save\n")

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

    # Auto-save after every label
    new_df = pd.DataFrame(new_labels)
    save_df = pd.concat([labelled_df, new_df], ignore_index=True)
    save_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  [saved {len(save_df)} total]")

print(f"\nDone. Total labelled: {len(labelled_df) + len(new_labels)}")