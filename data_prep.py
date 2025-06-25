import pandas as pd

# Load your dataset
print("📥 Loading Reviews.csv...")
df = pd.read_csv("Reviews.csv")

# Drop rows with missing text
df = df[['Text', 'Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator']].dropna()
df['Text'] = df['Text'].astype(str).str.strip()
df = df[df['Text'] != ""]  # Remove empty strings

# Default Label: REAL
df['Label'] = 0

# Avoid division by zero for helpfulness ratio
df['HelpfulnessDenominator'] = df['HelpfulnessDenominator'].replace(0, 1)

# Label as FAKE if extreme rating + low helpfulness ratio
df.loc[
    ((df['Score'] == 1) | (df['Score'] == 5)) &
    ((df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']) < 0.3),
    'Label'
] = 1  # FAKE

# Show class distribution
print("\n📊 Label distribution before balancing:")
print(df['Label'].value_counts())

# Balance dataset (optional but recommended)
print("\n⚖️ Balancing dataset...")
real_reviews = df[df['Label'] == 0]
fake_reviews = df[df['Label'] == 1]
min_len = min(len(real_reviews), len(fake_reviews))

df_balanced = pd.concat([
    real_reviews.sample(min_len, random_state=42),
    fake_reviews.sample(min_len, random_state=42)
])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Save cleaned and balanced dataset
df_balanced[['Text', 'Label']].to_csv("cleaned_data.csv", index=False)
print(f"\n✅ Cleaned and balanced dataset saved as 'cleaned_data.csv' with {len(df_balanced)} rows total.")
