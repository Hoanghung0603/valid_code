import pandas as pd
from urllib.parse import urlparse
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze dropped URLs that were verified by LLM.")
    parser.add_argument('--input', type=str, default='data/raw/corpus_text/dropped_urls_merged/checked_delete.parquet', help='Path to verified parquet file')
    args = parser.parse_args()

    # Load the verified dataset
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        print(f"Error loading {args.input}: {e}")
        return

    print("Total samples:", len(df))
    print("Columns:", df.columns.tolist())

    # We want to find URLs that the LLM disagreed with dropping, meaning they should be KEPT (they are medical).
    # Based on previous info, 'agreement' == 'Không đồng ý'
    if 'agreement' not in df.columns:
        print("Column 'agreement' not found in dataset. Ensure LLM outputs are properly parsed.")
        # Try to infer if 'llm_response' or similar exists
        for col in df.columns:
            if 'llm' in col.lower() or 'response' in col.lower():
                print(f"Sample of {col}:")
                print(df[col].head())
        return

    # Assuming 'Không đồng ý' means the heuristic was wrong, and the URL *is* medical.
    # We might need to check exact wording. Let's look at unique values of 'agreement'
    print("\nAgreement distribution:")
    print(df['agreement'].value_counts(dropna=False))

    keep_df = df[df['agreement'].str.contains('Không đồng ý', na=False, case=False)].copy()
    print(f"\nNumber of URLs to recover (LLM said 'Không đồng ý' with dropping): {len(keep_df)}")

    if len(keep_df) == 0:
        print("No URLs to recover found.")
        return

    # Extract domain
    keep_df['domain'] = keep_df['url'].apply(lambda x: urlparse(x).netloc)

    # Top domains
    print("\n--- Top Domains to Recover ---")
    domain_counts = keep_df['domain'].value_counts()
    print(domain_counts.head(30))

    # Path keywords analysis
    def extract_path_keywords(url):
        path = urlparse(url).path
        words = re.split(r'\W+', path.lower())
        return [w for w in words if len(w) > 2 and not w.isdigit()]

    all_keywords = []
    for url in keep_df['url']:
        all_keywords.extend(extract_path_keywords(url))

    print("\n--- Top Path Keywords in Recovered URLs ---")
    keyword_counts = pd.Series(all_keywords).value_counts()
    print(keyword_counts.head(30))

    # Find common patterns in the URL
    print("\n--- Potential Regex Patterns for Whitelisting ---")
    # For domains that appear frequently (e.g., > 10 times)
    frequent_domains = domain_counts[domain_counts >= 10].index.tolist()
    if frequent_domains:
        print("Whitelisted domains pattern:")
        escaped_domains = [d.replace('.', '\\.') for d in frequent_domains]
        print(f"r'https?://(?:www\\.)?(?:{'|'.join(escaped_domains)})/'")
    
    # For common medical keywords in path
    medical_keywords = ['thuoc', 'benh', 'vien', 'duoc', 'ykhoa', 'suckhoe', 'kham', 'chua', 'dieutri', 'y-te', 'bac-si']
    found_medical = [kw for kw in medical_keywords if kw in keyword_counts and keyword_counts[kw] > 5]
    if found_medical:
        print("Whitelisted path keywords pattern:")
        print(f"r'/(?:.*-)?(?:{'|'.join(found_medical)})(?:-.*)?/'")


if __name__ == "__main__":
    main()
