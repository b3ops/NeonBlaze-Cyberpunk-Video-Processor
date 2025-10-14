import argparse
import csv
import os
import re
from collections import OrderedDict

def parse_quote(line):
    """Parse a line into quote, author, source. Expects: 'Quote text - Author (Source)' or 'Quote text - Author'."""
    # Match: "Quote text - Author (Source)" or "Quote text - Author"
    match = re.match(r'^(.*?)\s*-\s*([^()]+?)(?:\s*\((.+)\))?$', line.strip())
    if match:
        quote, author, source = match.groups()
        quote = quote.strip().strip('"')
        author = author.strip()
        source = source.strip() if source else "Unknown"
        return quote, author, source
    else:
        # Fallback: treat whole line as quote, no author/source
        return line.strip(), "Unknown", "Unknown"

def convert_to_csv(input_file, output_file):
    """Convert quotes text file to CSV, removing duplicates."""
    # Resolve input file path
    input_path = os.path.expanduser(input_file)
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found. Check path or file existence.")
        return

    # Read all lines, strip empty
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading '{input_path}': {e}")
        return

    # Parse quotes and deduplicate using OrderedDict to preserve order
    quote_dict = OrderedDict()
    for line in lines:
        quote, author, source = parse_quote(line)
        # Key by (quote, author, source) to catch exact duplicates
        quote_dict[(quote, author, source)] = None

    # Prepare CSV
    headers = ['quote', 'author', 'source']
    output_path = os.path.expanduser(output_file)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for quote, author, source in quote_dict.keys():
                writer.writerow([quote, author, source])
        print(f"Converted {len(quote_dict)} unique quotes from '{input_path}' to '{output_path}'")
    except Exception as e:
        print(f"Error writing '{output_path}': {e}")
        return

    print(f"CSV saved: {output_path}. Preview with: head {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert quotes text file to CSV dataset")
    parser.add_argument('--input_file', default='~/progs/cyberpunk_quotes.txt', 
                        help="Input text file with quotes (default: ~/progs/cyberpunk_quotes.txt)")
    parser.add_argument('--output_file', default='~/progs/cyberpunk_quotes.csv', 
                        help="Output CSV file (default: ~/progs/cyberpunk_quotes.csv)")
    args = parser.parse_args()

    convert_to_csv(args.input_file, args.output_file)