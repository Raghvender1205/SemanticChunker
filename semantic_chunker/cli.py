import argparse
import json
import sys
from semantic_chunker.chunker import SemanticChunker

def main():
    parser = argparse.ArgumentParser(
        prog="chunker",
        description="Semantic code chunking with Gemini embeddings"
    )
    parser.add_argument("input", help="Path to your source code file")
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.7,
        help="Similarity threshold for clustering (0–1)"
    )
    parser.add_argument(
        "--max-lines", "-l", type=int, default=200,
        help="Max lines per initial AST chunk"
    )
    parser.add_argument(
        "--max-tokens", "-k", type=int, default=512,
        help="Max tokens per merged chunk"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON path (default: stdout)"
    )
    args = parser.parse_args()

    try:
        source = open(args.input, encoding="utf-8").read()
    except Exception as e:
        print(f"Error reading {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    chunker = SemanticChunker(
        cluster_threshold=args.threshold,
        max_lines=args.max_lines,
        max_tokens=args.max_tokens,
    )
    results = chunker.chunk_and_embed(source)
    out_json = json.dumps(results, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_json)
    else:
        print(out_json)

if __name__ == "__main__":
    main()
