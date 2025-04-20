import os
import ast

from dotenv import load_dotenv, find_dotenv
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

load_dotenv(find_dotenv())

# Gemini client
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))


class SemanticChunker:
    def __init__(
        self,
        model: str = "gemini-embedding-exp-03-07",
        cluster_threshold: float = 0.7,
        max_lines: int = 200,
        max_tokens: int = 512,
    ):
        self.model = model
        self.cluster_threshold = cluster_threshold
        self.max_lines = max_lines
        self.max_tokens = max_tokens

    def split_code(self, source: str) -> List[str]:
        """
        1. Extract top-level func/classes via AST
        2. Bundle module-level code as a chunk
        3. If any chunk > max_lines, slice into line windows
        """
        lines = source.splitlines(keepends=True)
        used = set()
        chunks: List[str] = []

        # Extract functions & classes
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                snippet = "".join(lines[node.lineno - 1 : node.end_lineno])
                chunks.append(snippet)
                used.update(range(node.lineno - 1, node.end_lineno))

        # Module-level remainder
        module_snip = "".join(lines[i] for i in range(len(lines)) if i not in used)
        if module_snip.strip():
            chunks.insert(0, module_snip)
        
        # Enfore max_lines
        final: List[str] = []
        for chunk in chunks:
            chunk_lines = chunk.splitlines(keepends=True)
            if len(chunk_lines) <= self.max_lines:
                final.append(chunk)
            else:
                for i in range(0, len(chunk_lines), self.max_lines):
                    final.append("".join(chunk_lines[i : i + self.max_lines]))
        
        return final
    
    def embed(self, chunks: List[str]) -> List[List[float]]:
        """
        Batch up to 250 chunks per request to Gemini
        """
        all_vecs: List[List[float]] = []
        for i in range(0, len(chunks), 250):
            batch = chunks[i : i + 250]
            resp = _client.models.embed_content(
                model=self.model,
                contents=batch
            )
            all_vecs.append(resp.embeddings)
        
        return all_vecs

    def cluster(self, embeddings: List[List[float]]) -> List[int]:
        """
        Compute cosine similarities and union-find cluster
        """
        sim = cosine_similarity(embeddings)
        n = len(sim)
        parent = list(range(n))

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            
            return u
    
        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pv] = pu
        
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= self.cluster_threshold:
                    union(i, j)
        
        return [find(i) for i in range(n)]
    
    def _token_count(self, text: str) -> int:
        # naive whitespace tokenizer; swap in a real tokenizer if needed
        return len(text.split())
    
    def merge(self, chunks: List[str], clusters: List[int]) -> List[str]:
        """
        Concatenate per cluster, respecting max_tokens.
        """
        by_cluster: Dict[int, List[str]] = {}
        for idx, cid in enumerate(clusters):
            by_cluster.setdefault(cid, []).append(chunks[idx])

        merged: List[str] = []
        for cid in sorted(by_cluster):
            current = ""
            for piece in by_cluster[cid]:
                candidate = (current + "\n" + piece).strip()
                if self._token_count(candidate) > self.max_tokens:
                    if current:
                        merged.append(current)
                    current = piece
                else:
                    current = candidate
            if current:
                merged.append(current)
        
        return merged
    
    def chunk_and_embed(self, source: str) -> List[Dict[str, object]]:
        """
        Returns list of {"chunk": str, "embedding": List[float]}.
        """
        chunks = self.split_code(source)
        embeddings = self.embed(chunks)
        clusters = self.cluster(embeddings)
        merged = self.merge(chunks, clusters)
        # re‑embed merged chunks
        final_embs = self.embed(merged)
        
        return [{"chunk": c, "embedding": e} for c, e in zip(merged, final_embs)]