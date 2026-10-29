"""engram in 20 lines — run: python examples/quickstart.py"""
from engram import Brain

brain = Brain("quickstart")

# 1. feed it anything
brain.ingest("examples/sample_notes")

# 2. ask in natural language
for q in ["what did we decide about pricing?", "when is the deploy window?"]:
    print(f"\nQ: {q}")
    for h in brain.search(q, top_k=2):
        print(f"  [{h.score:.3f}] {h.source.split('/')[-1]}")
        print(f"         {h.text.strip()[:120]}")

# 3. it persists — a new Brain object sees the same index
again = Brain("quickstart")
print(f"\nindex holds {len(again)} chunks, {again.stats()['sources_count']} sources")
