# Duplicate and Near-Duplicate Comment Detection

## Why this matters

Regulatory dockets routinely receive thousands of comments generated from
campaign templates — organizations distribute a form letter that commenters
send verbatim or with minor personalization. For the RISE docket (~17,730
comments) treating each of those as an independent signal inflates perceived
support or opposition and obscures the actual distribution of unique viewpoints.
The goal of `assign_duplicate_clusters` is to group comments so downstream analysis can count campaigns rather than raw volume.

---

## Two-pass strategy

Detection runs in two ordered passes over the cleaned text.

### Pass 1: Exact matching

Before any machine-learning, an inexpensive signature-based pass handles
truly identical comments. Each comment is:

1. HTML-stripped and boilerplate-removed via `clean_comment_text`
2. Lowercased and whitespace-collapsed to a canonical key

Comments that share a canonical key are immediately linked.  This
handles the bulk of template duplicates simply and efficiently.

### Pass 2: Fuzzy matching with blocking

Comments that differ slightly — a personalized opener, a name at the bottom,
one substituted word — won't share an exact key but still indicate a form letter or template. Detecting them requires similarity scoring, which is expensive if done naively across all pairs.

**Blocking** limits comparisons to plausible candidates.  
Comments are bucketed by `len(text) // length_bucket_chars` (default 80).  A 400-character comment is only ever compared to other 320–480-character comments.  Template variants tend
to be nearly the same length, so this heuristic captures almost all real pairs
while discarding the vast majority of cross-campaign comparisons.

Inside each block:

- A `TfidfVectorizer` with unigrams and bigrams (`ngram_range=(1,2)`)
  converts texts to sparse L2-normalised vectors.
- `linear_kernel` (a plain dot product) scores every pair.  Because the vectors
  are already L2-normalised, the dot product equals cosine similarity — without
  the redundant re-normalisation step that `cosine_similarity` performs.
- `np.argwhere(np.triu(sim, k=1) >= threshold)` extracts pairs above the
  similarity threshold in one vectorised NumPy call.

Pairs above the threshold are fed into Pass 1's same union structure, so exact
and fuzzy results merge into a single consistent clustering.

---

## Union-Find (Disjoint Set Union)

Both passes produce *pairs*: "comment A and comment B are duplicates."  The
challenge is efficiently collapsing those pairs into *clusters* — groups of
arbitrary size — including transitive links (if A≡B and B≡C, all three belong
together even if we never directly compared A and C).

Union-Find is the standard data structure for this.  It maintains a forest of
trees where every node eventually points to the root of its component, and two
nodes are in the same cluster if and only if they share a root.

### Structure

```
_parent[i]  — the parent of node i (root when _parent[i] == i)
_rank[i]    — approximate tree depth, used to keep trees shallow
```

### `find(x)` — path compression

```
find(x):
    if _parent[x] != x:
        _parent[x] = find(_parent[x])   # collapse path to root
    return _parent[x]
```

The first call walks up the tree to find the root.  On the way back it
*flattens* every node it visited directly to the root.  Subsequent `find` calls
on those nodes cost O(1).

### `union(a, b)` — union by rank

```
union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb: return              # already same cluster
    if rank[ra] < rank[rb]: swap
    _parent[rb] = ra                 # attach shorter tree under taller
    if rank[ra] == rank[rb]: rank[ra] += 1
```

Attaching the smaller (shallower) tree under the larger one keeps total tree
height bounded at O(log n), ensuring `find` stays fast even without path
compression.  Together, path compression and union by rank give effectively
O(1) amortised cost per operation (technically inverse-Ackermann, which is ≤5
for any realistic n).

### Why not a graph / connected-components library?

For this use case Union-Find has two advantages:

1. **Online**: pairs can be added one at a time as we scan blocks; no need to
   materialise the full edge list before computing components.
2. **Minimal overhead**: the entire structure is two Python lists; no graph
   object, no adjacency storage.

### Final cluster assignment

After all unions are applied:

```python
roots = [find(i) for i in range(n)]
root_to_cluster = {r: k for k, r in enumerate(sorted(set(roots)))}
cluster_ids = [root_to_cluster[r] for r in roots]
```

This maps the arbitrary root indices to dense sequential integers (0, 1, 2 …)
so `cluster_id` in the output DataFrame is human-readable and easy to
`group_by`.

---

## Tuning

| Parameter | Default | Effect |
|---|---|---|
| `similarity_threshold` | 0.92 | Higher → stricter; lower → more aggressive merging. Start at 0.90–0.95 and calibrate on a sample. |
| `length_bucket_chars` | 80 | Smaller buckets → fewer cross-length comparisons; may miss near-dupes of different lengths. |
| `max_features` | 4096 | TF-IDF vocabulary cap per block. Increase for very long comments; decrease to speed up large blocks. |

Thresholds are heuristic.  A useful calibration workflow: run clustering, then
sample 20–30 pairs near the threshold boundary and inspect them manually.
