import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set, Any

# =========================================================
# Normalization controls
# =========================================================

LOWERCASE_ENTITIES = True  # set True if your KB is consistently lowercase

# =========================================================
# Normalization helpers
# =========================================================

_PUNCT_MAP = {
    "\u00A0": " ",  # NBSP -> space
    "\u2019": "'",  # right single quote -> '
    "\u2018": "'",  # left single quote  -> '
    "\u201C": '"',  # left double quote  -> "
    "\u201D": '"',  # right double quote -> "
    "\u2013": "-",  # en dash -> -
    "\u2014": "-",  # em dash -> -
    "\u2212": "-",  # minus sign -> -
    "\u00B4": "'",  # acute -> '
    "\u0060": "'",  # grave -> '
    "\u02BC": "'",  # modifier letter apostrophe -> '
}

_PUNCT_TRANS = str.maketrans(_PUNCT_MAP)

_WS_RE = re.compile(r"\s+")

def normalize_entity(s: str) -> str:
    """
    Normalize entity strings to avoid accidental splits:
      - unify a few common Unicode punctuation variants
      - strip leading/trailing whitespace
      - collapse internal whitespace
      - optional lowercase
    """
    if s is None:
        return s
    s = s.translate(_PUNCT_TRANS)
    s = s.strip()
    s = _WS_RE.sub(" ", s)
    if LOWERCASE_ENTITIES:
        s = s.lower()
    return s

def normalize_relation(r: str) -> str:
    """
    Relations in MetaQA typically contain underscores and no spaces.
    We only trim and unify NBSP/punct just in case, but DO NOT lowercase by default.
    """
    if r is None:
        return r
    r = r.translate(_PUNCT_TRANS).strip()
    r = _WS_RE.sub(" ", r)
    return r

def reverse_relation_name(rel: str) -> str:
    return f"{rel}_reversed"

# =========================================================
# Utilities
# =========================================================

def extract_source_entity(question: str) -> str:
    m = re.search(r"\[(.*?)\]", question)
    if not m:
        raise ValueError(f"No [source] entity found in question: {question}")
    return normalize_entity(m.group(1))

def split_answers(ans_raw: str) -> List[str]:
    parts = [a.strip() for a in ans_raw.split("|") if a.strip()]
    return [normalize_entity(a) for a in parts]

# =========================================================
# Base Knowledge Graph (for traversal; includes reversed)
# =========================================================

class KnowledgeGraph:
    """
    Directed multigraph via adjacency[(head, relation)] = set(tails).
    Reversed edges are auto-added for each KB triple so traversal can go both ways.
    """
    def __init__(self):
        self.adjacency: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()

        # For duplicate diagnostics:
        self._raw_to_norm_entities: Dict[str, str] = {}
        self._norm_to_raws: Dict[str, Set[str]] = defaultdict(set)

    def _touch_entity(self, raw: str) -> str:
        norm = normalize_entity(raw)
        self._raw_to_norm_entities[raw] = norm
        self._norm_to_raws[norm].add(raw)
        self.entities.add(norm)
        return norm

    def add_edge(self, head_raw: str, relation_raw: str, tail_raw: str):
        h = self._touch_entity(head_raw)
        t = self._touch_entity(tail_raw)
        r = normalize_relation(relation_raw)
        self.adjacency[(h, r)].add(t)
        self.relations.add(r)

    def load_from_file(self, kb_path: str, sep: str = "|"):
        with open(kb_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    head, rel, tail = [x for x in line.split(sep)]
                except ValueError:
                    raise ValueError(f"Bad KB line (expect head{sep}rel{sep}tail): {line}")

                # forward
                self.add_edge(head, rel, tail)
                # reversed for traversal
                self.add_edge(tail, reverse_relation_name(normalize_relation(rel)), head)

    def neighbors(self, head: str, relation: str) -> Set[str]:
        return self.adjacency.get((head, relation), set())

    def edge_count(self) -> int:
        return sum(len(tails) for tails in self.adjacency.values())

    # ---- diagnostics ----
    def suspicious_duplicates(self, top_k: int = 20) -> List[Tuple[str, List[str]]]:
        """
        Return up to top_k normalized forms with >1 distinct raw spellings.
        """
        suspects = [(norm, sorted(list(raws)))
                    for norm, raws in self._norm_to_raws.items() if len(raws) > 1]
        # sort by number of variants desc
        suspects.sort(key=lambda x: len(x[1]), reverse=True)
        return suspects[:top_k]

# =========================================================
# Pair → Relation mapping and qtype resolution
# =========================================================

def build_pair_to_relation() -> Dict[Tuple[str, str], str]:
    """
    Map (node_type_from, node_type_to) → concrete KG relation.
    Use *_reversed for backwards hops (e.g., director → movie).
    Extend as needed for your dataset.
    """
    return {
        # Movie ↔ Director
        ("movie",    "director"): "directed_by",
        ("director", "movie")   : "directed_by_reversed",

        # Movie ↔ Writer
        ("movie",  "writer"): "written_by",
        ("writer", "movie") : "written_by_reversed",

        # Movie ↔ Actor
        ("movie", "actor") : "starred_actors",
        ("actor", "movie") : "starred_actors_reversed",

        # Movie ↔ Release year
        ("movie", "year") : "release_year",
        ("year",  "movie"): "release_year_reversed",

        # Movie ↔ Language
        ("movie",    "language"): "in_language",
        ("language", "movie")   : "in_language_reversed",

        # Movie ↔ Genre
        ("movie", "genre") : "has_genre",
        ("genre", "movie") : "has_genre_reversed",
    }

def relseq_from_qtype(qtype: str, pair_to_relation: Dict[Tuple[str, str], str]) -> List[str]:
    """
    qtype example: 'director_to_movie_to_writer'
      tokens: ['director','movie','writer']
      pairs : ('director','movie'), ('movie','writer')
      rels  : ['directed_by_reversed','written_by']
    """
    tokens = qtype.strip().lower().split("_to_")
    if len(tokens) < 2:
        raise ValueError(f"QType must contain at least one '_to_': {qtype}")

    relseq: List[str] = []
    for i in range(len(tokens) - 1):
        key = (tokens[i], tokens[i + 1])
        if key not in pair_to_relation:
            known = ", ".join([f"{a}_to_{b}" for (a, b) in sorted(pair_to_relation)])
            raise KeyError(
                f"No mapping for pair '{tokens[i]}_to_{tokens[i+1]}' in qtype '{qtype}'. "
                f"Add it to build_pair_to_relation(). Known pairs: {known}"
            )
        relseq.append(pair_to_relation[key])
    return relseq

# =========================================================
# QA & QTYPE parsing
# =========================================================

def load_qa_file(qa_path: str) -> List[Dict[str, Any]]:
    """
    qa_*.txt lines: '<question>\t<answer or answer|answer|...>'
    Returns list of dicts with question, source, answers (list[str]).
    """
    items = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                q, a = line.split("\t")
            except ValueError:
                raise ValueError(f"{qa_path}: line {i} malformed (expect question<TAB>answer): {line}")

            src = extract_source_entity(q)
            answers = split_answers(a)
            items.append({"question": q, "source": src, "answers": answers})
    return items

def load_qtype_file(qtype_path: str) -> List[str]:
    """Return one qtype string per line (keeps alignment with QA items)."""
    qtypes = []
    with open(qtype_path, "r", encoding="utf-8") as f:
        for line in f:
            qtypes.append(line.strip())
    return qtypes

# =========================================================
# Evidence path search (two-hop)
# =========================================================

def find_two_hop_path(
    kg: KnowledgeGraph,
    source: str,
    answer: str,
    rel_seq: List[str],
) -> Optional[List[str]]:
    """
    Return [source, rel1, mid, rel2, answer] if a two-hop path exists, else None.
    """
    if len(rel_seq) != 2:
        raise ValueError(f"Expected 2 relations, got {rel_seq}")
    rel1, rel2 = rel_seq

    mids = kg.neighbors(source, rel1)
    if not mids:
        return None

    for mid in sorted(mids):
        tails = kg.neighbors(mid, rel2)
        if answer in tails:
            return [source, rel1, mid, rel2, answer]
    return None

# =========================================================
# Build JSON per split AND collect single-answer evidence triplets
# =========================================================

def build_split_json_collect_single_triplets(
    qa_path: str,
    qtype_path: str,
    kg: KnowledgeGraph,
    out_json_path: str,
    pair_to_relation: Optional[Dict[Tuple[str, str], str]] = None,
    print_every: int = 0
) -> Tuple[int, List[Tuple[str, str, str]]]:
    """
    Writes one JSON array with dicts:
      { "question": str, "answers": List[str], "evidences": List[Optional[List[str]]] }

    Returns:
      single_count: number of single-answer questions in this split
      single_triplets: list of (e1, r, e2) collected only from successful
                       evidences of single-answer items (two triplets per path).
    """
    if pair_to_relation is None:
        pair_to_relation = build_pair_to_relation()

    items = load_qa_file(qa_path)
    qtypes = load_qtype_file(qtype_path)
    if len(items) != len(qtypes):
        raise ValueError(
            f"Line-count mismatch: {qa_path} has {len(items)} items, "
            f"{qtype_path} has {len(qtypes)} lines."
        )

    out = []
    single_count = 0
    single_triplets: List[Tuple[str, str, str]] = []

    for idx, (it, qt) in enumerate(zip(items, qtypes), 1):
        question = it["question"]
        source   = it["source"]
        answers  = it["answers"]

        # Derive relation sequence from qtype (pair-wise)
        rel_seq: Optional[List[str]] = None
        if qt:
            try:
                rel_seq = relseq_from_qtype(qt, pair_to_relation)
            except KeyError:
                rel_seq = None  # no mapping; evidences will be None
        else:
            rel_seq = None

        evidences: List[Optional[List[str]]] = []
        if rel_seq:
            for ans in answers:
                path = find_two_hop_path(kg, source, ans, rel_seq)
                evidences.append(path)
        else:
            evidences = [None] * len(answers)

        out.append({
            "question": question,
            "answers": answers,
            "evidences": evidences
        })

        # For single-answer items, if we have a successful path,
        # collect the two forward triplets into the evidence KG.
        if len(answers) == 1:
            single_count += 1
            if evidences and evidences[0] is not None:
                src, r1, mid, r2, ans = evidences[0]
                single_triplets.append((src, r1, mid))
                single_triplets.append((mid, r2, ans))

        if print_every and idx % print_every == 0:
            print(f"[{qa_path}] processed {idx}/{len(items)} items")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[{qa_path}] wrote {len(out)} items → {out_json_path}")
    return single_count, single_triplets

# =========================================================
# Evidence KG helpers (built ONLY from single-answer evidences)
# =========================================================

def build_evidence_kg(triplets: List[Tuple[str, str, str]]) -> Tuple[Set[str], Set[str], Set[Tuple[str, str, str]]]:
    """
    Given a list of forward triplets (e1, r, e2), build:
      - node_set: all entities appearing in triplets
      - relation_set: all relation names appearing
      - triplet_set: unique triplets
    """
    node_set: Set[str] = set()
    relation_set: Set[str] = set()
    triplet_set: Set[Tuple[str, str, str]] = set()

    for h, r, t in triplets:
        node_set.add(h)
        node_set.add(t)
        relation_set.add(r)
        triplet_set.add((h, r, t))

    return node_set, relation_set, triplet_set

# =========================================================
# Evidence ⊆ Base-KG verification
# =========================================================

def base_edges_set(kg: KnowledgeGraph) -> Set[Tuple[str, str, str]]:
    edges: Set[Tuple[str, str, str]] = set()
    for (h, r), tails in kg.adjacency.items():
        for t in tails:
            edges.add((h, r, t))
    return edges

def check_evidence_subset_of_base(
    evidence_triplets: Set[Tuple[str, str, str]],
    base_edges: Set[Tuple[str, str, str]],
    max_show: int = 20
):
    missing = [tr for tr in evidence_triplets if tr not in base_edges]
    if not missing:
        print("✔ Evidence edges are a subset of the base KG edges.")
        return
    print(f"⚠ Found {len(missing)} evidence edges not present in base KG (showing up to {max_show}):")
    for tr in missing[:max_show]:
        print("   ", tr)

# =========================================================
# Duplicate diagnostics (entities)
# =========================================================

def print_suspicious_duplicates(kg: KnowledgeGraph, top_k: int = 20):
    suspects = kg.suspicious_duplicates(top_k=top_k)
    if not suspects:
        print("No suspicious duplicate entities after normalization.")
        return
    print(f"\nTop {len(suspects)} suspicious duplicates (different raw strings → same normalized):")
    for norm, raw_variants in suspects:
        print(f"  - {norm!r}  <= {raw_variants}")

# =========================================================
# Orchestration + final STATS for the EVIDENCE KG (single-answer only)
# =========================================================

def main(
    kb_path: str,
    qa_train: str, qtype_train: str, out_train: str,
    qa_dev: str,   qtype_dev: str,   out_dev: str,
    qa_test: str,  qtype_test: str,  out_test: str,
):
    # 1) Base KG (with reversed relations) using NORMALIZED entities
    kg = KnowledgeGraph()
    kg.load_from_file(kb_path)

    # Print base-KG stats
    kb_nodes = len(kg.entities)
    kb_edges = kg.edge_count()
    kb_rels  = len(kg.relations)
    print("\n========== BASE KG (traversal graph, normalized, incl. reversed) ==========")
    print(f"Nodes (entities)             : {kb_nodes}")
    print(f"Edges                        : {kb_edges}")
    print(f"Unique relations             : {kb_rels}")

    # Duplicate diagnostics
    print_suspicious_duplicates(kg, top_k=20)

    # 2) Build per-split JSONs (answers/evidences) and collect single-answer evidences
    pair_map = build_pair_to_relation()

    train_single_count, train_triplets = build_split_json_collect_single_triplets(
        qa_train, qtype_train, kg, out_train, pair_map, print_every=1000
    )
    dev_single_count, dev_triplets = build_split_json_collect_single_triplets(
        qa_dev,   qtype_dev,   kg, out_dev,   pair_map
    )
    test_single_count, test_triplets = build_split_json_collect_single_triplets(
        qa_test,  qtype_test,  kg, out_test,  pair_map
    )

    # 3) Build the NEW evidence KG (single-answer only; forward triplets from evidences)
    all_triplets = train_triplets + dev_triplets + test_triplets
    node_set, relation_set, triplet_set = build_evidence_kg(all_triplets)

    # 4) Evidence ⊆ Base-KG check
    base_edges = base_edges_set(kg)
    check_evidence_subset_of_base(triplet_set, base_edges, max_show=20)

    # 5) Final report for the EVIDENCE KG
    print("\n========== EVIDENCE KG (built from single-answer evidences) ==========")
    print(f"Nodes (entities)             : {len(node_set)}")
    print(f"Edges / Triplets (e1,r,e2)   : {len(triplet_set)}")
    print(f"Unique relations             : {len(relation_set)}")

    print("\nSingle-answer questions contributing (counts per split):")
    print(f"  Train: {train_single_count}")
    print(f"  Dev  : {dev_single_count}")
    print(f"  Test : {test_single_count}")

    # Totals before de-dup (for visibility)
    print("\nTriplets collected (with dupes before building set):")
    print(f"  Total collected           : {len(all_triplets)}")



if __name__ == "__main__":
    main(
        kb_path="dataset/metaqa/kb.txt",
        qa_train="dataset/metaqa/vanilla/qa_train.txt", qtype_train="dataset/metaqa/qa_train_qtype.txt",
        out_train="metaqa_train_evidences.json",
        qa_dev="dataset/metaqa/vanilla/qa_dev.txt",     qtype_dev="dataset/metaqa/qa_dev_qtype.txt",
        out_dev="metaqa_dev_evidences.json",  
        qa_test="dataset/metaqa/vanilla/qa_test.txt",   qtype_test="dataset/metaqa/qa_test_qtype.txt",
        out_test="metaqa_test_evidences.json", 
    )
