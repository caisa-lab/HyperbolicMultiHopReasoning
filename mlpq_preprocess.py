#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

# -------------------------------------------------------------------
# Mapping French relation names -> English relation names
# Extend / adapt this mapping to match your dataset exactly.
# Keys must be exactly as they appear in the input files.
# -------------------------------------------------------------------
RELATION_MAP = {
    "langue": "language",
    "secteursd'activités": "business_sectors",
    "secteurs d’activités": "business_sectors",
    "sociétémère": "parent_company",
    "maire": "mayor",
    "légendedrapeau": "banner",
    "comté": "county",
    "région": "region",
    "domaineinternet": "internet_domain_name",
    "typegouvernement": "government_type",
    "championnatactuel": "current_championship",
    "préfecture": "prefecture",
    "personnagesclés": "key_person",
    "profession": "occupation",
    "profession(s)": "occupation",
    "profession(s)_": "occupation",
    "pays": "country",
    "équipementsportif": "stadium",
    "siège(ville)_": "headquarter(city)",
    "ville": "city",
    "département": "department",
    "hymnenational": "national_anthem",
    "filiale": "subsidiary",
    "plusgrandeville": "largest_city",
    "typeindépendance": "independent_type",
    "fondateur": "founder",
    "label": "record_company",
    "siège": "headquarter",
    "légendeblason": "badge",
    "lieudenaissance": "birthplace",
    "albumprécédent": "previous_album",
    "filiales": "subsidiaries",
    "pointculminant": "highest_point",
    "gouverneur": "governor",
    "nationalité": "nationality",
    "subdivision": "province",
    "genre": "genre",
    "monnaie": "currency",
    "propriétaire": "owner",
    "capitale": "capital",
    "arrondissement": "administrative_area",
}

unique_nodes = set()


def parse_line(line: str, file_type: str):
    """
    Parse a single line from one of the input files.

    line format:
        question<TAB>entity@@@relation@@@entity###entity@@@relation@@@entity

    file_type: "en_fr" or "fr_en"
        - "en_fr": first middle entity is English
        - "fr_en": second middle entity is English
    """
    line = line.strip()
    if not line:
        return None

    try:
        question, path = line.split("\t", 1)
    except ValueError:
        # line does not have the expected tab structure
        return None

    segments = path.split("###")
    if len(segments) != 2:
        return None
    
    try:
        h1, r1, m1 = segments[0].split("@@@")
        m2, r2, t = segments[1].split("@@@")
    except ValueError:
        # segment does not split into 3 parts
        return None
    

    # other_mapping = {'placeofbirth': 'birthplace', 'timezonedst': 'timezone', 'timezone1dst': 'timezone'}
    other_mapping = {}
    # Choose middle entity depending on file_type
    if file_type == "en_fr":
        middle = m1
        r2_en = RELATION_MAP.get(r2.lower().strip(), r2.lower().strip())
        r1_en = other_mapping.get(r1.lower(), r1.lower())
    elif file_type == "fr_en":
        middle = m2
        r1_en = RELATION_MAP.get(r1.lower().strip(), r1.lower().strip())
        r2_en = other_mapping.get(r2.lower(), r2.lower())
    else:
        raise ValueError(f"Unknown file_type: {file_type}")
    unique_nodes.add(h1)
    unique_nodes.add(middle)
    unique_nodes.add(t)
    evidences = [h1, r1_en, middle, r2_en, t]

    sample = {
        "question": question,
        "answer": t,
        "evidences": evidences,
    }
    return sample


def read_file(path: Path, file_type: str):
    path = Path(path)
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line, file_type)
            if parsed is not None:
                samples.append(parsed)
    return samples


def split_dataset(samples, train_ratio=0.8, dev_ratio=0.1, seed=42):
    random.Random(seed).shuffle(samples)
    n = len(samples)
    n_train = int(train_ratio * n)
    n_dev = int(dev_ratio * n)
    n_test = n - n_train - n_dev

    train = samples[:n_train]
    dev = samples[n_train:n_train + n_dev]
    test = samples[n_train + n_dev:]
    return train, dev, test


def main():
    parser = argparse.ArgumentParser(
        description="Create train/dev/test JSON files from r2r_en_fr_question_en and r2r_fr_en_question_en."
    )
    parser.add_argument("--en_fr", type=Path, required=True,
                        help="Path to r2r_en_fr_question_en file")
    parser.add_argument("--fr_en", type=Path, required=True,
                        help="Path to r2r_fr_en_question_en file")
    parser.add_argument("--out_dir", type=Path, default=Path("."),
                        help="Output directory for train.json, dev.json, test.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    # Read and parse both files
    samples_en_fr = read_file(args.en_fr, "en_fr")
    samples_fr_en = read_file(args.fr_en, "fr_en")

    all_samples = samples_en_fr + samples_fr_en

    train, dev, test = split_dataset(all_samples, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data in [("2hop_train_question_evidences", train), ("2hop_dev_question_evidences", dev), ("2hop_test_question_evidences", test)]:
        out_path = args.out_dir / f"{split_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(data)} samples to {out_path}")

    print(len(unique_nodes))


if __name__ == "__main__":
    main()
