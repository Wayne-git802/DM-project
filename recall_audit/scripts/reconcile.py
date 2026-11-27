"""Reconcile RECALL-MM rows to the CPSC announcement catalogue.

Matches are auditable record-linkage decisions. Keep candidate lists for review.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import unicodedata
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

API_URL = "https://www.saferproducts.gov/RestWebServices/Recall?format=json&RecallDateStart=2000-01-01&RecallDateEnd=2024-12-31"
LISTING_URL = "https://www.cpsc.gov/s3fs-public/recall-data/recalls_recall_listing.csv"
SOURCE_SHA256 = "cf5b2bc18040aa4ec2cef6006fea858d6c267e07e0746686d3493d88548f1761"
ARCHIVED_NOTICES = {
    2013: {
        "number": "00-190", "date": "2000-09-27",
        "url": "https://www.cpsc.gov/Recalls/2000/cpsc-warns-about-defective-furnaces-in-california",
        "hazard": "Certain gas-fired horizontal furnaces made by Consolidated Industries present a fire risk.",
        "notice_note": "CPSC safety alert. The web page dates the alert September 27, and describes about 190,000 manufactured units.",
        "units": "About 190,000 manufactured", "count_not_recalled": True,
    },
    2795: {
        "number": "00-501", "date": "2007-11-02",
        "url": "https://www.cpsc.gov/Recalls/2008/Bombardier-ATV-Recall",
        "hazard": "A balljoint circlip problem can make the ATV wheels unstable and cause serious injury.",
        "notice_note": "The page's date field is November 2, 2007. The archived release body is dated June 2003 without a day.",
        "units": "",
    },
}
VERIFIED_NON_RECALL = {
    4576: (
        "https://www.cpsc.gov/Newsroom/News-Releases/2000/Nintendo-Settlement-with-New-York-Attorney-General",
        "Nintendo Mario Party protective-glove settlement, matched by hand injuries and the remedy. "
        "The CPSC reports a settlement and does not list a recall number.",
    )
}
FIELD_CORRECTIONS = {
    1009: {
        "recall_description": "The clamp attaching the jumper seat to a door frame can break, allowing the seat to fall.",
        "hazard_classification": ["FALLING"],
        "note": "The name, blue/yellow jumper, and instructions to cut the strap match Kids II 05-198. The original hazard and CUTS tag describe Infantino 05-197, an unrelated soft gym.",
    },
    3111: {
        "recall_description": "The left end of the bicycle crankset spindle can break, posing a fall hazard to the rider.",
        "note": "The BMX model, raw title and spindle replacement identify QBP 16-023. The original bottom-bracket hazard instead matches the 18-074 Brompton folding-bike notice.",
    },
    1185: {
        "visual_product_description": "",
        "note": "The image caption describes a mattress. The recorded hazard and replacement fuse trace to the Cybex treadmill notice.",
    },
    6116: {
        "recall_description": "The recalled charge station can overheat, posing a burn hazard to consumers.",
        "note": "The announcement identifies the charge station, rather than the sensors it supports, as the overheating component.",
    },
}
TEXT_FIELDS = (
    "product_name", "visual_product_description", "recall_description", "remedies",
)
LABEL_FIELDS = (
    "product_classification", "hazard_classification", "remedies_classification",
)
HAZARD_ALIASES = {"HEAT_RELATED_EXPOLOSION": "HEAT_RELATED_EXPLOSION"}
PLACEHOLDERS = {
    "", "-", "n/a", "na", "none", "null", "unknown", "not applicable",
    "not specified", "not available", "not provided", "no description available",
    "a visual description is not available",
}


def clean_text(value):
    if not isinstance(value, str):
        return ""
    value = " ".join(value.split())
    return "" if value.casefold().rstrip(".") in PLACEHOLDERS else value


def words(value):
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", clean_text(value)).casefold()))


def compact_number(value):
    return "".join(re.findall(r"[a-z0-9]+", value.casefold()))


def official_date(value):
    for date_format in ("%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value.strip(), date_format).date().isoformat()
        except ValueError:
            continue
    return ""


def distinct(values):
    return list(dict.fromkeys(v for v in values if v))


def native_announcement_number(value):
    match = re.fullmatch(r"(\d{2})(\d{3}[a-z]*)", value, flags=re.I)
    return "-".join(match.groups()) if match else value


def load_official(api_file, listing_file):
    api = json.loads(api_file.read_text(encoding="utf-8-sig"))
    with listing_file.open(encoding="utf-8-sig", newline="") as handle:
        listing = list(csv.DictReader(handle))
    display_by_key = collections.defaultdict(list)
    for record in listing:
        number = record.get("Recall Number") or ""
        if number.strip():
            key = (compact_number(number), official_date(record["Date"]))
            display_by_key[key].append(record)
    official = []
    for record in api:
        date = record["RecallDate"][:10]
        choices = display_by_key.get((compact_number(record["RecallNumber"]), date), [])
        listing_row = next((row for row in choices if words(row["Recall Heading"]) == words(record["Title"])), None)
        if not listing_row and len(choices) == 1:
            listing_row = choices[0]
        number = listing_row["Recall Number"].strip() if listing_row else native_announcement_number(record["RecallNumber"])
        product_names = distinct(clean_text(x.get("Name")) for x in record["Products"])
        units = distinct(clean_text(x.get("NumberOfUnits")) for x in record["Products"])
        hazards = distinct(clean_text(x.get("Name")) for x in record["Hazards"])
        remedies = distinct(clean_text(x.get("Name")) for x in record["Remedies"])
        official.append({
            "id": record["RecallID"], "number": number, "date": date,
            "number_source": "cpsc_listing" if listing_row else "cpsc_api",
            "url": record["URL"], "title": clean_text(record["Title"]),
            "name": " | ".join(product_names),
            "product_description": clean_text(record["Description"]),
            "hazard": " | ".join(hazards), "hazard_items": hazards,
            "listing_hazard": clean_text(listing_row["Hazard Description"]) if listing_row else "",
            "remedy": " | ".join(remedies),
            "units": " | ".join(units),
            "listing_units": clean_text(listing_row["Units"]) if listing_row else "",
            "remedy_options": distinct(clean_text(x.get("Option")) for x in record["RemedyOptions"]),
            "listing_remedy_type": clean_text(listing_row["Remedy Type"]) if listing_row else "",
        })
    seen_numbers = {compact_number(item["number"]) for item in official}
    for archive_key, details in ARCHIVED_NOTICES.items():
        number = compact_number(details["number"])
        entries = [row for (key, _), rows in display_by_key.items() if key == number for row in rows]
        if number in seen_numbers or not entries:
            continue
        listing_row = entries[0]
        official.append({
            **details, "id": archive_key, "archived": True, "number_source": "cpsc_archived_page",
            "title": clean_text(listing_row["Recall Heading"]), "name": clean_text(listing_row["Name of product"]),
            "product_description": clean_text(listing_row["Description"]),
            "listing_hazard": "", "hazard_items": [details["hazard"]],
            "remedy": clean_text(listing_row["Remedy"]), "remedy_options": [],
            "listing_units": clean_text(listing_row["Units"]),
            "listing_remedy_type": clean_text(listing_row["Remedy Type"]),
        })
    ids = [item["id"] for item in official]
    nums = [compact_number(item["number"]) for item in official]
    if len(set(ids)) != len(official) or len(set(nums)) != len(official) or not all(nums):
        raise ValueError("CPSC catalogue must have unique and nonempty identifiers.")
    return official


def text_vectors(left, right, analyzer, ngrams):
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, min_df=2, dtype=np.float32)
    combined = vectorizer.fit_transform(left + right)
    return combined[:len(left)], combined[len(left):].T.tocsc()


def top_columns(scores, n):
    return np.argpartition(scores, -n, axis=1)[:, -n:]


def rank_candidates(raw, official, batch_size=96):
    raw_hazard = [words(r["recall_description"]) for r in raw]
    raw_names = [words(r["product_name"]) for r in raw]
    raw_remedies = [words(r["remedies"]) for r in raw]
    official_hazard = [words(r["hazard"]) for r in official]
    official_names = [words(r["name"]) for r in official]
    official_remedy = [words(r["remedy"]) for r in official]

    hazard_chars = text_vectors(raw_hazard, official_hazard, "char_wb", (3, 5))
    hazard_words = text_vectors(raw_hazard, official_hazard, "word", (1, 2))
    names = text_vectors(raw_names, official_names, "char_wb", (3, 5))
    remedies = text_vectors(raw_remedies, official_remedy, "char_wb", (3, 5))
    appearance = text_vectors(
        [words(row["visual_product_description"]) for row in raw],
        [words(row["product_description"]) for row in official],
        "char_wb", (3, 5),
    )
    exact_remedies = collections.defaultdict(set)
    for i, remedy in enumerate(official_remedy):
        if len(remedy.split()) >= 8:
            exact_remedies[remedy].add(i)

    ids = {row["id"]: i for i, row in enumerate(official)}
    official_dates = np.asarray([row["date"] for row in official], dtype="U10")
    exact_hazards = collections.defaultdict(set)
    for i, record in enumerate(official):
        for value in [official_hazard[i], *(words(s) for s in record["hazard_items"])]:
            if value:
                exact_hazards[value].add(i)

    for start in range(0, len(raw), batch_size):
        end = min(start + batch_size, len(raw))
        batch = slice(start, end)
        hazards = 0.55 * (hazard_chars[0][batch] @ hazard_chars[1]).toarray()
        hazards += 0.45 * (hazard_words[0][batch] @ hazard_words[1]).toarray()
        name_scores = (names[0][batch] @ names[1]).toarray()
        visual_scores = (appearance[0][batch] @ appearance[1]).toarray()
        remedy_scores = (remedies[0][batch] @ remedies[1]).toarray()
        raw_dates = np.asarray([row["recall_date"][:10] for row in raw[batch]], dtype="U10")
        same_dates = raw_dates[:, None] == official_dates[None, :]
        text_scores = 0.64 * hazards + 0.21 * name_scores + 0.15 * remedy_scores
        score = text_scores + 0.075 * same_dates + 0.045 * visual_scores
        for local_i, row in enumerate(raw[batch]):
            anchor = ids.get(row["recall_number"])
            if anchor is not None:
                score[local_i, anchor] += 0.04
        top = top_columns(score, 5)
        best_hazards = top_columns(hazards, 3)
        for local_i, row in enumerate(raw[batch]):
            source_index = start + local_i
            anchor = ids.get(row["recall_number"])
            exact = exact_hazards.get(raw_hazard[source_index], set())
            exact_remedy = exact_remedies.get(raw_remedies[source_index], set())
            candidate_indices = set(top[local_i]) | set(best_hazards[local_i]) | exact | exact_remedy
            if anchor is not None:
                candidate_indices.add(anchor)
            candidates = []
            for j in sorted(candidate_indices, key=lambda j: (-score[local_i, j], official[j]["id"])):
                item = official[j]
                candidates.append({
                    "id": item["id"], "number": item["number"], "date": item["date"], "url": item["url"],
                    "score": round(float(score[local_i, j]), 5),
                    "hazard": round(float(hazards[local_i, j]), 5),
                    "name": round(float(name_scores[local_i, j]), 5),
                    "remedy": round(float(remedy_scores[local_i, j]), 5),
                    "visual": round(float(visual_scores[local_i, j]), 5),
                    "exact_remedy": j in exact_remedy,
                    "same_date": bool(same_dates[local_i, j]), "exact_hazard": j in exact,
                    "anchor": bool(j == anchor),
                })
            yield candidates


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


COUNT = re.compile(
    r"^(?:(?:about|approximately|around|at least|up to|more than|over|nearly)\s+)?"
    r"([0-9][0-9,]*(?:\.[0-9]+)?)(?:\s+(thousand|million|billion))?\b",
    re.I,
)
NON_COUNT = re.compile(
    r"\b(?:gallons?|pounds?|ounces?|liters?|litres?|kilograms?|tons?|feet|yards?|grams?)\b",
    re.I,
)
UNIT_SCALE = {"thousand": 10**3, "million": 10**6, "billion": 10**9}
OFFICIAL_REMEDY_CODES = {
    "refund": "REFUND", "repair": "REPAIR", "replace": "REPLACE",
    "dispose": "DISPOSE", "new instructions": "NEW_INSTRUCTIONS",
    "remedy no longer available": "REMEDY_NO_LONGER_AVAILABLE",
    "no remedy available": "REMEDY_NO_LONGER_AVAILABLE",
}


def count_from_units(value):
    # Parenthetical overseas counts are kept in the source text, not added.
    main = re.sub(r"\([^)]*\)", " ", clean_text(value)).strip()
    match = COUNT.match(main)
    if not match or NON_COUNT.search(main):
        return None
    suffix = main[match.end():]
    if re.search(r"\d", suffix) or re.match(r"\s*(?:[-–/]|to\b|through\b)", suffix):
        return None
    if re.search(
        r"\b(?:reports?|injur(?:y|ies)|fatalit(?:y|ies)|incident(?:s)?|annually|"
        r"(?:each|per|every) (?:year|month|week|day))\b",
        suffix, re.I,
    ):
        return None
    number, scale = match.groups()
    value = Decimal(number.replace(",", "")) * UNIT_SCALE.get((scale or "").lower(), 1)
    return int(value) if value > 0 and value == value.to_integral_value() else None


def official_count(item):
    if item.get("count_not_recalled"):
        return None, "manufacturing_total_only"
    values = distinct([count_from_units(item["units"]), count_from_units(item["listing_units"])])
    if len(values) > 1:
        return None, "official_counts_disagree"
    if values:
        return values[0], "readable_notice_count"
    return None, "no_usable_count"


def has_support(candidate):
    return candidate["hazard"] >= 0.49 or candidate["name"] >= 0.51 or candidate["remedy"] >= 0.67


def resolve_row(row, candidates, official, is_repeated):
    best, alternative = candidates[:2]
    anchor = next((c for c in candidates if c["anchor"]), None)
    if anchor is not None and anchor["exact_hazard"] and (
        best is anchor or best["score"] - anchor["score"] < 0.13
    ):
        return anchor, "source_id_exact_hazard"

    distinct_exact = sum(c["exact_hazard"] for c in candidates) == 1
    if not best["anchor"]:
        score_gap = best["score"] - alternative["score"]
        gain_on_anchor = best["score"] - anchor["score"] if anchor else 1.0
        double_exact = (
            best["exact_hazard"] and best["exact_remedy"]
            and sum(c["exact_hazard"] and c["exact_remedy"] for c in candidates) == 1
            and best["name"] >= 0.28
        )
        exact_supported = (
            best["exact_hazard"] and distinct_exact and score_gap >= 0.037
            and (best["name"] >= 0.36 or best["remedy"] >= 0.43 or best["visual"] >= 0.28)
        )
        fuzzy_supported = (
            best["hazard"] >= 0.74 and best["score"] >= 0.71 and score_gap >= 0.080
            and (best["name"] >= 0.48 and best["remedy"] >= 0.42 or best["remedy"] >= 0.58
                 or best["hazard"] >= 0.87 and best["visual"] >= 0.19)
            and (anchor is None or best["hazard"] - anchor["hazard"] >= 0.17)
        )
        if gain_on_anchor >= 0.055 and (double_exact or exact_supported or fuzzy_supported):
            if double_exact:
                return best, "reassigned_exact_hazard_and_remedy"
            return best, "reassigned_exact_hazard" if exact_supported else "reassigned_corroborated_text"

    if anchor is None:
        return None, "source_id_unavailable"
    qty, _ = official_count(official[anchor["id"]])
    quantity_matches = qty == row["product_quantity"] and qty is not None
    alternative_wins = not best["anchor"] and best["score"] - anchor["score"] > 0.055
    conflict = alternative_wins and best["hazard"] >= 0.66 and (
        best["hazard"] - anchor["hazard"] > 0.18 or best["remedy"] - anchor["remedy"] > 0.20
    ) and (best["name"] >= 0.18 or best["remedy"] >= 0.35 or best["visual"] >= 0.20)
    if anchor["same_date"] and not conflict:
        if best["anchor"] and has_support(anchor) and (
            not is_repeated or anchor["hazard"] >= 0.44
            or quantity_matches and anchor["name"] > 0.80
        ):
            return anchor, "source_id_date_and_text"
        if quantity_matches and (
            anchor["name"] >= (0.70 if is_repeated else 0.39) or anchor["remedy"] >= 0.53
            or anchor["hazard"] >= 0.36
        ) and (not alternative_wins or best["hazard"] < 0.52):
            return anchor, "source_id_date_count_and_text"
        if not is_repeated and max(anchor["name"], anchor["visual"]) >= 0.48 and (
            best["anchor"] or best["name"] < 0.15 and best["visual"] < 0.14 and best["remedy"] < 0.25
        ):
            return anchor, "source_id_date_and_product"

    if best["anchor"]:
        both_fields = anchor["hazard"] >= 0.70 and (anchor["name"] >= 0.38 or anchor["remedy"] >= 0.60)
        near_date = abs((datetime.fromisoformat(anchor["date"]) - datetime.fromisoformat(row["recall_date"][:10])).days) <= 1
        near_supported = near_date and best["score"] - alternative["score"] >= 0.10 and (
            anchor["hazard"] >= 0.43 and anchor["name"] >= 0.50 or anchor["hazard"] >= 0.60
        )
        if both_fields and best["score"] - alternative["score"] >= 0.09 or near_supported:
            return anchor, "source_id_corroborated_text"
    return None, "ambiguous_or_low_text_support"


def unique_texts(rows, field):
    values = {}
    for _, row, _ in rows:
        text = clean_text(row[field])
        key = words(text)
        if key:
            values.setdefault(key, text)
    return list(values.values())


def recode_label(label):
    return HAZARD_ALIASES.get(label, label)


def remedy_labels(item):
    content = item["remedy_options"] + re.split(r"[|;,\n]+", item["listing_remedy_type"])
    return sorted(set(
        OFFICIAL_REMEDY_CODES[normalized]
        for value in content
        if (normalized := words(value)) in OFFICIAL_REMEDY_CODES
    ))


def merge_notice(official, rows):
    variants = {field: unique_texts(rows, field) for field in TEXT_FIELDS}
    labels = {
        field: sorted(set(recode_label(label) for _, row, _ in rows for label in row[field]))
        for field in LABEL_FIELDS
    }
    official_qty, quantity_status = official_count(official)
    return {
        "recall_number": official["number"],
        "cpsc_recall_id": None if official.get("archived") else official["id"],
        "recall_date": official["date"],
        "product_quantity": official_qty,
        **{field: "\n\n".join(values) for field, values in variants.items()},
        **labels,
        "text_variants": variants,
        "source_rows": [index for index, _, _ in rows],
        "source_recall_numbers": sorted(set(row["recall_number"] for _, row, _ in rows)),
        "official": {
            "url": official["url"], "title": official["title"], "product_name": official["name"],
            "product_description": official["product_description"], "number_source": official["number_source"],
            "hazard_description": official["listing_hazard"] or official["hazard"],
            "remedy_text": official["remedy"], "remedy_labels": remedy_labels(official),
            "units_text": official["units"], "listing_units_text": official["listing_units"],
            "quantity_status": quantity_status, "notice_note": official.get("notice_note", ""),
        },
    }


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(candidate, official):
    entry = official[candidate["id"]]
    return {**candidate, "title": entry["title"], "official_hazard": entry["listing_hazard"] or entry["hazard"],
            "official_product": entry["name"], "units": entry["units"]}


def reconcile(raw, catalogue, ranked, overrides=None):
    by_id = {official["id"]: official for official in catalogue}
    sizes = collections.Counter(row["recall_number"] for row in raw)
    groups = collections.defaultdict(list)
    audit = []
    for i, (row, candidates) in enumerate(zip(raw, ranked, strict=True), start=1):
        original_id = row["recall_number"]
        matched, how = resolve_row(row, candidates, by_id, sizes[original_id] > 1)
        # Each manual exception carries a CPSC URL and the distinguishing evidence.
        override = (overrides or {}).get(i)
        if override is not None:
            official_id, note = override
            matched = next((c for c in candidates if c["id"] == official_id), None)
            if matched is None and official_id is not None:
                item = by_id[official_id]
                matched = {"id": item["id"], "number": item["number"], "date": item["date"], "url": item["url"]}
            how = "reviewed_against_cpsc" if matched else "reviewed_still_ambiguous"
        else:
            note = ""
        if i in VERIFIED_NON_RECALL:
            matched, how = None, "official_settlement_without_recall_number"
            note = VERIFIED_NON_RECALL[i][1]
        item = by_id[matched["id"]] if matched else None
        change = FIELD_CORRECTIONS.get(i) if item else None
        quantity = official_count(item)[0] if item else None
        entry = {
            "source_row": i, "source_recall_number": original_id, "within_source_id_rows": sizes[original_id],
            "source_recall_date": row["recall_date"][:10], "source_quantity": row["product_quantity"],
            "source_product_name": row["product_name"], "decision": how, "review_note": note,
            "official_number": item["number"] if item else None,
            "official_id": item["id"] if item and not item.get("archived") else None,
            "official_date": item["date"] if item else None, "official_quantity": quantity,
            "official_url": item["url"] if item else None,
            "changed_id": item["id"] != original_id if item else None,
            "changed_date": item["date"] != row["recall_date"][:10] if item else None,
            "changed_quantified_quantity": quantity != row["product_quantity"] if quantity is not None else None,
        }
        if i in VERIFIED_NON_RECALL:
            entry["official_url"] = VERIFIED_NON_RECALL[i][0]
        if matched:
            assigned_row = row
            if change:
                assigned_row = {**row, **{field: value for field, value in change.items() if field != "note"}}
                entry["field_correction"] = change
                entry["source_record"] = row
            groups[item["id"]].append((i, assigned_row, how))
            entry["matched_evidence"] = matched
        else:
            entry["source_record"] = row
        if not matched or item["id"] != original_id or how == "reviewed_against_cpsc":
            entry["candidate_evidence"] = [evidence(c, by_id) for c in candidates[:3]]
        audit.append(entry)

    records = [
        merge_notice(by_id[official_id], rows)
        for official_id, rows in groups.items()
    ]
    records.sort(key=lambda row: (row["recall_date"], row["recall_number"]))
    assigned = [source_row for record in records for source_row in record["source_rows"]]
    pending = [entry["source_row"] for entry in audit if entry["official_number"] is None]
    assert len(set(record["recall_number"] for record in records)) == len(records)
    assert sorted(assigned + pending) == list(range(1, len(raw) + 1))
    return records, audit


def audit_summary(records, audit):
    mapped = [a for a in audit if a["official_number"] is not None]
    source_groups = collections.defaultdict(list)
    for entry in audit:
        source_groups[entry["source_recall_number"]].append(entry)
    repeated_groups = [rows for rows in source_groups.values() if len(rows) > 1]
    split_groups = sum(len({row["official_number"] for row in rows if row["official_number"]}) > 1 for rows in repeated_groups)
    non_recall = sum(a["decision"] == "official_settlement_without_recall_number" for a in audit)
    return {
        "raw_rows": len(audit), "raw_ids": len(source_groups),
        "repeated_source_id_groups": len(repeated_groups),
        "rows_in_repeated_source_id_groups": sum(map(len, repeated_groups)),
        "mapped_rows": len(mapped), "unresolved_rows": len(audit) - len(mapped) - non_recall,
        "officially_identified_non_recall_rows": non_recall,
        "manual_field_corrections": sum("field_correction" in entry for entry in audit),
        "corrected_announcements": len(records), "merged_source_rows": len(mapped) - len(records),
        "announcements_with_multiple_rows": sum(len(row["source_rows"]) > 1 for row in records),
        "announcements_with_multiple_visual_descriptions": sum(
            len(row["text_variants"]["visual_product_description"]) > 1 for row in records
        ),
        "confirmed_split_source_groups": split_groups,
        "changed_internal_id_rows": sum(bool(a["changed_id"]) for a in mapped),
        "changed_date_rows": sum(bool(a["changed_date"]) for a in mapped),
        "changed_quantity_rows": sum(bool(a["changed_quantified_quantity"]) for a in mapped),
        "quantity_unknown_announcements": sum(row["product_quantity"] is None for row in records),
        "visible_remedy_label_disagreements": sum(
            bool(row["official"]["remedy_labels"]) and row["remedies_classification"] != row["official"]["remedy_labels"]
            for row in records
        ),
        "by_decision": dict(collections.Counter(a["decision"] for a in audit)),
    }


OVERRIDES = {
    487: (803, "The image shows the fish and lizard with pop-out tongues named in the Push'n Pop notice."),
    686: (9331, "Avocado print plus the Tencel, cotton and spandex blend distinguish Loulou Lollipop."),
    920: (774, "Smart Solar's anti-mosquito/citronella 30-ounce bottle, rather than Real Flame's 29.9 or 29.98 ounces."),
    1009: (2952, "The blue seat, yellow strap, blue door-frame clamp and cut-strap remedy identify Kids II. Corrected the foreign hazard and its tag."),
    1446: (1778, "A center face surrounded by hearts, stars and rings identifies the Whoozit teether."),
    2530: (8167, "Gray two-piece pajamas with pink trim and the exact Happy Day V&L print."),
    3111: (6558, "BMX geometry, product name and inspection/replacement remedy identify QBP. Corrected the foreign hazard wording."),
    3195: (185, "Two sizes of clear glass bottles plus the verbatim stopper hazard and online return procedure."),
    3924: (785, "Five-inch yellow Tweety rattle with a detachable blue and white bib."),
    4960: (9833, "Magnets that fit the small parts cylinder, the animal pieces and the prepaid return label."),
    5050: (8208, "A three-inch red Surprise Fountain cylinder rather than the Wholesale Fireworks rockets/shells."),
    5328: (3825, "Front bearing carrier material thickness, distinct from the separate Hawkeye steering-post notice."),
    5406: (6797, "The 32-ounce cleaning bottle matches the bundled historical release for recalled Richfood bleach."),
    5472: (465, "The toy drum and detachable/small mallet ends match the Little Lessons Music Makers recall."),
    5512: (6776, "The etched illuminated number 20 race car matches the Home Depot/Tony Stewart model."),
    6478: (9309, "Copper Pearl two-piece print and prepaid mailers for direct purchasers."),
    6716: (6795, "Opposites Take-Home Pack is named explicitly in the official catalogue."),
    12: (7896, "Identical overheating and smoke wording, and the same partial refund or replacement remedy."),
    170: (5041, "The baby safety warning and supplier refund/exchange wording track the later Top Goods Trading notice."),
    219: (221, "Rail detachment from tracks, slat detachment, and the voucher for certain older models."),
    304: (889, "The distinctive refund based on prior use and unexpected rail drop match the same notice."),
    527: (389, "Oil entering the engine compartment and a fuel tank rubbing against the engine are both listed."),
    638: (46, "Deformed plastic hardware and the upside-down installation shown in the caption; fixed-side repair kit."),
    639: (9900, "Exact hazard, Tentock's disposal/refund procedure, and the orange 1,000 mL bottle."),
    664: (891, "B.O.B. canopy embroidery backing patch, distinct from the earlier drawstring notice."),
    790: (4292, "Detached impeller and the dealer's free inspection and impeller replacement."),
    949: (3422, "Rear brake-caliper fluid leak and reduced rear braking, not the other Prowler defect."),
    1003: (659, "Wiring, overheating or breakage, and replacement or refund for the Stealth aquarium heaters."),
    1036: (7749, "Eight-light Currey/Ethan Allen model and verbatim mounting-loop impact hazard."),
    1172: (1177, "Verbatim Spectrum mounting-loop hazard, matching notice date and 165-unit count."),
    1185: (4990, "Unexpected acceleration from the lower control board, and the free replacement fuse."),
    1691: (1412, "Tern frame hinge, same date and 175-unit count as row 5654."),
    1982: (4678, "Welded drive-axle joints cracking, locking the track and throwing the rider."),
    2100: (4067, "Square D counterfeit breakers and the distinct free inspection and replacement procedure."),
    2478: (75, "Draper's free retrofit removes both the exposed inner and operating cords."),
    2608: (3372, "The filler neck can separate following a crack and receives a reinforcement kit."),
    2748: (3116, "Suspension spindle, chafing fuel tank and the combined model list match the later notice."),
    2862: (4981, "Disengaged spring peg, matching the 600,000-unit Delta notice."),
    3049: (4993, "The model list includes the car, stackable bear and bell rattle; the rattle-handle standard is mentioned."),
    3151: (8553, "The white Igloo Marine Elite cooler's stainless-steel latch locks when the lid closes."),
    3152: (6040, "The CFI snowmobiles' fuel lines can make contact with one another."),
    3247: (230, "Exact wording, and the remedy offers both drawstring removal and a store refund."),
    3442: (895, "A reannouncement for shades redistributed without the repair kit."),
    3777: (1779, "A crane bucket with a 25-inch strap, one line of the combined Fisher-Price announcement."),
    3783: (2013, "Archived release 00-190 identifies Consolidated horizontal furnaces and burner/heat exchanger failures."),
    3831: (3961, "SIDS reduction claims and mesh-frame mattress match the 00-079 safety advisory in the catalogue."),
    3886: (883, "The Kenyon portable stove's automatic fuel shutoff may fail if its canister overheats."),
    3903: (1156, "Graco walker collapse and repair kit that includes the specific replacement play tray."),
    3920: (2006, "Kawasaki ATV drive belt can jam the transmission, with a next-generation replacement belt."),
    4049: (2109, "Tip-over under a heavy load on the open oven door and anti-tip bracket remedy."),
    4165: (2465, "The image caption names Honda Rincon; suspension-arm assembly and corrective torque distinguish this case."),
    4180: (2239, "The Eiger's manufacturing flaw can break the crankshaft, with a time-limited replacement option."),
    4563: (714, "Cable-light power switches produce shock risk; the product description uses the same rope-light definition."),
    4575: (2795, "Archived release 00-501 for the Bombardier DS 650 explicitly names the balljoint circlip."),
    5443: (8088, "Verbatim halogen bulb melting hazard in the Lumicentro recall."),
    5541: (8512, "Kubota zero-turn mower's throttle cable contacts internal electrical wiring."),
    5639: (8734, "Graco vibrating/rocking seat and cash refund or voucher, distinct from the same-date Delta case."),
    5654: (1412, "Tern frame hinge, same date and 175-unit count as row 1691."),
    5729: (3567, "Claire's pendant and backing-card descriptions, with refund or free replacement."),
    5799: (9513, "Properly Tied 100% cotton lounge pants with the prepaid return shipping remedy."),
    6116: (9869, "Specific Go Direct charging station, visible ports and the return-for-replacement remedy."),
    6332: (9077, "All three named chemicals and the child-resistant packaging remedy match the same Williams notice."),
    6844: (9504, "Selfie Craft color-in outlines for the long-sleeve pajama set, plus the destruction/refund procedure."),
}


def build_notice_bundle(raw, source_sha256, cpsc_api, cpsc_listing):
    """Reconcile the unchanged source rows so recorded row positions remain valid."""
    if source_sha256 != SOURCE_SHA256:
        raise ValueError("This review uses row-level decisions for the recorded RECALL-MM source hash.")
    catalogue = load_official(cpsc_api, cpsc_listing)
    records, audit = reconcile(raw, catalogue, rank_candidates(raw, catalogue), OVERRIDES)
    summary = audit_summary(records, audit)
    summary["sources"] = {
        "recall_mm_sha256": source_sha256,
        "cpsc_api_url": API_URL, "cpsc_api_sha256": hash_file(cpsc_api),
        "cpsc_listing_url": LISTING_URL, "cpsc_listing_sha256": hash_file(cpsc_listing),
    }
    return records, summary, audit


def write_notice_bundle(output_dir, records, summary, audit):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "dataset_corrected.json", records)
    write_json(output_dir / "reconciliation_summary.json", summary)
    with (output_dir / "source_row_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in audit:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cpsc_api", type=Path)
    parser.add_argument("cpsc_listing", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    records, summary, audit = build_notice_bundle(
        json.loads(source_bytes), hashlib.sha256(source_bytes).hexdigest(),
        args.cpsc_api, args.cpsc_listing,
    )
    write_notice_bundle(args.output_dir, records, summary, audit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
