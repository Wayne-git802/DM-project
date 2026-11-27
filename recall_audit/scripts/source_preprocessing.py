"""Build source-row quality and legacy views from the accepted RECALL-MM snapshot."""
from dataclasses import dataclass
import json
import re

import pandas as pd

from reconcile import HAZARD_ALIASES, LABEL_FIELDS, TEXT_FIELDS, clean_text


SOURCE_FIELDS = ["recall_number", "recall_date", "product_name", "product_quantity",
                 *TEXT_FIELDS[1:], *LABEL_FIELDS]
FIRST_YEAR, TRAIN_END, LAST_YEAR = 2000, 2018, 2024

ALLOWED_LABELS = {
    "product_classification": {
        "BABY_PRODUCTS", "CLOTHING_ACCESSORIES", "ELECTRICAL", "FURNITURE", "HOME_APPLIANCES",
        "OTHER", "OUTDOOR_EQUIPMENT", "SPORTS_RECREATION", "TOOLS_HARDWARE", "TOYS_CHILDREN", "TRANSPORTATION",
    },
    "hazard_classification": {
        "BURN", "CHOKING", "CRASH", "CUTS", "FALLING", "FIRE", "HEAT_RELATED_EXPLOSION",
        "IMPACT", "OTHER", "POISONING", "SAFETY_EQUIPMENT_MALFUNCTION",
    },
    "remedies_classification": {
        "DISPOSE", "NEW_INSTRUCTIONS", "REFUND", "REMEDY_NO_LONGER_AVAILABLE", "REPAIR", "REPLACE",
    },
}


HANDOFF_COLUMNS = ("recall_year", "appliance_subtype", "failure_modes", "visible_structures")

MODES = {
    '过热': r'overheat(?:s|ed|ing)?',
    '电气短路或电弧': r'short[- ]circuit(?:s|ed|ing)?|(?:electrical|electric) shorts?|arc(?:ing|s)',
    '破裂或断裂': r'break(?:s|ing)?|broken|crack(?:s|ed|ing)?|shatter(?:s|ed|ing)?|fracture(?:s|d)?',
    '松脱或分离': r'detach(?:es|ed|ing)?|separat(?:e|es|ed|ing)|(?:come|comes|become|becomes) loose|loosen(?:s|ed|ing)?|fall off',
    '泄漏': r'leak(?:s|ed|ing|age)?',
    '意外启动': r'(?:turn on|activate|start) (?:unexpectedly|unintentionally|by itself)|(?:unintentionally|unexpectedly) (?:turn on|activate|start)',
}


MODES['过热'] += r'|(?:get|gets|become) too hot|temperatures? (?:exceeds? allowable|can get too high)|elevated internal temperature|temperature of the storage drawer can become extremely high|heat to build up'


MODES['电气短路或电弧'] += r'|(?:electric|electrical) arc|short(?:s|ed)? out'


MODES['破裂或断裂'] += r'|ruptur(?:e|es|ed)|burst(?:s)?'


MODES['松脱或分离'] += r'|pulled loose|dislodg(?:e|es|ed)|detachments?|disengage|fall from (?:its housing|the ceiling|the wall)'


MODES['泄漏'] += r'|(?:flue gas|gas) (?:can|to) escape|gas to be released'


MODES['意外启动'] += r'|turn on (?:by themselves|without user input|spontaneously)|activated by accidental contact'

def matches(dictionary, phrase):
    return [name for name, pattern in dictionary.items()
            if re.search(r'\b(?:' + pattern + r')\b', phrase, re.I)]

TYPES = {
    'water heater': r'water heaters?',
    'boiler': r'boilers?',
    'fireplace': r'fireplaces?',
    'furnace': r'furnaces?',
    'heater': r'heaters?',
    'air conditioner or heat pump': r'air conditioners?|heat pumps?',
    'dehumidifier': r'dehumidifiers?',
    'refrigerator or freezer': r'refrigerators?|freezers?',
    'dishwasher': r'dishwashers?',
    'washing machine': r'washing machines?|clothes washers?',
    'coffee or espresso machine': r'coffee\s?(?:makers?|brewers?|machines?|centers?)|espresso (?:makers?|machines?)',
    'pressure cooker': r'pressure cookers?',
    'range, oven or cooktop': r'ranges?|ovens?|cooktops?',
    'fan': r'fans?',
    'vacuum cleaner': r'vacuums?',
}

EXCLUDE = re.compile(
    r'\b(?:range hoods?|oven mitts?|oven gloves?|vacuum flasks?|dutch ovens?|oven liners?|'
    r'oven rack guards?|cooktop fire suppressors?|drip pans?|fire suppression devices?|oven cleaner|'
    r'gas control valves for|water heater (?:gas )?temperature controls?|spas and hot tubs|'
    r'fireplace screens?|fireplace gas valves?|gas valves used in|remote control kits?|'
    r'attachment tools?|fan electric motor|coil drain pans?|boiler (?:operating|control|controller)|'
    r'control units for boilers|service kit|accessory (?:electric )?heaters?|defrost heaters?|'
    r'(?:spa|pool|immersion) heaters?)\b', re.I)

STRUCTURES = {
    "folding mechanism or hinge": r"foldable|folding|fold[- ]lock|fold release|hinges?|hinged",
    "wheel or caster": r"wheels|casters?|castors?",
    "strap, harness or seat belt": r"straps?|harness(?:es)?|seat belts?",
    "power cord or plug": r"power cords?|power cables?|electrical plugs?|power plugs?",
    "battery compartment or cover": r"battery compartments?|battery covers?|battery doors?",
    "drawstring or bead chain": r"drawstrings?|pull cords?|bead(?:ed)? chains?",
    "door or lid": r"doors?|lids?|hinged covers?",
    "vent or grille": r"air vents?|ventilation slots?|grilles?|grills?",
    "external handle": r"handles?|handgrips?|grip bars?",
}


FEATURE_PATTERNS = {name: re.compile(rf"\b(?:{terms})\b", re.I) for name, terms in STRUCTURES.items()}


OUTCOME_WORDS = re.compile(
    r"\b(?:hazards?|recalled?|safety recall|risks? of|poses?|unsafe|danger(?:ous)?|"
    r"overheat\w*|chok\w*|strangl\w*|poison\w*|injur\w*|lacerat\w*|electrocut\w*|"
    r"amputat\w*|caught in|finger entrapment|fire hazard|burn hazard)\b", re.I,
)


def visible_structure(text):
    sentences = [s for s in re.split(r"[.!?;](?=\s|$)", text) if not OUTCOME_WORDS.search(s)]
    return {name for name, pattern in FEATURE_PATTERNS.items() if any(pattern.search(s) for s in sentences)}

def recall_year(date):
    if isinstance(date, str) and len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    return None


def extract_row_features(row):
    """Generate the four features on the same source row as the texts and labels."""
    product_name = row.get("product_name") or ""
    subtype = None
    if "HOME_APPLIANCES" in row["product_classification"] and not EXCLUDE.search(product_name):
        subtype = next(iter(matches(TYPES, product_name)), None)
    return {
        "recall_year": recall_year(row.get("recall_date")),
        "appliance_subtype": subtype,
        "failure_modes": matches(MODES, row.get("recall_description") or ""),
        "visible_structures": sorted(visible_structure(row.get("visual_product_description") or "")),
    }


def validate_handoff(rows):
    """Check that every exported record has valid handoff features."""
    fields = set(HANDOFF_COLUMNS)
    for row in rows:
        number = row.get("recall_number")
        present = fields.intersection(row)
        if present != fields:
            raise ValueError(f"召回 {number} 缺少交接字段 {sorted(fields - present)}，请重新运行预处理。")

        year = row["recall_year"]
        if type(year) is not int or year != recall_year(row.get("recall_date")):
            raise ValueError(f"召回 {number} 的年份和日期不一致。")
        subtype = row["appliance_subtype"]
        if subtype is not None and (
            not isinstance(subtype, str) or subtype not in TYPES
            or "HOME_APPLIANCES" not in row["product_classification"]
        ):
            raise ValueError(f"召回 {number} 的家电子类不在交接词典或大类范围内。")
        for field, vocabulary in (("failure_modes", MODES), ("visible_structures", STRUCTURES)):
            values = row[field]
            if not isinstance(values, list) or not all(isinstance(v, str) and v in vocabulary for v in values):
                raise ValueError(f"召回 {number} 的 {field} 与交接词典不符。")


@dataclass
class SourceTables:
    overview: dict
    field_coverage: pd.DataFrame
    cleaned: pd.DataFrame
    handoff: pd.DataFrame
    quality_review: pd.DataFrame


def _is_missing(value):
    if isinstance(value, str):
        return not clean_text(value)
    if isinstance(value, (list, dict)):
        return not value
    return pd.isna(value)


def prepare_source_rows(records):
    """Clean a source-row view while leaving the original rows and their positions intact."""
    source = pd.DataFrame.from_records(records).reindex(columns=SOURCE_FIELDS)
    source.insert(0, "source_row", range(1, len(source) + 1))
    dates = pd.to_datetime(source["recall_date"], format="ISO8601", errors="coerce", utc=True)
    ids = pd.to_numeric(source["recall_number"], errors="coerce")
    valid_id = (ids.gt(0) & ids.mod(1).eq(0)
                & ~source["recall_number"].map(lambda value: isinstance(value, bool)))

    first_by_value = {}
    first_rows = pd.Series([
        first_by_value.setdefault(json.dumps(row, sort_keys=True), number)
        for number, row in enumerate(records, start=1)
    ], index=source.index)
    duplicate = source["source_row"].ne(first_rows)
    overview = {
        "Source rows": len(source),
        "Distinct valid source IDs": ids.where(valid_id).nunique(),
        "Exact duplicate rows": int(duplicate.sum()),
        "First source date": dates.min().date(),
        "Last source date": dates.max().date(),
    }
    field_coverage = pd.DataFrame({
        "Field": SOURCE_FIELDS,
        "Missing or placeholder rows": [int(source[field].map(_is_missing).sum()) for field in SOURCE_FIELDS],
    })
    cleaned = source.copy()
    review_entries = []

    def log_issue(mask, field, issue, decision, evidence=None):
        selected = source.loc[mask.fillna(False), ["source_row", "recall_number"]].copy()
        if selected.empty:
            return
        selected["field"] = field
        selected["issue"] = issue
        selected["decision"] = decision
        selected["original_value"] = source.loc[selected.index, field].map(repr)
        if evidence is not None:
            selected["evidence"] = evidence.loc[selected.index]
        review_entries.append(selected)

    log_issue(duplicate, "recall_number", "完全重复行", "保留首行", first_rows.map(lambda row: f"首行 {row}"))
    cleaned["recall_number"] = ids.where(valid_id).astype("Int64")
    cleaned["recall_date"] = dates.dt.strftime("%Y-%m-%d")
    cleaned["year"] = dates.dt.year.astype("Int64")
    date_conflict = cleaned.groupby("recall_number")["recall_date"].transform("nunique").gt(1)
    in_period = cleaned["year"].between(FIRST_YEAR, LAST_YEAR).fillna(False)
    valid_period = valid_id & in_period & ~date_conflict
    cleaned["split"] = "review"
    cleaned.loc[valid_period & cleaned["year"].le(TRAIN_END), "split"] = "history"
    cleaned.loc[valid_period & cleaned["year"].gt(TRAIN_END), "split"] = "evaluation"
    log_issue(~valid_id, "recall_number", "编号缺失或非法", "保留来源行，暂不进入实验")
    log_issue(~in_period, "recall_date", "日期非法、缺失或超出研究期", "保留来源行，暂不进入实验")
    log_issue(date_conflict, "recall_date", "同一编号日期冲突", "核对公告后再按时间分期")

    for field in TEXT_FIELDS:
        cleaned[field] = source[field].map(lambda value: clean_text(value) or None)
        log_issue(cleaned[field].isna(), field, "文本缺失、占位或格式异常", "记为未知，保留来源行")

    for field in LABEL_FIELDS:
        aliases = HAZARD_ALIASES if field == "hazard_classification" else {}
        parsed = source[field].map(lambda value: [s.strip() for s in value if isinstance(s, str)]
                                   if isinstance(value, list) else [])
        canonical = parsed.map(lambda labels: [aliases.get(label, label) for label in labels])
        vocabulary = ALLOWED_LABELS[field]
        cleaned[field] = canonical.map(lambda labels: list(dict.fromkeys(label for label in labels
                                                                         if label in vocabulary)))
        label_list_ok = source[field].map(lambda value: isinstance(value, list) and bool(value)
                                         and all(isinstance(label, str) for label in value))
        bad_label = ~label_list_ok | canonical.map(lambda labels: any(label not in vocabulary for label in labels))
        conflict = cleaned[field].map(lambda labels: tuple(sorted(labels))).groupby(
            cleaned["recall_number"]
        ).transform("nunique").gt(1)
        log_issue(bad_label, field, "标签缺失或无法识别", "保留有效标签，异常类别待复核")
        log_issue(parsed.map(lambda labels: any(label in aliases for label in labels)),
                  field, "已知拼写异体", "统一拼写", cleaned[field].map(repr))
        log_issue(conflict, field, "同一编号的标签组合不同", "保留来源行，按公告复核")
        if field == "product_classification":
            multiple = cleaned[field].map(len).gt(1)
            cleaned["category_needs_review"] = bad_label | conflict | multiple
            log_issue(multiple, field, "同一来源行有多个产品标签", "保留多标签，按公告复核")

    quantity = pd.to_numeric(source["product_quantity"], errors="coerce")
    cleaned["quantity_unknown"] = (~(quantity.gt(0) & quantity.lt(float("inf")))
                                    | source["product_quantity"].map(lambda value: isinstance(value, bool)))
    log_issue(cleaned["quantity_unknown"], "product_quantity", "零值、缺失或非法数量", "保留原值并标记未知")
    cleaned = cleaned.loc[~duplicate].reset_index(drop=True)
    review_columns = ["source_row", "recall_number", "field", "issue", "decision", "original_value", "evidence"]
    quality_review = (pd.concat(review_entries, ignore_index=True).reindex(columns=review_columns)
                      if review_entries else pd.DataFrame(columns=review_columns))

    features = pd.DataFrame([extract_row_features(row) for row in cleaned.to_dict("records")],
                            index=cleaned.index, columns=HANDOFF_COLUMNS)
    cleaned[list(HANDOFF_COLUMNS)] = features
    cleaned["recall_year"] = cleaned["recall_year"].astype("Int64")
    assert cleaned["recall_year"].equals(cleaned["year"])
    eligible = cleaned["split"].isin(["history", "evaluation"]) & (
        cleaned["product_classification"].map(bool) & cleaned["hazard_classification"].map(bool)
    )
    handoff = cleaned.loc[
        eligible, ["source_row", *SOURCE_FIELDS, *HANDOFF_COLUMNS, "split", "category_needs_review"],
    ].copy()
    text_columns = list(TEXT_FIELDS)
    handoff[text_columns] = handoff[text_columns].fillna("")
    handoff["product_quantity"] = pd.to_numeric(handoff["product_quantity"], errors="coerce").where(
        ~cleaned.loc[eligible, "quantity_unknown"]
    ).astype(float)

    # The historical notebooks use the original hazard spelling.
    legacy_spelling = {corrected: original for original, corrected in HAZARD_ALIASES.items()}
    handoff["hazard_classification"] = handoff["hazard_classification"].map(
        lambda labels: [legacy_spelling.get(label, label) for label in labels]
    )
    validate_handoff(handoff.to_dict("records"))
    assert cleaned["source_row"].is_unique
    assert len(cleaned) + int(duplicate.sum()) == len(source)
    assert handoff.groupby("recall_number")["split"].nunique().le(1).all()
    return SourceTables(overview, field_coverage, cleaned, handoff, quality_review)


def write_source_tables(directory, tables):
    directory.mkdir(parents=True, exist_ok=True)
    tables.cleaned.to_json(directory / "preprocessed_records.json", orient="records", force_ascii=False, indent=2)
    tables.handoff.to_json(directory / "project_input_records.json", orient="records", force_ascii=False, indent=2)
    tables.quality_review.to_csv(directory / "quality_review.csv", index=False)
