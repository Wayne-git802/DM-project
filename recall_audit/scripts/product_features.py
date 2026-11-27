"""Product text for analysis and conservative groups for evaluation splits."""

import re

import numpy as np
from sklearn.preprocessing import normalize


PRODUCT_TEXT_FIELDS = ("product_name", "visual_product_description")

# The newer notices explicitly refer to the earlier recalls on cpsc.gov.
LINKED_NOTICE_SERIES = (
    ("08-073", "17-094"),  # Carrier PTACs and heat pumps
    ("05-226", "06-030"),  # Mountain Buggy jogging strollers
)

OUTCOME_WORDS = re.compile(
    r"\b(?:hazards?|recalled?|recalls?|poses?|risks? of|unsafe|danger(?:ous)?|"
    r"overheat\w*|chok\w*|strangl\w*|poison\w*|injur\w*|lacerat\w*|"
    r"electrocut\w*|amputat\w*|entrapment|malfunction\w*|fail(?:ure|s|ed)|"
    r"detach(?:ed|es|ment)|crack(?:ed|s)|catch(?:es|ing)? fire|fire risk|"
    r"violate\w*|non[- ]compli\w*|consumer reports?|refund\w*|warranty|stop using)\b",
    re.I,
)


def normalized_official_names(records):
    names = records["official"].map(lambda item: item.get("product_name") or "")
    return names.str.casefold().str.replace(r"\W+", " ", regex=True).str.strip()


def announcement_groups(records):
    names = normalized_official_names(records)
    numbers = records["recall_number"]
    groups = ("name:" + names).where(names.ne(""), "notice:" + numbers)
    for series in LINKED_NOTICE_SERIES:
        linked_groups = groups[numbers.isin(series)]
        groups.loc[groups.isin(linked_groups)] = f"series:{series[0]}"
    return groups


def remove_outcome_sentences(variants):
    return [
        value for text in variants
        if (value := " ".join(
            sentence.strip() for sentence in re.split(r"(?<=[.!?;])\s+", text)
            if sentence.strip() and not OUTCOME_WORDS.search(sentence)
        ))
    ]


def product_text_variants(records):
    return {
        field: records["text_variants"].map(
            lambda variants: remove_outcome_sentences(variants[field])
        )
        for field in PRODUCT_TEXT_FIELDS
    }


def embed_variants(fields, model_source):
    """Encode each nonempty variant and average its unit vectors per announcement."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(min(6, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModel.from_pretrained(model_source, use_safetensors=True).eval()
    features = {}
    with torch.inference_mode():
        for field, rows in fields.items():
            unique = sorted(set().union(*rows), key=lambda value: (len(value), value))
            if not unique:
                features[field] = np.zeros((len(rows), model.config.hidden_size), dtype=np.float32)
                continue

            batches = []
            for start in range(0, len(unique), 64):
                tokens = tokenizer(unique[start:start + 64], padding=True, truncation=True,
                                   max_length=256, return_tensors="pt")
                hidden = model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                batches.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
            lookup = dict(zip(unique, np.vstack(batches), strict=True))
            vectors = np.stack([
                np.mean([lookup[variant] for variant in row], axis=0)
                if row else np.zeros(model.config.hidden_size, dtype=np.float32)
                for row in rows
            ])
            features[field] = normalize(vectors)
    return features
