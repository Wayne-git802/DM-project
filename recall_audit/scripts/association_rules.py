"""Apriori counts and category-to-hazard rules for the recall notebooks."""

from itertools import product

import pandas as pd
from mlxtend.frequent_patterns import apriori


RULE_COLUMNS = ["Product category", "Hazard label", "Category notices", "Joint notices", "Support", "Category share", "Overall share", "Lift"]


def apriori_counts(transactions, minimum_joint, max_len):
    if len(transactions) < minimum_joint:
        return {}
    frequent = apriori(
        transactions, min_support=minimum_joint / len(transactions),
        use_colnames=True, max_len=max_len,
    )
    return {
        itemset: round(support * len(transactions))
        for support, itemset in frequent[["support", "itemsets"]].itertuples(index=False, name=None)
    }


def category_hazard_rules(categories, hazards, minimum_joint=30):
    transactions = pd.concat([categories.add_prefix("C:"), hazards.add_prefix("H:")], axis=1)
    counts = apriori_counts(transactions, minimum_joint, max_len=2)
    rules = []
    for category, hazard in product(categories.columns, hazards.columns):
        left, right = f"C:{category}", f"H:{hazard}"
        joint = counts.get(frozenset((left, right)), 0)
        if not joint:
            continue
        category_count = counts[frozenset((left,))]
        confidence = joint / category_count
        frequency = counts[frozenset((right,))] / len(transactions)
        rules.append((category, hazard, category_count, joint, joint / len(transactions),
                      confidence, frequency, confidence / frequency))
    return pd.DataFrame(rules, columns=RULE_COLUMNS)


def highlight_rules(rules):
    """Keep one frequently occurring, above-average specific hazard per category."""
    eligible = rules[
        rules["Hazard label"].ne("OTHER") & rules["Category share"].ge(0.20) & rules["Lift"].gt(1)
    ]
    return eligible.sort_values(
        ["Lift", "Joint notices", "Product category", "Hazard label"], ascending=[False, False, True, True],
    ).drop_duplicates("Product category")
