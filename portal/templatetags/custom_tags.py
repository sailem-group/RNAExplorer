from django import template
import re

register = template.Library()

@register.filter
def get_item(dictionary, key):
    if dictionary and key in dictionary:
        return dictionary.get(key)
    return None

@register.filter
def is_numeric_col(key: str) -> bool:
    if not isinstance(key, str):
        return False
    return (
        key.startswith("mnc_")
        or key.startswith("k2_")
        or key.startswith("k3_")
        or key in ("gc_pct", "gc_skew", "at_au_skew", "length")
    )