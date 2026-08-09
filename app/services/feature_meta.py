"""Human-readable metadata and value formatters for model features.

Internal representations (feature codes like ``PAY_0``, repayment-status
enums, raw SHAP values) are translated into domain language here so that
both the API responses and the LLM prompts stay self-explanatory.
"""


def format_currency(value: float) -> str:
    """Format a monetary value with thousands separators."""
    return f"{value:,.0f}"


def format_pay_status(value: float) -> str:
    """Translate a repayment-status code into plain language.

    The PAY_* features encode -2 (no usage) through 8+ (months late).
    """
    mapping = {
        -2: "No usage that month",
        -1: "Paid in full",
        0: "Minimum payment",
        1: "1 month late",
        2: "2 months late",
        3: "3 months late",
    }
    v = int(round(value))
    if v in mapping:
        return mapping[v]
    if v > 3:
        return f"{v} months late"
    return str(value)


def describe_magnitude(shap_value: float) -> str:
    """Convert a SHAP value into a plain-language strength description.

    SHAP values are in log-odds units, so we bucket by absolute value
    and combine with direction.
    """
    abs_v = abs(shap_value)
    if abs_v >= 0.30:
        strength = "Very strongly"
    elif abs_v >= 0.15:
        strength = "Strongly"
    elif abs_v >= 0.05:
        strength = "Moderately"
    else:
        strength = "Slightly"
    direction = "increases risk" if shap_value >= 0 else "decreases risk"
    return f"{strength} {direction}"


def humanize_value(feature_name: str, value: float) -> str:
    """Format a raw feature value into a human-readable string.

    Args:
        feature_name: The model feature name (e.g. ``PAY_0``).
        value: The raw value the applicant provided.

    Returns:
        A plain-language interpretation of the value.
    """
    meta = FEATURE_META.get(feature_name)
    if meta is not None and "formatter" in meta:
        return meta["formatter"](value)
    return str(value)


FEATURE_META = {
    "LIMIT_BAL": {
        "label": "Credit Limit",
        "formatter": format_currency,
    },
    "AGE": {
        "label": "Age",
        "formatter": lambda v: f"{int(v)} years",
    },
    "PAY_0": {
        "label": "Repayment Status (Last Month)",
        "formatter": format_pay_status,
    },
    "PAY_2": {
        "label": "Repayment Status (2 Months Ago)",
        "formatter": format_pay_status,
    },
    "PAY_3": {
        "label": "Repayment Status (3 Months Ago)",
        "formatter": format_pay_status,
    },
    "PAY_AMT1": {
        "label": "Last Payment Amount",
        "formatter": format_currency,
    },
    "BILL_AMT1": {
        "label": "Last Bill Amount",
        "formatter": format_currency,
    },
}
