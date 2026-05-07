"""
database.py — Shared product catalog
=====================================
Every module reads/writes products through this file.
No module should touch products.json directly.

Usage:
    from database import get_all_products, get_product, update_product
"""

import json
import os
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "products.json")


def get_all_products() -> list[dict]:
    """Return the full product catalog."""
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_product(product_id: str) -> Optional[dict]:
    """Return a single product by ID, or None if not found."""
    products = get_all_products()
    for p in products:
        if p["id"] == product_id:
            return p
    return None


def update_product(product_id: str, fields: dict) -> bool:
    """
    Merge `fields` into the product with the given ID.
    Returns True on success, False if product not found.

    Example:
        update_product("prod_001", {"category": "electronics", "alt_text": "..."})
    """
    products = get_all_products()
    for p in products:
        if p["id"] == product_id:
            p.update(fields)
            _save(products)
            return True
    return False


def add_product(product: dict) -> bool:
    """
    Add a new product to the catalog.
    The product dict must have at least: id, name, image_path, price.
    All AI fields default to None if not provided.
    """
    defaults = {
        "category": None,
        "alt_text": None,
        "sentiment_score": None,
        "sentiment_label": None,
        "image_ok": None,
        "reviews": [],
    }
    full = {**defaults, **product}
    products = get_all_products()
    products.append(full)
    _save(products)
    return True


def _save(products: list[dict]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
