"""Niveles de agregación M5 (L1 total → L12 item-store).

Cada nivel define por qué dimensiones se agrupa y su granularidad temporal.
scripts/create_dataset.py recorre LEVELS para generar un parquet por nivel.
"""
from dataclasses import dataclass


@dataclass
class Level:
    id: int
    name: str
    dims: tuple[str, ...]
    grains: tuple[str, ...]  # ("daily", "weekly") | ("weekly",) | ("daily", "weekly")


LEVELS = [
    Level(1,  "total",      (),                                                    ("daily", "weekly")),
    Level(2,  "state",      ("state_id",),                                         ("daily", "weekly")),
    Level(3,  "cat",        ("cat_id",),                                           ("daily", "weekly")),
    Level(4,  "dept",       ("dept_id", "cat_id"),                                 ("daily", "weekly")),
    Level(5,  "store",      ("state_id","store_id"),                               ("daily", "weekly")),
    Level(6,  "state_cat",  ("cat_id", "state_id"),                                ("daily", "weekly")),
    Level(7,  "state_dept", ("dept_id", "cat_id", "state_id"),                     ("daily", "weekly")),
    Level(8,  "store_cat",  ("cat_id", "state_id", "store_id"),                    ("daily", "weekly")),
    Level(9,  "store_dept", ("dept_id", "cat_id", "store_id", "state_id"),         ("daily", "weekly")),
    Level(10, "item",       ("dept_id", "cat_id", "item_id"),                      ("daily", "weekly")),
    Level(11, "item_state", ("dept_id", "cat_id", "item_id", "state_id"),          ("daily", "weekly")),
    Level(12, "item_store", ("dept_id", "cat_id", "item_id", "state_id", "store_id"), ("daily", "weekly")),
]

LEVELS_BY_ID = {lv.id: lv for lv in LEVELS}