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
    grains: tuple[str, ...]  # ("daily",) | ("weekly",)
    # Si no está vacío, build_datasets.py genera un parquet (y luego un modelo)
    # independiente por cada combinación de valores de estas columnas, en vez de un
    # único dataset para todo el nivel -- evita datasets/entrenamientos gigantes en
    # los niveles de mayor cardinalidad (ver scripts/build_datasets.py).
    split_by: tuple[str, ...] = ()


LEVELS = [
    Level(1,  "total",      (),                                                    ("daily",)),
    Level(2,  "state",      ("state_id",),                                         ("daily",)),
    Level(3,  "cat",        ("cat_id",),                                           ("daily",)),
    Level(4,  "dept",       ("dept_id", "cat_id"),                                 ("daily",)),
    Level(5,  "state_cat",  ("cat_id", "state_id"),                                ("daily",)),
    Level(6,  "store",      ("store_id", "state_id"),                              ("daily",)),
    Level(7,  "state_dept", ("dept_id", "cat_id", "state_id"),                     ("daily",)),
    Level(8,  "store_cat",  ("cat_id", "store_id", "state_id"),                    ("daily",)),
    Level(9,  "store_dept", ("dept_id", "cat_id", "store_id", "state_id"),         ("daily",)),
    Level(10, "item",       ("dept_id", "cat_id", "item_id"),                      ("weekly", "daily")),
    Level(11, "item_state", ("dept_id", "cat_id", "item_id", "state_id"),          ("weekly", "daily")),
    Level(12, "item_store", ("dept_id", "cat_id", "item_id", "store_id", "state_id"), ("weekly", "daily"),
          ("store_id", "dept_id")),
]

LEVELS_BY_ID = {lv.id: lv for lv in LEVELS}