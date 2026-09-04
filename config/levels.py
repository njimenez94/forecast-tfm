"""Niveles de agregación M5 (L1 total → L12 item-store).

Cada nivel define por qué dimensiones se agrupa y su granularidad temporal.
scripts/create_dataset.py recorre LEVELS para generar un parquet por nivel.
"""
from dataclasses import dataclass, field


@dataclass
class Level:
    id: int
    name: str
    dims: tuple[str, ...]
    grains: tuple[str, ...]  # ("daily",) | ("weekly",)
    split_by: tuple[str, ...] = ()
    # Filtro opcional por dept_id/cat_id/state_id/store_id, p.ej. {"dept_id": ("FOODS_3",)}.
    # None/{} = sin filtrar (comportamiento normal). Útil en fase experimental para no
    # tener que materializar los ~70 datasets de un nivel denso (item/item_store) y
    # construir solo los de un dept/store/state concreto (scripts/process_data.py).
    filters: dict[str, tuple[str, ...]] = field(default_factory=dict)


LEVELS = [
    Level(1,  "total",      (),                                                             ("daily",)),
    Level(2,  "state",      ("state_id",),                                                  ("daily",)),
    Level(3,  "cat",        ("cat_id",),                                                    ("daily",)),
    Level(4,  "dept",       ("dept_id", "cat_id"),                                          ("daily",)),
    Level(5,  "state_cat",  ("cat_id", "state_id"),                                         ("daily",)),
    Level(6,  "store",      ("store_id", "state_id"),                                       ("daily",)),
    Level(7,  "state_dept", ("dept_id", "cat_id", "state_id"),                              ("daily",)),
    Level(8,  "store_cat",  ("cat_id", "store_id", "state_id"),                             ("daily",)),
    Level(9,  "store_dept", ("dept_id", "cat_id", "store_id", "state_id"),                  ("daily",)),
    Level(10, "item",       ("dept_id", "cat_id", "item_id"),                               ("daily",), ("dept_id",),
          filters={"dept_id": ("FOODS_3",)}),
    Level(11, "item_state", ("dept_id", "cat_id", "item_id", "state_id"),                   ("daily",), ("state_id", "dept_id"),
          filters={"dept_id": ("FOODS_3",), "state_id": ("CA",)}),
    Level(12, "item_store", ("dept_id", "cat_id", "item_id", "store_id", "state_id"),       ("daily",), ("store_id", "dept_id"),
          filters={"dept_id": ("FOODS_3",), "store_id": ("CA_3",)}),
]

LEVELS_BY_ID = {lv.id: lv for lv in LEVELS}

# Niveles activos por defecto (sin --levels) en process_data/build_datasets/train_dataset.
# En experimentación: fuera los niveles densos (10-12, item-level). Para activarlos todos:
# ACTIVE_LEVEL_IDS = tuple(LEVELS_BY_ID)
ACTIVE_LEVEL_IDS: tuple[int, ...] = (1, 4, 6, 9, 10, 12) # tuple(LEVELS_BY_ID)
ACTIVE_LEVELS = [lv for lv in LEVELS if lv.id in ACTIVE_LEVEL_IDS]