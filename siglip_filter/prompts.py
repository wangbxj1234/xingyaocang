"""各任务的正、负、不确定提示词。编码一次，按组取平均。"""

from __future__ import annotations

from dataclasses import dataclass

PROMPTS_VERSION = "2026-11-22.1"

ALIASES = {
    "复核包装发货1": "复核包装发货",
    "复核包装发货2": "复核包装发货",
    "复核包装发货3": "复核包装发货",
    "复合包装发货": "复核包装发货",
    "组套1": "组套",
    "组套2": "组套",
    "组套3": "组套",
    "组套-叠盒子": "组套",
    "组套-装物品": "组套",
    "组套-盖盒盖": "组套",
    "拆包销退": "销退拆包",
    "没写任务名": "通用",
    "测试": "通用",
}


@dataclass(frozen=True)
class Prompt:
    task: str
    polarity: str
    group: str
    text: str


def _rows(task: str, polarity: str, group: str, texts: list[str]) -> list[Prompt]:
    return [Prompt(task, polarity, group, text) for text in texts]


def _shared(task: str, place: str) -> list[Prompt]:
    rows = []
    rows += _rows(task, "negative", "empty_workstation", [
        f"an empty {place} with nobody there",
        f"a vacant {place} and a still camera",
        f"no person at the {place}",
    ])
    rows += _rows(task, "negative", "worker_leaving", [
        f"a worker walking away from the {place}",
        f"a person leaving the {place}",
        "the workstation is being left empty",
    ])
    rows += _rows(task, "negative", "unrelated_cleaning", [
        f"a person sweeping the floor away from the {place}",
        "wiping furniture that is not part of the task",
        "cleaning unrelated objects beside the workstation",
    ])
    rows += _rows(task, "negative", "conversation", [
        "two people standing and talking",
        "workers chatting instead of doing the task",
        "a conversation with no product in hand",
    ])
    rows += _rows(task, "negative", "unrelated", [
        "a person using a phone and not working",
        "an activity unrelated to the workstation task",
        "someone resting with no product on the table",
    ])
    rows += _rows(task, "uncertain", "observing", [
        "a worker looking at a product without moving it",
        "a person observing items on the table",
        "hands hovering while the worker studies the product",
    ])
    rows += _rows(task, "uncertain", "waiting_scan", [
        "a worker waiting to scan a barcode",
        "a person holding an item still before scanning",
        "waiting beside the scanner with a product in hand",
    ])
    rows += _rows(task, "uncertain", "standing_near", [
        f"a worker standing beside the {place}",
        f"a person standing at the {place}",
        "a worker present at the workstation but not yet moving the product",
    ])
    return rows


def _positive(task: str, groups: dict[str, list[str]]) -> list[Prompt]:
    rows = []
    for group, texts in groups.items():
        rows += _rows(task, "positive", group, texts)
    return rows


_POSITIVE = {
    "复核包装发货": {
        "product_handling": [
            "a worker handling a product at a packing workstation",
            "hands moving an item across the packing table",
            "a person picking up a packaged item",
        ],
        "barcode_scanning": [
            "a worker scanning a barcode on a package",
            "a barcode scanner aimed at a shipping label",
            "a person scanning an item before packing",
        ],
        "inspection": [
            "a worker inspecting a product before shipping",
            "a person checking an item at the packing table",
            "looking over a product to confirm the order",
        ],
        "label_checking": [
            "a worker reading a shipping label",
            "hands checking a label on a package",
            "comparing a label with the item",
        ],
        "packaging": [
            "a worker packing an item into a box",
            "a person closing a shipping box",
            "a worker applying tape to a package",
        ],
    },
    "组套": {
        "product_handling": [
            "a worker placing items into a kit box",
            "hands picking components for an order",
            "a person gathering products on a kitting table",
        ],
        "barcode_scanning": [
            "a worker scanning a barcode while kitting an order",
            "a scanner reading a product during kitting",
            "checking a barcode before placing an item in the kit",
        ],
        "inspection": [
            "a worker checking that the kit items match the order",
            "inspecting components before they go into the box",
            "a person verifying items laid out for a kit",
        ],
        "label_checking": [
            "a worker reading the kit label",
            "checking a label on a kitting box",
            "matching a printed list to the items",
        ],
        "packaging": [
            "a worker folding a box for a kit",
            "placing the last item into the kit box",
            "closing the kit box after the items are inside",
        ],
    },
    "销退拆包": {
        "product_handling": [
            "a worker opening a returned parcel",
            "hands taking a product out of a return package",
            "unpacking returned goods on the table",
        ],
        "barcode_scanning": [
            "scanning a barcode on a returned item",
            "a worker scanning the return label",
            "reading a barcode while unpacking a return",
        ],
        "inspection": [
            "inspecting a returned product",
            "a worker checking the condition of a return",
            "looking at an unpacked return item",
        ],
        "label_checking": [
            "reading the label on a returned package",
            "checking the return label against the item",
            "a worker examining a shipping label on a return",
        ],
        "packaging": [
            "emptying a returned box onto the workstation",
            "removing packing material from a return",
            "setting returned products down after opening the parcel",
        ],
    },
    "衣服质检": {
        "product_handling": [
            "a worker handling a garment on the inspection table",
            "hands spreading a piece of clothing",
            "picking up clothing for a quality check",
        ],
        "barcode_scanning": [
            "scanning the barcode on a garment tag",
            "a worker scanning a clothing label",
            "reading a hang-tag barcode",
        ],
        "inspection": [
            "inspecting a garment for defects",
            "a worker checking the quality of clothing",
            "looking closely at the fabric of a shirt",
        ],
        "label_checking": [
            "reading the care label on clothing",
            "checking the size label of a garment",
            "a worker examining a clothing tag",
        ],
        "packaging": [
            "folding a garment after inspection",
            "placing checked clothing into a pile",
            "setting a finished garment aside",
        ],
    },
    "扫码": {
        "product_handling": [
            "a worker holding a product up to scan it",
            "hands moving an item toward a scanner",
            "picking up the next item to scan",
        ],
        "barcode_scanning": [
            "a worker scanning a barcode",
            "a barcode scanner aimed at a product",
            "scanning items one after another",
        ],
        "inspection": [
            "checking a product while scanning it",
            "a worker looking at an item after the scan",
            "confirming the scanned product",
        ],
        "label_checking": [
            "reading a barcode label",
            "finding the barcode on a package",
            "checking that the label is the one being scanned",
        ],
        "packaging": [
            "moving a scanned item into a bin",
            "placing a product down after scanning",
            "stacking scanned packages",
        ],
    },
    "分拣": {
        "product_handling": [
            "a worker sorting packages on a table",
            "hands moving a parcel into a bin",
            "picking up a package to sort it",
        ],
        "barcode_scanning": [
            "scanning a package before sorting it",
            "a worker reading a barcode while sorting",
            "a scanner aimed at a parcel in a sorting station",
        ],
        "inspection": [
            "checking which bin a package belongs to",
            "a worker looking at a parcel while sorting",
            "inspecting a package during sorting",
        ],
        "label_checking": [
            "reading the destination label while sorting",
            "checking a shipping label at a sort station",
            "comparing a label to the sorting bins",
        ],
        "packaging": [
            "placing a sorted package into a tote",
            "stacking parcels after sorting",
            "moving a finished stack of sorted packages",
        ],
    },
    "销退质检": {
        "product_handling": [
            "a worker handling a returned product for inspection",
            "hands turning a returned item over",
            "picking up a return to check it",
        ],
        "barcode_scanning": [
            "scanning a returned product barcode",
            "a worker scanning an item during return inspection",
            "reading the barcode of a returned product",
        ],
        "inspection": [
            "inspecting the quality of a returned product",
            "checking a return for damage",
            "a worker examining a returned item closely",
        ],
        "label_checking": [
            "reading the label on a returned product",
            "checking the tag of an item being inspected",
            "comparing a return label with the product",
        ],
        "packaging": [
            "setting an inspected return into a pass or fail pile",
            "placing a checked return into a bin",
            "bagging a product after return inspection",
        ],
    },
    "通用": {
        "product_handling": [
            "a worker handling a product at a workstation",
            "hands moving an item on the work table",
            "a person picking up a product",
        ],
        "barcode_scanning": [
            "a worker scanning a barcode",
            "a barcode scanner aimed at a label",
            "scanning a product at the workstation",
        ],
        "inspection": [
            "a worker inspecting a product",
            "a person checking an item on the table",
            "looking closely at a product",
        ],
        "label_checking": [
            "a worker reading a label",
            "hands checking a label on a product",
            "examining a printed label",
        ],
        "packaging": [
            "a worker packing a product",
            "placing an item into a box",
            "closing a box at the workstation",
        ],
    },
}

_PLACES = {
    "复核包装发货": "packing workstation",
    "组套": "kitting workstation",
    "销退拆包": "returns unpacking table",
    "衣服质检": "garment inspection table",
    "扫码": "scanning station",
    "分拣": "sorting station",
    "销退质检": "return inspection table",
    "通用": "workstation",
}


def canonical_task(task: str) -> str:
    name = ALIASES.get(task, task)
    if name not in _POSITIVE:
        return "通用"
    return name


def prompts_for(task: str) -> list[Prompt]:
    name = canonical_task(task)
    return _positive(name, _POSITIVE[name]) + _shared(name, _PLACES[name])
