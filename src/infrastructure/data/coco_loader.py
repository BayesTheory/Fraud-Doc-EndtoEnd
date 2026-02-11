"""
COCO Dataset Loader for MIDV-2020 MRP passports.

Loads images + annotations from Roboflow COCO export,
groups bounding boxes by image, and provides structured
access to each passport field region for OCR + Rules.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Category mapping (from COCO annotations) ────────────────────────
# These are the field IDs we care about (skip captions, they're labels)
PASSPORT_FIELD_IDS = {
    1: "date_of_birth",
    3: "date_of_expiry",
    5: "date_of_issue",
    8: "document_code",
    10: "document_number",
    12: "face_image",
    13: "issue_authority",
    15: "issuing_state_code",
    17: "issuing_state_full",
    18: "mrz_lower_line",
    19: "mrz_upper_line",
    20: "nationality",
    22: "personal_number",
    24: "place_of_birth",
    26: "primary_identifier",
    28: "secondary_identifier",
    30: "sex",
    32: "signature",
}

# Caption IDs (we skip these — they're the label text, not field values)
CAPTION_IDS = {2, 4, 6, 7, 9, 11, 14, 16, 21, 23, 25, 27, 29, 31, 33}


@dataclass
class FieldRegion:
    """A bounding box region for a single passport field."""
    field_name: str
    category_id: int
    bbox: Tuple[float, float, float, float]  # x, y, w, h (COCO format)
    area: float

    @property
    def x(self) -> float:
        return self.bbox[0]

    @property
    def y(self) -> float:
        return self.bbox[1]

    @property
    def width(self) -> float:
        return self.bbox[2]

    @property
    def height(self) -> float:
        return self.bbox[3]

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Convert COCO [x,y,w,h] to [x1,y1,x2,y2] for cropping."""
        x1 = int(self.x)
        y1 = int(self.y)
        x2 = int(self.x + self.width)
        y2 = int(self.y + self.height)
        return (x1, y1, x2, y2)


@dataclass
class PassportSample:
    """A single passport image with all its annotated field regions."""
    image_id: int
    file_name: str
    original_name: str
    width: int
    height: int
    country_code: str  # e.g. "aze", "grc", "lva", "srb"
    fields: Dict[str, List[FieldRegion]] = field(default_factory=dict)

    @property
    def image_path(self) -> str:
        """Will be set relative to the split directory."""
        return self.file_name

    @property
    def has_mrz(self) -> bool:
        return "mrz_upper_line" in self.fields and "mrz_lower_line" in self.fields

    @property
    def field_names(self) -> List[str]:
        return list(self.fields.keys())


@dataclass
class COCODataset:
    """Loaded COCO dataset with all passport samples."""
    split: str  # "train", "valid", "test"
    base_dir: str
    samples: List[PassportSample] = field(default_factory=list)
    categories: Dict[int, str] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def get_by_country(self, country_code: str) -> List[PassportSample]:
        return [s for s in self.samples if s.country_code == country_code]

    @property
    def countries(self) -> List[str]:
        return sorted(set(s.country_code for s in self.samples))

    def stats(self) -> Dict:
        """Return dataset statistics."""
        by_country = {}
        for s in self.samples:
            by_country[s.country_code] = by_country.get(s.country_code, 0) + 1

        field_counts = {}
        for s in self.samples:
            for fname in s.fields:
                field_counts[fname] = field_counts.get(fname, 0) + 1

        return {
            "split": self.split,
            "total_images": self.num_samples,
            "by_country": by_country,
            "field_coverage": field_counts,
            "countries": self.countries,
        }


def _extract_country(filename: str) -> str:
    """Extract country code from filename like 'midv2020-aze-passport_01.jpg'."""
    parts = filename.split("-")
    if len(parts) >= 2:
        return parts[1].lower()
    return "unknown"


def load_coco_split(data_dir: str, split: str = "train") -> COCODataset:
    """
    Load a COCO split from the MIDV-2020 dataset.

    Args:
        data_dir: Path to 'data/raw/' directory
        split: One of 'train', 'valid', 'test'

    Returns:
        COCODataset with all parsed passport samples
    """
    split_dir = os.path.join(data_dir, split)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"COCO annotations not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # ── Parse categories ──
    categories = {}
    for cat in coco.get("categories", []):
        categories[cat["id"]] = cat["name"]

    # ── Parse images ──
    image_map: Dict[int, PassportSample] = {}
    for img in coco.get("images", []):
        img_id = img["id"]
        fname = img["file_name"]
        orig_name = img.get("extra", {}).get("name", fname)
        country = _extract_country(orig_name)

        sample = PassportSample(
            image_id=img_id,
            file_name=fname,
            original_name=orig_name,
            width=img["width"],
            height=img["height"],
            country_code=country,
        )
        image_map[img_id] = sample

    # ── Parse annotations ──
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]

        # Skip captions and the root category
        if cat_id in CAPTION_IDS or cat_id == 0:
            continue

        # Only process known passport fields
        field_name = PASSPORT_FIELD_IDS.get(cat_id)
        if field_name is None:
            continue

        if img_id not in image_map:
            continue

        region = FieldRegion(
            field_name=field_name,
            category_id=cat_id,
            bbox=tuple(ann["bbox"]),
            area=ann.get("area", 0),
        )

        sample = image_map[img_id]
        if field_name not in sample.fields:
            sample.fields[field_name] = []
        sample.fields[field_name].append(region)

    dataset = COCODataset(
        split=split,
        base_dir=split_dir,
        samples=list(image_map.values()),
        categories=categories,
    )

    return dataset


def load_all_splits(data_dir: str) -> Dict[str, COCODataset]:
    """Load train, valid, and test splits."""
    splits = {}
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            splits[split] = load_coco_split(data_dir, split)
    return splits


# ── CLI usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    for split in ["train", "valid", "test"]:
        try:
            ds = load_coco_split(data_dir, split)
            stats = ds.stats()
            print(f"\n{'='*50}")
            print(f"Split: {stats['split']}")
            print(f"Total images: {stats['total_images']}")
            print(f"Countries: {stats['countries']}")
            print(f"By country: {stats['by_country']}")
            print(f"Field coverage:")
            for field_name, count in sorted(stats['field_coverage'].items()):
                print(f"  {field_name}: {count}/{stats['total_images']}")
        except FileNotFoundError as e:
            print(f"Skipping {split}: {e}")
