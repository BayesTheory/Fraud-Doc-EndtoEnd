"""
Entity: Document

Representa um documento de identidade no domínio.
Modelo puro — sem dependência de framework ou banco.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DocType(str, Enum):
    RG = "RG"
    CNH = "CNH"
    CRLV = "CRLV"
    PASSPORT = "PASSPORT"
    UNKNOWN = "UNKNOWN"


class DocStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVIEW = "REVIEW"


@dataclass
class Document:
    """Entidade de domínio: Documento."""
    id: str
    doc_type: DocType = DocType.UNKNOWN
    status: DocStatus = DocStatus.PENDING
    source: str = ""                     # ex: "mobile_app", "web_upload"
    created_at: datetime = field(default_factory=datetime.utcnow)
    image_ref: str | None = None         # referência no storage
    metadata: dict = field(default_factory=dict)
