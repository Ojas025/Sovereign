"""Dependency-free local text knowledge store for demos and offline use."""

from __future__ import annotations

import builtins
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    name: str
    path: str
    text: str


class LocalKnowledgeStore:
    """Small lexical retriever; deliberately local and deterministic."""

    EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".log"}

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def ingest(self, path: str | Path) -> KnowledgeDocument:
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() not in self.EXTENSIONS:
            raise ValueError(f"unsupported knowledge file: {source.suffix or '(no extension)'}")
        text = source.read_text(encoding="utf-8", errors="replace")
        doc = KnowledgeDocument(source.name, str(source), text)
        documents = {item.path: item for item in self.list()}
        documents[doc.path] = doc
        self._save(documents.values())
        return doc

    def list(self) -> builtins.list[KnowledgeDocument]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return [KnowledgeDocument(**item) for item in data]
        except (json.JSONDecodeError, TypeError):
            return []

    def search(self, query: str, limit: int = 5) -> builtins.list[tuple[KnowledgeDocument, float]]:
        terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
        if not terms:
            return []
        scored: list[tuple[KnowledgeDocument, float]] = []
        for doc in self.list():
            tokens = re.findall(r"[a-z0-9_]+", doc.text.lower())
            if not tokens:
                continue
            counts = {token: tokens.count(token) for token in terms}
            score = sum(1 for value in counts.values() if value) / len(terms)
            if score:
                scored.append((doc, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].name))[:limit]

    def _save(self, documents: Iterable[KnowledgeDocument]) -> None:
        payload = [asdict(doc) for doc in documents]
        self.index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
