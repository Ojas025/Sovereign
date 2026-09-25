"""Persistent local human-review queue."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReviewItem:
    id: str
    session_id: str
    reason: str
    content: str
    status: str = "pending"
    created_at: float = 0.0
    reviewer_note: str = ""


class ReviewStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def add(self, session_id: str, reason: str, content: str) -> ReviewItem:
        item = ReviewItem(
            uuid.uuid4().hex[:10], session_id, reason, content, created_at=time.time()
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        return item

    def list(self, status: str | None = None) -> list[ReviewItem]:
        if not self.path.exists():
            return []
        items: list[ReviewItem] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = ReviewItem(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            if status is None or item.status == status:
                items.append(item)
        return items

    def decide(self, item_id: str, decision: str, note: str = "") -> ReviewItem:
        items = self.list()
        for index, item in enumerate(items):
            if item.id == item_id:
                updated = ReviewItem(
                    item.id, item.session_id, item.reason, item.content,
                    status=decision, created_at=item.created_at, reviewer_note=note,
                )
                items[index] = updated
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(
                    "".join(json.dumps(asdict(x), ensure_ascii=False) + "\n" for x in items),
                    encoding="utf-8",
                )
                return updated
        raise KeyError(item_id)
