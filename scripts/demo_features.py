"""Run the local governance features without starting an LLM server.

Usage: PYTHONPATH=src python scripts/demo_features.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from workbench.core.protocols import ChatMessage
from workbench.core.session import Session, SessionStore
from workbench.governance.pii import redact_pii
from workbench.governance.review import ReviewStore
from workbench.knowledge.local import LocalKnowledgeStore
from workbench.sovereignty.status import check_status


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sovereign-demo-") as raw:
        root = Path(raw)
        sessions = SessionStore(root / "sessions")
        session = Session.create(store=sessions, workspace=str(root))
        session.append_message(
            ChatMessage(role="user", content="Prepare an inspection summary for Pump P-104.")
        )
        session.append_message(
            ChatMessage(
                role="assistant",
                content="Pump P-104 requires inspection based on the recorded vibration threshold.",
            )
        )
        session.tier = "mid"
        session.record_turn(
            user_message="Prepare an inspection summary for Pump P-104.",
            outcome="completed",
            rounds=2,
            duration_s=1.2,
            intent="document",
        )

        source = root / "inspection_guidelines.md"
        source.write_text(
            "Pump P-104 inspection threshold is 7 mm/s. Escalate above threshold.",
            encoding="utf-8",
        )
        knowledge = LocalKnowledgeStore(root / ".knowledge")
        knowledge.ingest(source)
        hits = knowledge.search("P-104 inspection threshold")

        pii = redact_pii("Contact engineer@example.com at +91 9876543210.")
        review = ReviewStore(root / "reviews.jsonl")
        item = review.add(session.id, "high-risk", "Approve maintenance action")
        review.decide(item.id, "approve", "Demo approval")

        status = check_status(model_host="127.0.0.1")

        print("SOVEREIGN LOCAL FEATURE DEMO")
        print(f"Knowledge hits : {len(hits)}")
        print(f"PII redacted   : {pii.counts}")
        print(f"Review status  : {review.list()[0].status}")
        print(f"Offline policy : {status.status}")
        print(f"Demo workspace : {root}")
        print("All feature checks completed locally.")


if __name__ == "__main__":
    main()
