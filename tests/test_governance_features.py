from pathlib import Path

from workbench.governance.pii import redact_pii
from workbench.governance.review import ReviewStore
from workbench.knowledge.local import LocalKnowledgeStore
from workbench.sovereignty.status import check_status


def test_pii_redaction() -> None:
    result = redact_pii("mail a@b.com phone +91 9876543210 IP 192.168.1.20 PAN ABCDE1234F")
    assert "[EMAIL]" in result.text
    assert "[PHONE]" in result.text
    assert "[PAN_LIKE]" in result.text
    assert "[IPV4]" in result.text
    assert result.counts["email"] == 1


def test_aadhaar_redaction() -> None:
    result = redact_pii("Aadhaar 1234 5678 9012")
    assert "[AADHAAR_LIKE]" in result.text
    assert result.counts["aadhaar_like"] == 1


def test_review_queue_roundtrip(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "reviews.jsonl")
    item = store.add("sess1", "high-risk", "delete production")
    assert store.list("pending")[0].id == item.id
    updated = store.decide(item.id, "reject", "not allowed")
    assert updated.status == "reject"
    assert store.list("pending") == []
    assert store.list("reject")[0].reviewer_note == "not allowed"


def test_local_knowledge_ingest_and_search(tmp_path: Path) -> None:
    source = tmp_path / "safety.md"
    source.write_text("Pump P-104 inspection threshold is 7 mm/s.", encoding="utf-8")
    store = LocalKnowledgeStore(tmp_path / ".knowledge")
    store.ingest(source)
    hits = store.search("P-104 inspection threshold")
    assert hits
    assert hits[0][0].name == "safety.md"


def test_sovereignty_status(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    status = check_status(model_host="127.0.0.1")
    assert status.status == "PASS"
