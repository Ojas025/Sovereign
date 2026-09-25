"""Workbench routing questions — evaluated in one batched laya forward pass.

``difficulty`` keeps laya's stock router instructions (trained legend, so the
checkpoint stays in its calibrated distribution). ``needs_tools`` is rewritten
for the workbench domain: the stock question asks about *external* tools and
scored a real file-edit request 0.076, versus 0.831 with our wording.
``intent`` is authored for the seven workbench intents (plan §4.1).
"""

from __future__ import annotations

INTENT_CRITERIA: dict[str, str] = {
    "chat_qa": "answering a question or casual conversation, no work artifacts",
    "file_edit": "modifying existing files or text",
    "code_gen": "writing new code from scratch",
    "shell_task": "running shell commands, git, installs, system management",
    "search_analysis": "searching or analysing the codebase or data",
    "planning": "designing a plan or deciding steps before acting",
    "meta": "questions about the assistant itself or its capabilities",
}

# Ordinal 0-3 legend (trivial/easy/moderate/hard), verbatim from laya's preset.
DIFFICULTY_CRITERIA: list[str] = [
    "trivial: a lookup or one-liner",
    "easy: short answer, no reasoning",
    "moderate: several steps",
    "hard: long multi-step reasoning or specialist knowledge",
]

WORKBENCH_QUESTIONS: dict[str, dict[str, object]] = {
    "intent": {
        "type": "choice",
        "instructions": (
            "What is the user trying to do in this terminal workbench session?"
        ),
        "criteria": INTENT_CRITERIA,
    },
    "difficulty": {
        "type": "score",
        "instructions": "How hard is `request` for a language model?",
        "criteria": DIFFICULTY_CRITERIA,
    },
    "needs_tools": {
        "type": "noul",
        "instructions": (
            "Does completing this request require editing files, writing code "
            "or running shell commands?"
        ),
    },
    "has_pii": {
        "type": "noul",
        "instructions": (
            "Does `request` contain personally identifiable information (PII) such as "
            "email addresses, phone numbers, IP addresses, government IDs, "
            "or personal contact details?"
        ),
    },
}
