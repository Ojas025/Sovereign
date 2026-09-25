"""Lightweight markdown parser for document generation (Word/PDF).

Extracts structural blocks (headings, paragraphs, code blocks, lists, tables,
horizontal rules) and inline formatting spans (bold, italic, inline code)
from model-generated markdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InlineSpan:
    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False


@dataclass(frozen=True, slots=True)
class Heading:
    level: int  # 1 to 6
    text: str


@dataclass(frozen=True, slots=True)
class Paragraph:
    text: str


@dataclass(frozen=True, slots=True)
class CodeBlock:
    language: str
    code: str


@dataclass(frozen=True, slots=True)
class ListBlock:
    ordered: bool
    items: list[str]


@dataclass(frozen=True, slots=True)
class Table:
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True, slots=True)
class HorizontalRule:
    pass


Block = Heading | Paragraph | CodeBlock | ListBlock | Table | HorizontalRule


_HR_PATTERN = re.compile(r"^ {0,3}([-*_])\s*(?:\1\s*){2,}$")
_HEADING_PATTERN = re.compile(r"^ {0,3}(#{1,6})\s+(.*)$")
_ORDERED_LIST_PATTERN = re.compile(r"^ {0,3}\d+[.)]\s+(.*)$")
_UNORDERED_LIST_PATTERN = re.compile(r"^ {0,3}[-*+]\s+(.*)$")
_TABLE_ROW_PATTERN = re.compile(r"^ {0,3}\|.*\|\s*$")
_TABLE_DIVIDER_PATTERN = re.compile(r"^ {0,3}\|(?:\s*:?-+:?\s*\|)+\s*$")


def parse_inlines(text: str) -> list[InlineSpan]:
    """Parse text into styled spans (bold, italic, code)."""
    # Tokenize bold (**text** or __text__), italic (*text* or _text_), code (`text`)
    pattern = re.compile(
        r"(\*\*(?P<bold_star>.+?)\*\*|__(?P<bold_under>.+?)__|"
        r"`(?P<code>[^`]+)`|"
        r"\*(?P<italic_star>[^*]+)\*|_(?P<italic_under>[^_]+)_)"
    )
    spans: list[InlineSpan] = []
    last_end = 0

    for match in pattern.finditer(text):
        start, end = match.span()
        if start > last_end:
            spans.append(InlineSpan(text=text[last_end:start]))

        if match.group("bold_star") is not None or match.group("bold_under") is not None:
            val = match.group("bold_star") or match.group("bold_under") or ""
            spans.append(InlineSpan(text=val, bold=True))
        elif match.group("code") is not None:
            val = match.group("code")
            spans.append(InlineSpan(text=val, code=True))
        elif match.group("italic_star") is not None or match.group("italic_under") is not None:
            val = match.group("italic_star") or match.group("italic_under") or ""
            spans.append(InlineSpan(text=val, italic=True))
        last_end = end

    if last_end < len(text):
        spans.append(InlineSpan(text=text[last_end:]))

    return spans if spans else [InlineSpan(text=text)]


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def parse_markdown(text: str) -> list[Block]:
    """Parse a markdown string into structural blocks."""
    lines = text.splitlines()
    blocks: list[Block] = []
    idx = 0
    total = len(lines)

    while idx < total:
        line = lines[idx]

        # 1. Fenced code block
        if line.strip().startswith("```"):
            fence = line.strip()[:3]
            lang = line.strip()[3:].strip()
            idx += 1
            code_lines: list[str] = []
            while idx < total and not lines[idx].strip().startswith(fence):
                code_lines.append(lines[idx])
                idx += 1
            if idx < total:
                idx += 1  # consume closing fence
            blocks.append(CodeBlock(language=lang, code="\n".join(code_lines)))
            continue

        # 2. Blank line
        if not line.strip():
            idx += 1
            continue

        # 3. Horizontal rule
        if _HR_PATTERN.match(line):
            blocks.append(HorizontalRule())
            idx += 1
            continue

        # 4. Heading
        heading_match = _HEADING_PATTERN.match(line)
        if heading_match:
            hashes, title = heading_match.groups()
            blocks.append(Heading(level=len(hashes), text=title.strip()))
            idx += 1
            continue

        # 5. Table (must have header, separator line, and optional rows)
        if (
            _TABLE_ROW_PATTERN.match(line)
            and idx + 1 < total
            and _TABLE_DIVIDER_PATTERN.match(lines[idx + 1])
        ):
            headers = _split_table_row(line)
            idx += 2  # skip header and divider
            rows: list[list[str]] = []
            while idx < total and _TABLE_ROW_PATTERN.match(lines[idx]):
                rows.append(_split_table_row(lines[idx]))
                idx += 1
            blocks.append(Table(headers=headers, rows=rows))
            continue

        # 6. Unordered list
        unordered_match = _UNORDERED_LIST_PATTERN.match(line)
        if unordered_match:
            items: list[str] = [unordered_match.group(1).strip()]
            idx += 1
            while idx < total:
                next_match = _UNORDERED_LIST_PATTERN.match(lines[idx])
                if next_match:
                    items.append(next_match.group(1).strip())
                    idx += 1
                elif lines[idx].strip() == "":
                    # Peek next line to see if list continues
                    if idx + 1 < total and _UNORDERED_LIST_PATTERN.match(lines[idx + 1]):
                        idx += 1
                    else:
                        break
                else:
                    break
            blocks.append(ListBlock(ordered=False, items=items))
            continue

        # 7. Ordered list
        ordered_match = _ORDERED_LIST_PATTERN.match(line)
        if ordered_match:
            items = [ordered_match.group(1).strip()]
            idx += 1
            while idx < total:
                next_match = _ORDERED_LIST_PATTERN.match(lines[idx])
                if next_match:
                    items.append(next_match.group(1).strip())
                    idx += 1
                elif lines[idx].strip() == "":
                    if idx + 1 < total and _ORDERED_LIST_PATTERN.match(lines[idx + 1]):
                        idx += 1
                    else:
                        break
                else:
                    break
            blocks.append(ListBlock(ordered=True, items=items))
            continue

        # 8. Paragraph (accumulate consecutive non-empty lines until blank line or special block)
        para_lines: list[str] = [line.strip()]
        idx += 1
        while idx < total:
            next_line = lines[idx]
            if not next_line.strip():
                break
            if (
                next_line.strip().startswith("```")
                or _HR_PATTERN.match(next_line)
                or _HEADING_PATTERN.match(next_line)
                or _UNORDERED_LIST_PATTERN.match(next_line)
                or _ORDERED_LIST_PATTERN.match(next_line)
                or (
                    _TABLE_ROW_PATTERN.match(next_line)
                    and idx + 1 < total
                    and _TABLE_DIVIDER_PATTERN.match(lines[idx + 1])
                )
            ):
                break
            para_lines.append(next_line.strip())
            idx += 1

        blocks.append(Paragraph(text=" ".join(para_lines)))

    return blocks
