"""
PDF ingestion pipeline for Indian Legal Documents.
Extracts text + rich metadata (act name, year, chapter, section) from PDFs.
"""

import os
import re
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document


# ── Metadata heuristics ────────────────────────────────────────────────────

ACT_META = {
    "copyright act": {
        "act_name": "The Copyright Act, 1957",
        "year": "1957",
        "domain": "Intellectual Property",
    },
    "themuslimwomen": {
        "act_name": "The Muslim Women (Protection of Rights on Marriage) Act, 2019",
        "year": "2019",
        "domain": "Personal Law",
    },
    "tribunal act": {
        "act_name": "The Tribunals Reforms Act, 2021",
        "year": "2021",
        "domain": "Judicial Reform",
    },
    # ── New ───────────────────────────────────────────────────────
    "farm laws repeal": {
        "act_name": "The Farm Laws Repeal Act, 2021",
        "year": "2021",
        "domain": "Agricultural Law",
    },
    "citizenship": {
        "act_name": "The Citizenship (Amendment) Act, 2019",
        "year": "2019",
        "domain": "Citizenship Law",
    },
}

# Regex patterns for structural markers
SECTION_RE = re.compile(
    r"(?:^|\n)\s*(\d+[A-Z]?)\.\s+([A-Z][^\n]{3,80})", re.MULTILINE
)
CHAPTER_RE = re.compile(
    r"(?:^|\n)\s*CHAPTER\s+([IVXLC\d]+)[.\s—–-]*([A-Z][^\n]{0,80})",
    re.MULTILINE | re.IGNORECASE,
)
# Schedules: "THE FIRST SCHEDULE", "THE SECOND SCHEDULE", "THE SCHEDULE", etc.
# Matches both "THE FIRST SCHEDULE" and a bare "THE SCHEDULE" at start of line.
SCHEDULE_RE = re.compile(
    r"(?:^|\n)\s*THE\s+(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|)\s*SCHEDULE\b",
    re.MULTILINE,
)
# Ordinal label → numeric id for cleaner section_id metadata
_ORDINAL_TO_NUM = {
    "FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4", "FIFTH": "5",
    "SIXTH": "6", "SEVENTH": "7", "EIGHTH": "8", "NINTH": "9", "TENTH": "10",
    "": "1",  # bare "THE SCHEDULE" — treat as Schedule 1
}


def _detect_act_meta(filename: str) -> Dict[str, str]:
    """Map filename to act metadata."""
    fname_lower = filename.lower().replace(" ", "").replace("_", "")
    for key, meta in ACT_META.items():
        if key.replace(" ", "") in fname_lower:
            return meta
    return {
        "act_name": Path(filename).stem.title(),
        "year": "unknown",
        "domain": "Law",
    }


def _extract_sections(text: str) -> List[Dict[str, str]]:
    """
    Split document text into sections with metadata.
    Returns list of {section_id, section_title, text} dicts.
    """
    # Find chapter boundaries
    chapter_matches = list(CHAPTER_RE.finditer(text))
    section_matches = list(SECTION_RE.finditer(text))

    if not section_matches:
        # No section markers found — return whole text as one chunk
        return [{"section_id": "1", "section_title": "Full Text", "text": text}]

    sections = []
    for i, match in enumerate(section_matches):
        start = match.start()
        end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(text)
        section_text = text[start:end].strip()

        if len(section_text) < 30:
            continue

        # Determine which chapter this section falls under
        chapter_id = "1"
        chapter_title = "General"
        for cm in chapter_matches:
            if cm.start() <= start:
                chapter_id = cm.group(1)
                chapter_title = cm.group(2).strip() if cm.group(2) else "General"

        sections.append(
            {
                "section_id": match.group(1),
                "section_title": match.group(2).strip(),
                "chapter_id": chapter_id,
                "chapter_title": chapter_title,
                "text": section_text,
            }
        )

    return sections if sections else [{"section_id": "1", "section_title": "Full Text", "text": text}]


def _extract_schedules(text: str) -> List[Dict[str, str]]:
    """
    Extract Schedule blocks (e.g. lists of abolished tribunals, salary tables).
    Returns list of {section_id: 'Schedule-N', section_title, text} dicts.
    Schedules begin with 'THE FIRST SCHEDULE' / 'THE SCHEDULE' and run to the
    next schedule heading or end of document.
    """
    all_matches = list(SCHEDULE_RE.finditer(text))
    if not all_matches:
        return []

    # The Table of Contents at the top of an act often lists schedules by name
    # (e.g. "THE FIRST SCHEDULE\nTHE SECOND SCHEDULE"). The real schedules sit
    # at the END of the document. Dedupe by keeping only the LAST match per
    # ordinal — that's the substantive one, not the TOC reference.
    by_ord: Dict[str, "re.Match[str]"] = {}
    for m in all_matches:
        ord_key = (m.group(1) or "").upper()
        by_ord[ord_key] = m  # later matches overwrite earlier ones
    matches = sorted(by_ord.values(), key=lambda m: m.start())

    schedules = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        if len(block) < 50:
            continue  # likely a stray TOC reference, not the actual schedule

        ordinal = (m.group(1) or "").upper()
        sched_num = _ORDINAL_TO_NUM.get(ordinal, str(i + 1))

        # First non-empty line after the heading is usually the title
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        title = lines[1] if len(lines) > 1 else f"Schedule {sched_num}"
        # Trim very long titles
        title = title[:120]

        schedules.append(
            {
                "section_id": f"Schedule-{sched_num}",
                "section_title": title,
                "chapter_id": "Schedule",
                "chapter_title": "Schedule",
                "text": block,
            }
        )
    return schedules


def load_pdf_as_documents(pdf_path: str) -> List[Document]:
    """
    Load a single PDF and return a list of LangChain Documents,
    one per section, each enriched with metadata.
    """
    path = Path(pdf_path)
    act_meta = _detect_act_meta(path.name)

    # Extract full text via pdfplumber
    full_text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text_pages.append((page_num, text))

    full_text = "\n".join(t for _, t in full_text_pages)

    # Split into sections + schedules (schedules sit after the last section)
    sections = _extract_sections(full_text)
    schedules = _extract_schedules(full_text)
    if schedules:
        print(f"  → {len(schedules)} schedule(s) extracted")

    docs = []
    for sec in sections + schedules:
        metadata = {
            **act_meta,
            "section_id": sec.get("section_id", ""),
            "section_title": sec.get("section_title", ""),
            "chapter_id": sec.get("chapter_id", ""),
            "chapter_title": sec.get("chapter_title", ""),
            "source": path.name,
        }
        docs.append(Document(page_content=sec["text"], metadata=metadata))

    return docs


def load_all_pdfs(pdf_dir: str) -> List[Document]:
    """
    Load all PDFs from a directory.
    Returns combined list of Documents with metadata.
    """
    pdf_dir = Path(pdf_dir)
    all_docs: List[Document] = []

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        docs = load_pdf_as_documents(str(pdf_path))
        all_docs.extend(docs)
        print(f"  → {len(docs)} sections extracted")

    print(f"\nTotal documents loaded: {len(all_docs)}")
    return all_docs
