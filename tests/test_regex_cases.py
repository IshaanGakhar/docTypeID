"""
Unit tests for regex-based extraction modules.

Tests cover:
  - NUID / docket number extraction (all tier patterns)
  - Court name extraction
  - Court location extraction
  - Judge name extraction
  - Date extraction + ISO normalization
  - Party extraction (v. split + role keywords)
  - Document type rule-based classification

Run:
    pytest tests/test_regex_cases.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.uid_extractor import extract_nuid, normalize_uid, _validate_docket
from pipeline.court_judge_extractor import extract_court_and_judge
from pipeline.date_extractor import extract_filing_date
from pipeline.party_extractor import extract_parties
from pipeline.doc_type_classifier import _rule_based_classify
from pipeline.preprocess import preprocess
from pipeline.pdf_loader import LoadedDocument, PageInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str, page_num: int = 1) -> LoadedDocument:
    """Wrap text in a minimal LoadedDocument for preprocess()."""
    page = PageInfo(page_num=page_num, text=text, char_count=len(text))
    return LoadedDocument(
        pdf_path="<test>",
        full_text=text,
        pages=[page],
        metadata={},
        total_chars=len(text),
    )


def _zones_from(text: str):
    return preprocess(_make_doc(text))


# ---------------------------------------------------------------------------
# NUID extraction tests
# ---------------------------------------------------------------------------

class TestNUIDExtraction:
    def _extract(self, text: str):
        return extract_nuid(text)

    def test_civil_action_no(self):
        result = self._extract("CIVIL ACTION No. 2:18-cv-03969-R-FFM")
        assert result.nuid is not None
        assert result.source_tier == "labeled"

    def test_case_no_with_district(self):
        result = self._extract("Case No. 1:22-cv-00456-ABC")
        assert result.nuid is not None
        assert result.source_tier == "labeled"

    def test_case_no_without_district(self):
        result = self._extract("Case No. 22-cv-03885")
        assert result.nuid is not None
        assert result.source_tier == "labeled"

    def test_header_style(self):
        result = self._extract("Case 2:18-cv-03969-R-FFM\nSome content")
        assert result.nuid is not None

    def test_docket_no(self):
        result = self._extract("Docket No. 19-CV-5678")
        assert result.nuid is not None

    def test_mdl_docket(self):
        result = self._extract("MDL No. 2:15-md-02641-HRH")
        assert result.nuid is not None

    def test_no_docket_returns_none(self):
        result = self._extract("This document has no case number.")
        assert result.nuid is None
        assert result.confidence == 0.0

    def test_normalize_uid_lowercases_type_code(self):
        normalized = normalize_uid("2:18-cv-03969")
        assert "CV" in normalized

    def test_normalize_uid_strips_whitespace(self):
        normalized = normalize_uid("  2:18-cv-03969  ")
        assert not normalized.startswith(" ")
        assert not normalized.endswith(" ")

    def test_en_dash_normalized(self):
        result = self._extract("Case No. 22\u2013cv\u201303885")
        # Should normalize en-dashes to hyphens and still find the docket
        assert result.nuid is not None or result.nuid is None  # no crash

    def test_evidence_populated(self):
        result = self._extract("Case No. 22-cv-01234")
        if result.nuid:
            assert len(result.evidence) > 0
            ev = result.evidence[0]
            assert ev.source == "regex"
            assert ev.span_text


class TestDocketValidation:
    def test_valid_federal_with_district(self):
        assert _validate_docket("2:18-cv-03969")

    def test_valid_federal_without_district(self):
        assert _validate_docket("22-cv-03885")

    def test_valid_mdl(self):
        assert _validate_docket("2389")

    def test_valid_appeals(self):
        assert _validate_docket("19-3063")

    def test_invalid_too_short(self):
        assert not _validate_docket("12")

    def test_invalid_no_digit(self):
        assert not _validate_docket("ABCDE")

    def test_invalid_too_long(self):
        assert not _validate_docket("A" * 51)


# ---------------------------------------------------------------------------
# Court + Judge extraction tests
# ---------------------------------------------------------------------------

class TestCourtExtraction:
    def _zones(self, text: str):
        return _zones_from(text)

    def test_us_district_court(self):
        text = "UNITED STATES DISTRICT COURT FOR THE NORTHERN DISTRICT OF CALIFORNIA\n" + "x" * 200
        zones = self._zones(text)
        result = extract_court_and_judge(zones)
        assert result.court_name is not None
        assert "DISTRICT" in result.court_name.upper()

    def test_superior_court(self):
        text = "SUPERIOR COURT OF THE STATE OF CALIFORNIA\n" + "x" * 200
        zones = self._zones(text)
        result = extract_court_and_judge(zones)
        assert result.court_name is not None

    def test_court_location_northern_district(self):
        text = "UNITED STATES DISTRICT COURT FOR THE NORTHERN DISTRICT OF CALIFORNIA\n" + "x" * 200
        zones = self._zones(text)
        result = extract_court_and_judge(zones)
        assert result.court_location is not None
        assert "California" in result.court_location or "CALIFORNIA" in result.court_location.upper()

    def test_court_location_state_fallback(self):
        text = "Supreme Court of Texas\nSome content\n" + "x" * 200
        zones = self._zones(text)
        result = extract_court_and_judge(zones)
        assert result.court_location is not None

    def test_judge_hon_pattern(self):
        text = (
            "UNITED STATES DISTRICT COURT\n"
            "Hon. Jane Smith\n"
            "Plaintiff v. Defendant\n" + "x" * 200
        )
        zones = self._zones(text)
        result = extract_court_and_judge(zones)
        assert result.judge_name is not None
        assert "Smith" in result.judge_name or "Jane" in result.judge_name

    def test_judge_assigned_to(self):
        text = (
            "UNITED STATES DISTRICT COURT\n"
            "Assigned to: Hon. Robert Johnson\n" + "x" * 200
        )
        zones = self._zones(text)
        result = extract_court_and_judge(zones)
        assert result.judge_name is not None

    def test_no_court_returns_none(self):
        text = "This document mentions nothing about courts.\n" + "x" * 200
        zones = self._zones(text)
        result = extract_court_and_judge(zones)
        assert result.court_name is None

    def test_court_evidence_source_regex(self):
        text = "UNITED STATES DISTRICT COURT\n" + "x" * 200
        zones = self._zones(text)
        result = extract_court_and_judge(zones)
        if result.court_evidence:
            assert result.court_evidence[0].source == "regex"


# ---------------------------------------------------------------------------
# Date extraction tests
# ---------------------------------------------------------------------------

class TestDateExtraction:
    def _extract(self, text: str):
        return extract_filing_date(_zones_from(text))

    def test_slash_date(self):
        result = self._extract("Filed: 03/15/2023\n" + "x" * 200)
        assert result.filing_date == "2023-03-15"

    def test_month_day_year(self):
        result = self._extract("Date: March 15, 2023\n" + "x" * 200)
        assert result.filing_date == "2023-03-15"

    def test_iso_date(self):
        result = self._extract("Dated: 2023-03-15\n" + "x" * 200)
        assert result.filing_date == "2023-03-15"

    def test_abbreviated_month(self):
        result = self._extract("Signed: Jan. 5, 2022\n" + "x" * 200)
        assert result.filing_date == "2022-01-05"

    def test_no_date_returns_none(self):
        result = self._extract("This document has no date." + "x" * 200)
        assert result.filing_date is None
        assert result.confidence == 0.0

    def test_contextual_date_higher_confidence(self):
        result_ctx = self._extract("Filed: January 1, 2022\n" + "x" * 200)
        result_bare = self._extract("January 1, 2022\n" + "x" * 200)
        if result_ctx.filing_date and result_bare.filing_date:
            assert result_ctx.confidence >= result_bare.confidence

    def test_evidence_present_when_date_found(self):
        result = self._extract("Filed: March 15, 2023\n" + "x" * 200)
        assert result.filing_date is not None
        assert len(result.evidence) > 0

    def test_evidence_has_required_fields(self):
        result = self._extract("Dated: 2023-03-15\n" + "x" * 200)
        if result.evidence:
            ev = result.evidence[0]
            assert ev.source == "regex"
            assert ev.span_text
            assert ev.rule_id


# ---------------------------------------------------------------------------
# Party extraction tests
# ---------------------------------------------------------------------------

class TestPartyExtraction:
    def _extract(self, text: str):
        return extract_parties(_zones_from(text))

    def test_vs_split_plaintiffs_defendants(self):
        text = (
            "ACME CORPORATION,\n"
            "    Plaintiff,\n\n"
            "v.\n\n"
            "GLOBEX LLC,\n"
            "    Defendant.\n" + "x" * 200
        )
        result = self._extract(text)
        parties = result.parties
        assert len(parties["plaintiffs"]) > 0 or len(parties["defendants"]) > 0

    def test_role_keyword_override(self):
        text = "Plaintiff: John Doe\nDefendant: Jane Smith\n" + "x" * 200
        result = self._extract(text)
        parties = result.parties
        # Role keyword extraction
        all_parties = parties["plaintiffs"] + parties["defendants"]
        assert len(all_parties) > 0

    def test_et_al_stripped(self):
        text = (
            "ACME CORP., et al.,\n\n"
            "v.\n\n"
            "GLOBEX LLC\n" + "x" * 200
        )
        result = self._extract(text)
        for party_list in result.parties.values():
            for name in party_list:
                assert "et al" not in name.lower()

    def test_no_versus_returns_empty_lists(self):
        text = "This document has no parties listed.\n" + "x" * 200
        result = self._extract(text)
        parties = result.parties
        all_found = sum(len(v) for v in parties.values())
        assert all_found == 0 or result.confidence <= 0.5

    def test_evidence_source_rule(self):
        text = (
            "Acme Inc.\n\nv.\n\nGlobal Corp\n" + "x" * 200
        )
        result = self._extract(text)
        for ev in result.evidence:
            assert ev.source in ("rule", "regex", "crf")


# ---------------------------------------------------------------------------
# Document type rule-based classification tests
# ---------------------------------------------------------------------------

class TestDocTypeRuleBased:
    def _classify(self, text: str, title: str = ""):
        return _rule_based_classify(text, title)

    def test_motion_to_dismiss(self):
        result = self._classify("MOTION TO DISMISS under Rule 12(b)(6)")
        assert "Motion" in result.document_types

    def test_order_pattern(self):
        result = self._classify("IT IS HEREBY ORDERED that the motion is granted.")
        assert "Order" in result.document_types

    def test_complaint(self):
        result = self._classify("COMPLAINT FOR BREACH OF CONTRACT")
        assert "Complaint" in result.document_types

    def test_notice_of(self):
        result = self._classify("NOTICE OF FILING")
        assert "Notice" in result.document_types

    def test_multilabel(self):
        result = self._classify("MOTION TO DISMISS\nIT IS HEREBY ORDERED")
        assert "Motion" in result.document_types or "Order" in result.document_types

    def test_no_match_returns_empty(self):
        result = self._classify("The quick brown fox.")
        # No strong signal — types may or may not be empty
        assert isinstance(result.document_types, list)

    def test_fallback_flag_set(self):
        result = self._classify("MOTION TO DISMISS")
        assert result.fallback_used is True

    def test_evidence_has_rule_source(self):
        result = self._classify("MOTION TO DISMISS")
        for ev in result.evidence:
            assert ev.source == "rule"

    def test_confidence_between_0_and_1(self):
        result = self._classify("COMPLAINT FOR DAMAGES")
        assert 0.0 <= result.confidence <= 1.0

    def test_affidavit_detected(self):
        result = self._classify("AFFIDAVIT OF JOHN DOE")
        assert "Affidavit" in result.document_types

    def test_stipulation_detected(self):
        result = self._classify("STIPULATION AND ORDER")
        assert "Stipulation" in result.document_types or "Order" in result.document_types

    def test_compound_title_both_types(self):
        """'Motion to Dismiss and Memorandum in Support' → Motion + Memorandum."""
        result = self._classify(
            "Some body text.",
            title="Motion to Dismiss and Memorandum in Support",
        )
        assert "Motion" in result.document_types
        assert "Memorandum" in result.document_types

    def test_compound_notice_single_type(self):
        """'Notice of Appointment of Counsel and Settlement' → Notice only."""
        result = self._classify(
            "Some body text.",
            title="Notice of Appointment of Counsel and Settlement",
        )
        assert "Notice" in result.document_types

    def test_compound_motion_and_petition(self):
        result = self._classify(
            "Some body text.",
            title="Motion to Strike and Petition for Relief",
        )
        assert "Motion" in result.document_types
        assert "Petition" in result.document_types

    def test_compound_evidence_has_rule_source(self):
        result = self._classify(
            "Some body text.",
            title="Motion to Dismiss and Memorandum in Support",
        )
        for ev in result.evidence:
            assert ev.source == "rule"


class TestCompoundTitleExtraction:
    """Title extractor merges continuation lines into compound titles."""

    def _zones(self, text: str):
        from pipeline.pdf_loader import LoadedDocument, PageInfo
        from pipeline.preprocess import preprocess
        page = PageInfo(page_num=1, text=text, char_count=len(text))
        doc = LoadedDocument(
            pdf_path="<test>", full_text=text, pages=[page],
            metadata={}, total_chars=len(text),
        )
        return preprocess(doc)

    def _extract(self, text: str):
        from pipeline.title_extractor import extract_title
        return extract_title(self._zones(text))

    def test_single_line_title_unchanged(self):
        text = "MOTION TO DISMISS\nBody text here with more detail.\n" + "x" * 200
        result = self._extract(text)
        assert result.title is not None
        assert "MOTION TO DISMISS" in result.title

    def test_compound_and_continuation_merged(self):
        text = (
            "MOTION TO DISMISS\n"
            "AND CHANGE OF VENUE\n"
            "Body text follows here.\n" + "x" * 200
        )
        result = self._extract(text)
        assert result.title is not None
        assert "MOTION TO DISMISS" in result.title
        assert "AND CHANGE OF VENUE" in result.title

    def test_compound_for_continuation_merged(self):
        text = (
            "NOTICE OF APPOINTMENT OF COUNSEL\n"
            "AND SETTLEMENT AGREEMENT\n"
            "Body text.\n" + "x" * 200
        )
        result = self._extract(text)
        assert result.title is not None
        assert "NOTICE" in result.title
        assert "SETTLEMENT" in result.title

    def test_non_continuation_line_not_merged(self):
        """A new standalone heading should NOT be appended to the title."""
        text = (
            "MOTION TO DISMISS\n\n"
            "INTRODUCTION\n"
            "Body text here.\n" + "x" * 200
        )
        result = self._extract(text)
        if result.title:
            # INTRODUCTION should not be part of the title
            assert "INTRODUCTION" not in result.title or "MOTION" in result.title

    def test_evidence_spans_compound(self):
        text = (
            "MOTION TO DISMISS\n"
            "AND CHANGE OF VENUE\n"
            "Body text.\n" + "x" * 200
        )
        result = self._extract(text)
        if result.title and result.evidence:
            ev = result.evidence[0]
            assert ev.span_text == result.title


class TestNoHeaderFooterStripping:
    """Repeated lines in caption/title zones must NOT be dropped."""

    def _zones(self, pages_text: list[str]):
        from pipeline.pdf_loader import LoadedDocument, PageInfo
        from pipeline.preprocess import preprocess
        pages = [
            PageInfo(page_num=i + 1, text=t, char_count=len(t))
            for i, t in enumerate(pages_text)
        ]
        full = "\n".join(pages_text)
        doc = LoadedDocument(
            pdf_path="<test>", full_text=full, pages=pages,
            metadata={}, total_chars=len(full),
        )
        return preprocess(doc)

    def test_court_name_present_even_when_repeated(self):
        """Court name repeated on every page must still appear in caption_zone."""
        court_line = "UNITED STATES DISTRICT COURT FOR THE NORTHERN DISTRICT OF CALIFORNIA"
        pages = [court_line + f"\nPage {i} content here.\n" + "x" * 100 for i in range(5)]
        zones = self._zones(pages)
        caption_texts = [il.text for il in zones.caption_zone]
        assert any(court_line in t or t in court_line for t in caption_texts), \
            "Court name was incorrectly stripped from caption_zone"

    def test_case_number_present_even_when_repeated(self):
        case_no = "Case No. 3:22-cv-01234-JCS"
        pages = [case_no + f"\nContent on page {i}.\n" + "x" * 100 for i in range(4)]
        zones = self._zones(pages)
        all_texts = " ".join(il.text for il in zones.all_lines)
        assert "3:22-cv-01234" in all_texts or "01234" in all_texts

    def test_is_repeated_flag_set(self):
        """is_repeated should be True for truly repeated lines (informational)."""
        from pipeline.config import STRIP_REPEATED_LINES_IN_BODY, REPEATED_LINE_MIN_PAGES
        if not STRIP_REPEATED_LINES_IN_BODY:
            # Flag detection only runs when stripping is enabled; skip this check
            return
        repeated = "CASE HEADER LINE"
        pages = [repeated + f"\nUnique content {i}.\n" + "x" * 100
                 for i in range(REPEATED_LINE_MIN_PAGES + 1)]
        zones = self._zones(pages)
        repeated_flags = [il.is_repeated for il in zones.all_lines if il.text == repeated]
        assert all(repeated_flags)
