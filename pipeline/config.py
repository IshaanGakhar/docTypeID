"""
Central configuration for the legal document pipeline.
All regex patterns and thresholds are defined here to keep other modules clean.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent
MODEL_DIR  = BASE_DIR / "models"
CACHE_DIR  = BASE_DIR / ".cache"

MODEL_DOCTYPE_PATH  = MODEL_DIR / "doctype_model.joblib"
MODEL_CRF_PATH      = MODEL_DIR / "crf_ner.joblib"
LABEL_BINARIZER_PATH = MODEL_DIR / "label_binarizer.joblib"
VECTORIZER_WORD_PATH = MODEL_DIR / "vectorizer_word.joblib"
VECTORIZER_CHAR_PATH = MODEL_DIR / "vectorizer_char.joblib"

# ---------------------------------------------------------------------------
# Extraction thresholds
# ---------------------------------------------------------------------------
MIN_CHARS          = 150      # skip document if total text < this
CAPTION_ZONE_LINES = 40
TITLE_ZONE_LINES   = 60
TFIDF_TITLE_BOOST  = 1.5      # score multiplier for boosted title patterns
TITLE_MIN_CAPS_LEN = 8        # minimum length for all-caps title candidate
DOCTYPE_THRESHOLD  = 0.5      # min probability to label a document type
HEADER_FRAC        = 0.12     # top fraction of page height treated as header band

# Header/footer stripping
# Off by default: legal documents deliberately repeat the caption (court name,
# case number, party names) in ECF running headers — stripping them removes
# exactly the fields we want to extract.
STRIP_REPEATED_LINES_IN_BODY = False   # if True, repeated lines removed from body zone only
REPEATED_LINE_MIN_PAGES      = 4       # pages threshold when stripping is enabled

# ---------------------------------------------------------------------------
# Document type labels
# Specific subtypes come first; generic fallbacks are at the end.
# ---------------------------------------------------------------------------
DOCUMENT_TYPE_LABELS = [
    # Motion subtypes
    "Motion to Dismiss",
    "Motion to Stay",
    "Motion for Summary Judgment",
    "Motion to Compel",
    "Motion in Limine",
    "Motion to Strike",
    "Motion to Transfer",
    "Motion to Remand",
    "Motion for Class Certification",
    "Motion for Injunction",
    "Motion for Sanctions",
    "Motion for Reconsideration",
    "Motion for Leave to Amend",
    "Motion for Extension of Time",
    "Motion for Default Judgment",
    "Motion to Quash",
    "Motion to Intervene",
    "Motion for Judgment on the Pleadings",
    "Motion for Protective Order",
    "Motion",                         # generic fallback
    # Order subtypes
    "Temporary Restraining Order",
    "Preliminary Injunction",
    "Scheduling Order",
    "Order",                          # generic fallback
    # Complaint subtypes
    "Amended Complaint",
    "Class Action Complaint",
    "Complaint",                      # generic fallback
    # Brief subtypes
    "Opening Brief",
    "Reply Brief",
    "Opposition Brief",
    "Appellate Brief",
    "Brief",                          # generic fallback
    # Notice subtypes
    "Notice of Appeal",
    "Notice of Removal",
    "Notice of Motion",
    "Notice",                         # generic fallback
    # Other specific types
    "Petition",
    "Answer",
    "Memorandum",
    "Stipulation",
    "Judgment",
    "Subpoena",
    "Affidavit",
    "Declaration",
    "Transcript",
    "Summons",
    "Counterclaim",
    "Cross-Claim",
    "Third-Party Complaint",
    "Proposed Order",
    "Citation",
    "Pro Hac Vice Motion",
    "Response",
    "Entry of Appearance",
    "Corporate Disclosure Statement",
    "Order of Transfer",
    "Exhibit",
    "Retainer Agreement",
    "Notice of Voluntary Dismissal",
]

# ---------------------------------------------------------------------------
# Classifier backend  ("lr" | "svm" | "cnb")
# ---------------------------------------------------------------------------
CLASSIFIER_BACKEND = "lr"   # LogisticRegression

# ---------------------------------------------------------------------------
# Title extraction — keyword prefixes that identify candidate lines
# ---------------------------------------------------------------------------
TITLE_PREFIXES = [
    "MOTION",
    "NOTICE",
    "ORDER",
    "MEMORANDUM",
    "BRIEF",
    "COMPLAINT",
    "PETITION",
    "ANSWER",
    "RESPONSE",
    "REPLY",
    "DECLARATION",
    "AFFIDAVIT",
    "STIPULATION",
    "JUDGMENT",
    "SUBPOENA",
    "SUMMONS",
    "APPLICATION",
    "OBJECTION",
    "OPPOSITION",
    "REQUEST",
    "PROPOSED",
]

TITLE_BOOST_PHRASES = [
    r"MOTION\s+TO\b",
    r"MEMORANDUM\s+IN\s+SUPPORT\b",
    r"NOTICE\s+OF\b",
    r"ORDER\s+(?:GRANTING|DENYING|RE:)\b",
    r"PETITION\s+FOR\b",
    r"COMPLAINT\s+FOR\b",
]

# Words / phrases that start a continuation line of a compound title.
# e.g. "MOTION TO DISMISS" followed by "AND CHANGE OF VENUE"
TITLE_CONTINUATION_WORDS = [
    "AND", "OR", "FOR", "TO", "OF", "IN", "WITH", "UPON", "AS",
    "PURSUANT", "REGARDING", "RE:", "&",
]

# Maximum line-number gap between a title candidate and its continuation
TITLE_CONTINUATION_MAX_GAP = 2

OPENING_PATTERNS = [
    r"COMES\s+NOW\b",
    r"NOW\s+COMES\b",
    r"^Plaintiff\b",
    r"^Defendant\b",
    r"^Petitioner\b",
    r"^Respondent\b",
]

# ---------------------------------------------------------------------------
# Document type — rule-based regex patterns (fallback when model missing)
# ---------------------------------------------------------------------------
DOCTYPE_RULES: dict[str, list[str]] = {
    # ── Motion subtypes (specific first, generic last) ────────────────────
    "Motion to Dismiss":              [r"\bMOTION\s+TO\s+DISMISS\b"],
    "Motion to Stay":                 [r"\bMOTION\s+TO\s+STAY\b"],
    "Motion for Summary Judgment":    [r"\bMOTION\s+FOR\s+SUMMARY\s+JUDGMENT\b",
                                       r"\bSUMMARY\s+JUDGMENT\s+MOTION\b"],
    "Motion to Compel":               [r"\bMOTION\s+TO\s+COMPEL\b"],
    "Motion in Limine":               [r"\bMOTION\s+IN\s+LIMINE\b"],
    "Motion to Strike":               [r"\bMOTION\s+TO\s+STRIKE\b"],
    "Motion to Transfer":             [r"\bMOTION\s+TO\s+TRANSFER\b",
                                       r"\bMOTION\s+TO\s+CHANGE\s+VENUE\b"],
    "Motion to Remand":               [r"\bMOTION\s+TO\s+REMAND\b"],
    "Motion for Class Certification": [r"\bMOTION\s+FOR\s+CLASS\s+CERTIFICATION\b",
                                       r"\bCLASS\s+CERTIFICATION\s+MOTION\b"],
    "Motion for Injunction":          [r"\bMOTION\s+FOR\s+(?:PRELIMINARY\s+)?INJUNCTION\b",
                                       r"\bMOTION\s+FOR\s+(?:TEMPORARY\s+RESTRAINING\s+ORDER|TRO)\b"],
    "Motion for Sanctions":           [r"\bMOTION\s+FOR\s+SANCTIONS\b"],
    "Motion for Reconsideration":     [r"\bMOTION\s+FOR\s+RECONSIDERATION\b",
                                       r"\bMOTION\s+FOR\s+REHEARING\b"],
    "Motion for Leave to Amend":      [r"\bMOTION\s+FOR\s+LEAVE\s+TO\s+AMEND\b",
                                       r"\bMOTION\s+TO\s+AMEND\b"],
    "Motion for Extension of Time":   [r"\bMOTION\s+FOR\s+(?:AN?\s+)?EXTENSION\s+OF\s+TIME\b",
                                       r"\bMOTION\s+TO\s+EXTEND\b"],
    "Motion for Default Judgment":    [r"\bMOTION\s+FOR\s+DEFAULT\s+JUDGMENT\b",
                                       r"\bDEFAULT\s+JUDGMENT\s+MOTION\b"],
    "Motion to Quash":                [r"\bMOTION\s+TO\s+QUASH\b"],
    "Motion to Intervene":            [r"\bMOTION\s+TO\s+INTERVENE\b"],
    "Motion for Judgment on the Pleadings": [r"\bMOTION\s+FOR\s+JUDGMENT\s+ON\s+THE\s+PLEADINGS\b"],
    "Motion for Protective Order":    [r"\bMOTION\s+FOR\s+(?:A\s+)?PROTECTIVE\s+ORDER\b"],
    "Motion":                         [r"\bMOTION\s+TO\b", r"\bMOTION\s+FOR\b",
                                       r"\bMOTION\s+IN\b", r"\bMOTION\b"],

    # ── Order subtypes ────────────────────────────────────────────────────
    "Temporary Restraining Order":    [r"\bTEMPORARY\s+RESTRAINING\s+ORDER\b", r"\bT\.?R\.?O\.?\b"],
    "Preliminary Injunction":         [r"\bPRELIMINARY\s+INJUNCTION\b"],
    "Scheduling Order":               [r"\bSCHEDULING\s+ORDER\b", r"\bCASE\s+MANAGEMENT\s+ORDER\b"],
    "Order":                          [r"\bORDER\b", r"\bIT\s+IS\s+(?:HEREBY\s+)?ORDERED\b"],

    # ── Complaint subtypes ────────────────────────────────────────────────
    "Amended Complaint":              [r"\bAMENDED\s+COMPLAINT\b", r"\bFIRST\s+AMENDED\s+COMPLAINT\b",
                                       r"\bSECOND\s+AMENDED\s+COMPLAINT\b"],
    "Class Action Complaint":         [r"\bCLASS\s+ACTION\s+COMPLAINT\b",
                                       r"\bCOMPLAINT\b.{0,60}\bCLASS\s+ACTION\b"],
    "Complaint":                      [r"\bCOMPLAINT\b", r"\bCOMPLAINT\s+FOR\b"],

    # ── Brief subtypes ────────────────────────────────────────────────────
    "Opening Brief":                  [r"\bOPENING\s+BRIEF\b"],
    "Reply Brief":                    [r"\bREPLY\s+BRIEF\b", r"\bREPLY\s+IN\s+SUPPORT\b"],
    "Opposition Brief":               [r"\bOPPOSITION\s+(?:TO|BRIEF)\b",
                                       r"\bMEMORANDUM\s+IN\s+OPPOSITION\b"],
    "Appellate Brief":                [r"\bAPPELLATE\s+BRIEF\b", r"\bAPPEAL\s+BRIEF\b"],
    "Brief":                          [r"\bBRIEF\b"],

    # ── Notice subtypes ───────────────────────────────────────────────────
    "Notice of Appeal":               [r"\bNOTICE\s+OF\s+APPEAL\b"],
    "Notice of Removal":              [r"\bNOTICE\s+OF\s+REMOVAL\b"],
    "Notice of Motion":               [r"\bNOTICE\s+OF\s+MOTION\b"],
    "Notice":                         [r"\bNOTICE\s+OF\b", r"\bNOTICE\s+TO\b"],

    # ── Other types ───────────────────────────────────────────────────────
    "Petition":                       [r"\bPETITION\s+FOR\b", r"\bPETITION\b"],
    "Answer":                         [r"\bANSWER\s+(?:TO|AND)\b", r"\bANSWER\b"],
    "Memorandum":                     [r"\bMEMORANDUM\b", r"\bMEMO\s+OF\s+LAW\b"],
    "Stipulation":                    [r"\bSTIPULATION\b", r"\bSTIPULATED\b"],
    "Judgment":                       [r"\bJUDGMENT\b", r"\bFINAL\s+JUDGMENT\b"],
    "Subpoena":                       [r"\bSUBPOENA\b"],
    "Affidavit":                      [r"\bAFFIDAVIT\b", r"\bSWORN\s+STATEMENT\b"],
    "Declaration":                    [r"\bDECLARATION\s+OF\b", r"\bDECLARATION\b"],
    "Transcript":                     [r"\bTRANSCRIPT\b", r"\bPROCEEDINGS\b"],
    "Summons":                        [r"\bSUMMONS\b"],
    "Counterclaim":                   [r"\bCOUNTERCLAIM\b"],
    "Cross-Claim":                    [r"\bCROSS[- ]CLAIM\b"],
    "Third-Party Complaint":          [r"\bTHIRD[- ]PARTY\s+COMPLAINT\b"],
    "Proposed Order":                 [r"\bPROPOSED\s+ORDER\b"],
    "Citation":                       [r"\bCITATION\b.{0,40}\byou\s+have\s+been\s+sued\b",
                                       r"\byou\s+have\s+been\s+sued\b",
                                       r"\bCITATION\s+BY\s+(?:SERVING|PUBLICATION)\b",
                                       r"\bTHE\s+STATE\s+OF\s+TEXAS\b.{0,60}\bCITATION\b"],
    "Proof of Service":               [r"\bPROOF\s+OF\s+SERVICE\b"],
    "Certificate of Service":         [r"\bCERTIFICATE\s+OF\s+SERVICE\b"],
    "Cover Letter":                   [r"\bCOVER\s+LETTER\b", r"\bTRANSMITTAL\s+LETTER\b"],
    "Demand Letter":                  [r"\bDEMAND\s+LETTER\b", r"\bNOTICE\s+AND\s+DEMAND\b"],
    "Civil Cover Sheet":              [r"\bCIVIL\s+COVER\s+SHEET\b"],
    "Letter":                         [r"\bLETTER\s+BRIEF\b",
                                       r"\bLETTER\s+TO\s+(?:THE\s+)?(?:COURT|JUDGE)\b",
                                       r"\bDEAR\s+(?:JUDGE|HON(?:ORABLE)?\.?|MAGISTRATE|JUSTICE)\b"],
    "Pro Hac Vice Motion":            [r"\bPRO\s+HAC\s+VICE\b",
                                       r"\bAPPEAR\s+PHV\b",
                                       r"\bADMISSION\s+PRO\s+HAC\b"],
    "Response":                       [r"\bRESPONSE\s+(?:TO|IN)\b",
                                       r"\bOMNIBUS\s+RESPONSE\b"],
    "Entry of Appearance":            [r"\bENTRY\s+OF\s+APPEARANCE\b",
                                       r"\bAPPEARANCE\s+OF\s+COUNSEL\b",
                                       r"\bNOTICE\s+OF\s+APPEARANCE\b"],
    "Corporate Disclosure Statement": [r"\bCORPORATE\s+DISCLOSURE\s+STATEMENT\b",
                                       r"\bRULE\s+7\.1\s+(?:CORPORATE\s+)?DISCLOSURE\b",
                                       r"\b7\.1\s+DISCLOSURE\s+STATEMENT\b"],
    "Order of Transfer":              [r"\bORDER\s+OF\s+TRANSFER\b",
                                       r"\bTRANSFER\s+ORDER\b"],
    "Exhibit":                        [r"\bEXHIBIT\s+[A-Z0-9]+\b"],
    "Retainer Agreement":             [r"\bRETAINER\s+(?:AGREEMENT|LETTER)\b",
                                       r"\bRETAINED\b.{0,40}\bTO\s+REPRESENT\s+YOU\b"],
    "Notice of Voluntary Dismissal":  [r"\bVOLUNTARY\s+DISMISSAL\b"],
}

# ---------------------------------------------------------------------------
# Court extraction regex patterns
# ---------------------------------------------------------------------------
COURT_PATTERNS = [
    r"(UNITED\s+STATES\s+DISTRICT\s+COURT[^\n]{0,80})",
    r"(UNITED\s+STATES\s+BANKRUPTCY\s+COURT[^\n]{0,80})",
    r"(UNITED\s+STATES\s+COURT\s+OF\s+APPEALS[^\n]{0,80})",
    r"(UNITED\s+STATES\s+SUPREME\s+COURT[^\n]{0,80})",
    r"(SUPERIOR\s+COURT[^\n]{0,80})",
    r"(SUPREME\s+COURT\s+OF[^\n]{0,80})",
    r"(CIRCUIT\s+COURT[^\n]{0,80})",
    r"(FAMILY\s+COURT[^\n]{0,80})",
    r"(COURT\s+OF\s+APPEALS[^\n]{0,80})",
    r"(STATE\s+OF\s+[A-Z][A-Za-z\s]+COURT[^\n]{0,40})",
    r"(IN\s+THE\s+(?:UNITED\s+STATES\s+)?(?:DISTRICT|SUPERIOR|CIRCUIT|FAMILY|SUPREME)\s+COURT[^\n]{0,80})",
]

COURT_LOCATION_PATTERNS = [
    # 1. "FOR THE NORTHERN DISTRICT OF CALIFORNIA" (most specific — includes directional)
    r"(?:FOR[ \t]+THE|IN[ \t]+THE)[ \t]+((?:NORTHERN|SOUTHERN|EASTERN|WESTERN|CENTRAL|MIDDLE)[ \t]+DISTRICT[ \t]+OF[ \t]+[A-Z][A-Za-z ]+?)(?:\n|$|,)",
    # 2. Standalone directional: "NORTHERN DISTRICT OF TEXAS" (must come before bare DISTRICT OF
    #    so "NORTHERN DISTRICT OF TEXAS" isn't truncated to just "TEXAS")
    r"\b((?:NORTHERN|SOUTHERN|EASTERN|WESTERN|CENTRAL|MIDDLE)[ \t]+DISTRICT[ \t]+OF[ \t]+[A-Z][A-Za-z ]+?)(?:\n|$|,|[ \t]{2,})",
    # 3a. County courts: "COUNTY OF SANTA CLARA" or "COUNTY OF LOS ANGELES"
    #     Limit to 3 words max after "OF" so form labels like
    #     "County of Residence of First Listed Plaintiff" don't match.
    r"\b(COUNTY[ \t]+OF[ \t]+[A-Z][A-Za-z]{1,}(?:[ \t]+[A-Z][A-Za-z]{1,}){0,2})(?:\n|$|,|[ \t]{2,})",
    # 3b. Reversed county: "DALLAS COUNTY" / "SANTA CLARA COUNTY"
    r"\b([A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,2}[ \t]+COUNTY)\b",
    # 3c. Parish (Louisiana): "ORLEANS PARISH" or "PARISH OF ORLEANS"
    r"\b((?:[A-Z][A-Za-z]+[ \t]+)?PARISH(?:[ \t]+OF[ \t]+[A-Z][A-Za-z ]+?)?)\b",
    # 4. "DISTRICT OF MARYLAND" bare (least specific — captures only state; last resort)
    r"(?:DISTRICT[ \t]+OF[ \t]+)([A-Z][A-Za-z ]{2,30})(?:\n|$|,)",
]

US_STATES = (
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|"
    "Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|"
    "Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|"
    "Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|"
    "North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|"
    "South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
    "West Virginia|Wisconsin|Wyoming|District of Columbia"
)

# ---------------------------------------------------------------------------
# Judge extraction regex patterns
# ---------------------------------------------------------------------------
# A name part is either a full word (Dale, Tillery) or a single-letter initial
# optionally followed by a period (B. or B), with 1+ whitespace between parts.
_NAME_PART     = r"[A-Z][a-z]+"                 # mixed-case word: Dale, Reice
_INITIAL       = r"[A-Z]\.?"                     # middle initial: B or B.
_MC_NAME_WORD  = rf"(?:{_NAME_PART}|{_INITIAL})" # either form
_MC_NAME       = rf"(?:{_MC_NAME_WORD}\s+){{1,4}}{_NAME_PART}"  # 2-5 parts, last must be a word

_UC_NAME_PART  = r"[A-Z]+\.?"                    # ALL-CAPS word or initial
_UC_NAME       = rf"(?:{_UC_NAME_PART}\s+){{1,4}}{_UC_NAME_PART}"

JUDGE_PATTERNS = [
    rf"(?:Hon(?:orable)?\.?\s+)({_MC_NAME})",
    rf"(?:Judge\s+)({_MC_NAME})",
    rf"(?:Magistrate\s+Judge\s+)({_MC_NAME})",
    rf"(?:Chief\s+Judge\s+)({_MC_NAME})",
    rf"(?:Senior\s+Judge\s+)({_MC_NAME})",
    rf"(?:Justice\s+)({_MC_NAME})",
    rf"(?:JUDGE\s+)({_UC_NAME})",
    rf"(?:HONORABLE\s+)({_UC_NAME})",
    rf"(?:Before\s+(?:the\s+)?(?:Honorable|Hon\.?)\s+)({_MC_NAME})",
    rf"(?:Assigned\s+to[:\s]+)({_MC_NAME})",
]

# ---------------------------------------------------------------------------
# Date extraction regex patterns
# ---------------------------------------------------------------------------
DATE_PATTERNS = [
    # MM/DD/YYYY or M/D/YYYY
    r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
    # Month DD, YYYY
    r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
    # DD Month YYYY
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
    # Abbreviated month
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4})\b",
    # YYYY-MM-DD
    r"\b(\d{4}-\d{2}-\d{2})\b",
]

# Context keyword stems that can precede a filing date.
# Used two ways:
#   1. Anchored combined patterns (prefix + separator + date) — highest confidence
#   2. Lookbehind window scan (legacy fallback)
DATE_CONTEXT_KEYWORDS = [
    # Legacy window-scan patterns (kept for fallback)
    r"(?:filed|dated|entered|signed|issued|effective)\s+(?:on\s+)?",
    r"Date(?:d)?[:\s]+",
    r"Filing\s+Date[:\s]+",
]

# Keyword stems for anchored prefix+date patterns.
# Each stem is combined with every separator variant and every date format.
DATE_PREFIX_STEMS = [
    r"filed",
    r"date\s+filed",
    r"case\s+filed",
    r"dated",
    r"date",
    r"filing\s+date",
    r"entered",
    r"signed",
    r"issued",
    r"effective",
    r"received",
    r"recorded",
    r"submitted",
    r"stamped",
    r"given\s+under\s+my\s+hand",
    r"this\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?\s+day\s+of",
]

# Separator variants between keyword and date:
#   colon, hyphen, en-dash, em-dash — each optionally surrounded by spaces.
#   Also plain whitespace (no punctuation).
DATE_PREFIX_SEPARATORS = [
    r"\s*:\s*",      # ": "
    r"\s*-\s*",      # " - "
    r"\s*–\s*",      # " – " (en dash)
    r"\s*—\s*",      # " — " (em dash)
    r"\s+",          # plain space
]

# ECF (Electronic Case Filing) header pattern — authoritative date source
# Matches: "Case 1:19-cv-00886-DLC Document 168 Filed 05/26/21 Page 1 of 2"
ECF_HEADER_PATTERN = (
    r"(?:Case|Docket)\s+[\w:\-]+\s+Document\s+\d+\s+Filed\s+"
    r"(\d{1,2}/\d{1,2}/\d{2,4})"
)

# Negative-context patterns: dates appearing near these are NOT filing dates
DATE_NEGATIVE_PATTERNS = [
    # Case citations: "138 S. Ct. 1061 (2018)"
    r"\d+\s+S\.\s*Ct\.\s*\d+",
    # Federal reporter citations: "123 F.3d 456"
    r"\d+\s+F\.\d[a-z]*\s+\d+",
    # U.S. reporter: "123 U.S. 456"
    r"\d+\s+U\.S\.\s+\d+",
    # Parenthetical year at end of citation: "(2018)" or "(S.D.N.Y. 2020)"
    r"\([A-Z][A-Za-z\.\s,]*\d{4}\)",
    # Biographical / incorporation
    r"\bborn\s+(?:on\s+)?",
    r"\bincorporat(?:ed|ion)\s+(?:(?:on|in)\s+)?",
    # Legal history of other cases
    r"\bamended\s+(?:on\s+)?",
    r"\bdecided\s+(?:on\s+)?",
    # News / article attributions
    r"\barticle\s+(?:dated|from)\s+",
]

# ---------------------------------------------------------------------------
# Party extraction patterns
# ---------------------------------------------------------------------------
VERSUS_PATTERN = r"\bv(?:s?\.?|ersus)\b"

PARTY_ROLE_PATTERNS = {
    "plaintiffs":   [r"\bPlaintiff(?:s)?\b", r"\bPLAINTIFF(?:S)?\b"],
    "defendants":   [r"\bDefendant(?:s)?\b", r"\bDEFENDANT(?:S)?\b"],
    "petitioners":  [r"\bPetitioner(?:s)?\b", r"\bPETITIONER(?:S)?\b"],
    "respondents":  [r"\bRespondent(?:s)?\b", r"\bRESPONDENT(?:S)?\b"],
    "appellants":   [r"\bAppellant(?:s)?\b", r"\bAPPELLANT(?:S)?\b"],
    "appellees":    [r"\bAppellee(?:s)?\b", r"\bAPPELLEE(?:S)?\b"],
}

PARTY_NOISE_PATTERNS = [
    r"\bet\s+al\.?\b",
    r"\band\s+others\b",
    r",?\s+individually\b",
    r",?\s+et\s+ux\.?\b",
    r",?\s+et\s+vir\.?\b",
]

# ---------------------------------------------------------------------------
# Clause / heading extraction
# ---------------------------------------------------------------------------
NUMBERED_HEADING_PATTERN = r"^(?:[IVXLC]+\.|[A-Z]\.|[\d]+\.)\s+([A-Z][A-Za-z\s\-\(\)]{4,70})$"

CANONICAL_CLAUSE_HEADINGS: dict[str, str] = {
    "JURISDICTION":            "jurisdiction",
    "VENUE":                   "venue",
    "FACTS":                   "facts",
    "FACTUAL BACKGROUND":      "facts",
    "BACKGROUND":              "background",
    "INTRODUCTION":            "introduction",
    "ARGUMENT":                "argument",
    "LEGAL ARGUMENT":          "argument",
    "LEGAL STANDARD":          "legal_standard",
    "STANDARD OF REVIEW":      "legal_standard",
    "DISCUSSION":              "discussion",
    "ANALYSIS":                "analysis",
    "CONCLUSION":              "conclusion",
    "PRAYER FOR RELIEF":       "relief",
    "RELIEF REQUESTED":        "relief",
    "RELIEF SOUGHT":           "relief",
    "CAUSES OF ACTION":        "causes_of_action",
    "CAUSE OF ACTION":         "causes_of_action",
    "CLAIMS FOR RELIEF":       "causes_of_action",
    "STATEMENT OF FACTS":      "facts",
    "STATEMENT OF THE CASE":   "statement_of_case",
    "STATEMENT OF ISSUES":     "issues",
    "ISSUES PRESENTED":        "issues",
    "SUMMARY OF ARGUMENT":     "summary",
    "SUMMARY":                 "summary",
    "TABLE OF CONTENTS":       "table_of_contents",
    "TABLE OF AUTHORITIES":    "table_of_authorities",
    "PARTIES":                 "parties",
    "NATURE OF THE ACTION":    "nature_of_action",
    "PROCEDURAL HISTORY":      "procedural_history",
    "PROCEDURAL BACKGROUND":   "procedural_history",
    "CERTIFICATE OF SERVICE":  "certificate_of_service",
    "SIGNATURE":               "signature",
    "WHEREFORE":               "relief",
    "THEREFORE":               "relief",
}

# ---------------------------------------------------------------------------
# Texas citation boilerplate detection
# ---------------------------------------------------------------------------
# When a document matches this pattern on its first page, it is a citation /
# service-of-process form.  The boilerplate text (answer, petition, judgment)
# must be suppressed so it doesn't pollute doc-type and title extraction.
CITATION_BOILERPLATE_RE_STR = (
    r"(?:you\s+have\s+been\s+sued|"
    r"citation\s+by\s+(?:serving|publication)|"
    r"issued\s+this\s+citation\b)"
)

MIN_CLAUSE_TEXT_LENGTH = 20  # minimum body text length to keep a clause

# ---------------------------------------------------------------------------
# CRF NER feature config
# ---------------------------------------------------------------------------
CRF_CONTEXT_WINDOW = 2   # tokens of context on each side for CRF features
CRF_ENTITY_LABELS  = ["COURT", "JUDGE", "DATE", "CASE_NO", "PARTY", "O"]
