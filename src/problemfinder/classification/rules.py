"""Rule-based classifier implementing the Version 2 guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(slots=True)
class ClassificationResult:
    """Structured output from the rule engine."""

    intent: str
    is_problem: str
    is_software_solvable: str
    is_external: str
    problem_reason: str
    software_reason: str
    external_reason: str
    detected_patterns: str
    confidence: float


CLASSIFICATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "is_problem": {"type": "string"},
        "is_software_solvable": {"type": "string"},
        "is_external": {"type": "string"},
        "problem_reason": {"type": "string"},
        "software_reason": {"type": "string"},
        "external_reason": {"type": "string"},
        "detected_patterns": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "intent",
        "is_problem",
        "is_software_solvable",
        "is_external",
        "problem_reason",
        "software_reason",
        "external_reason",
        "detected_patterns",
        "confidence",
    ],
}


INTENT_LABELS = {
    "seeking_help": "1",
    "sharing_advice": "2",
    "showcasing": "3",
    "discussing": "4",
}


class Version2RuleEngine:
    """Rule-based classifier that codifies Version 2 guidance."""

    PROBLEM_CUES = {
        "problem",
        "issue",
        "bug",
        "error",
        "help",
        "can't",
        "cannot",
        "stuck",
        "frustrated",
        "need",
        "looking for",
        "struggling",
        "anyone else",
        "recommend",
        "how do i",
        "how to",
        "should i",
        "broken",
        "glitch",
        "fail",
        "fails",
        "failed",
        "failing",
        "failure",
    }

    NEGATED_PROBLEM_PATTERNS = {
        "no problem": {"problem"},
        "no problems": {"problem"},
        "not a problem": {"problem"},
        "not an issue": {"issue"},
        "no issue": {"issue"},
        "no issues": {"issue"},
        "without issue": {"issue"},
        "without issues": {"issue"},
        "never an issue": {"issue"},
        "not broken": {"broken"},
        "no longer broken": {"broken"},
        "no bug": {"bug"},
        "no bugs": {"bug"},
        "without fail": {"fail"},
        "never failed": {"fail", "failed"},
        "never fails": {"fail", "fails"},
        "never failing": {"fail", "failing"},
        "never failure": {"fail", "failure"},
    }

    RESOLVED_PATTERNS = {
        "already fixed",
        "finally solved",
        "fixed it by",
        "fixed this by",
        "i fixed it by",
        "i fixed this by",
        "issue was resolved when",
        "it was resolved when",
        "ended up fixing it by",
        "randomly got",
        "solved it by",
        "solved this by",
        "turns out it was",
        "was able to fix it by",
    }

    ADVICE_PATTERNS = {
        "here's my advice",
        "here's how i",
        "here is how i",
        "i learned",
        "lessons learned",
        "guide",
        "how i",
        "playbook",
        "tips",
        "tips and tricks",
        "tutorial",
        "walkthrough",
        "workflow",
    }

    ADVICE_CONTEXT_CUES = {
        "for anyone else",
        "in case it helps",
        "hope this helps",
        "sharing my experience",
        "wanted to share",
        "here's what worked",
        "here's what i did",
        "here is what i did",
        "here is what worked",
        "my solution",
        "the solution was",
        "solution:",
        "i fixed",
        "i solved",
        "fixed it by",
        "solved it by",
        "fixed this by",
        "solved this by",
        "fix was",
        "worked for me",
    }

    SOFTWARE_CUES = {
        "app",
        "software",
        "application",
        "program",
        "tool",
        "code",
        "script",
        "automation",
        "api",
        "driver",
        "update",
        "crash",
        "bug",
        "error",
        "config",
        "settings",
        "install",
    }

    NON_SOFTWARE_CUES = {
        "hardware",
        "firmware",
        "motherboard",
        "cpu",
        "gpu",
        "graphics card",
        "power supply",
        "printer",
        "camera",
        "device",
        "replacement",
        "warranty",
        "career",
        "job",
        "education",
        "course",
        "class",
        "buy",
        "purchase",
        "recommend a",
        "looking for a",
        "alternative",
    }

    EXTERNAL_CUES = {
        "warranty",
        "manufacturer",
        "support",
        "customer service",
        "rma",
        "repair",
        "buy",
        "buy a",
        "buy new",
        "buying",
        "looking to buy",
        "purchase",
        "purchase a",
        "replacement",
        "recommend a",
        "recommend",
        "alternative",
        "which should i",
        "vendor",
        "supplier",
        "retailer",
        "store",
        "dealer",
        "quote",
        "estimate",
        "career",
        "job",
        "school",
        "college",
        "printer",
        "hardware",
        "device",
        "samsung",
        "apple",
        "dell",
        "hp",
        "lenovo",
        "asus",
        "acer",
        "microsoft",
        "sony",
        "lg",
        "toshiba",
        "msi",
        "vevor",
        "goxlr",
    }

    PROTOTYPE_PATTERNS = {
        "prototype",
        "sensor",
        "manufacturing",
        "deployment",
    }

    QUESTION_CUES = (
        "how do i",
        "how to",
        "what should i",
        "can anyone",
        "does anyone",
        "anyone know",
        "is there a way",
        "who knows",
        "could someone",
    )

    UNCERTAINTY_WORDS = {
        "maybe",
        "not sure",
        "probably",
        "i think",
        "i guess",
        "perhaps",
    }

    def classify(self, text: str) -> ClassificationResult:
        """Classify ``text`` and return labels with reasoning."""

        lowered = text.lower()

        intent = self._infer_intent(lowered)
        intent_label = INTENT_LABELS[intent]

        is_problem, problem_reason = self._classify_problem(lowered, intent)
        is_software, software_reason = self._classify_software(lowered, is_problem)
        is_external, external_reason = self._classify_external(lowered, is_problem)

        detected_patterns = self._detect_edge_cases(lowered)
        confidence = self._confidence(lowered, detected_patterns)

        return ClassificationResult(
            intent=intent_label,
            is_problem=is_problem,
            is_software_solvable=is_software,
            is_external=is_external,
            problem_reason=problem_reason,
            software_reason=software_reason,
            external_reason=external_reason,
            detected_patterns=", ".join(detected_patterns),
            confidence=confidence,
        )

    def _infer_intent(self, text: str) -> str:
        """Infer the high-level intent of the post."""

        stripped = text.strip()

        has_direct_question = "?" in text or any(
            stripped.startswith(cue) or cue in text for cue in self.QUESTION_CUES
        )
        if has_direct_question:
            return "seeking_help"

        if any(pattern in text for pattern in self.ADVICE_PATTERNS):
            return "sharing_advice"

        if any(pattern in text for pattern in self.RESOLVED_PATTERNS):
            return "sharing_advice"

        if any(pattern in text for pattern in self.ADVICE_CONTEXT_CUES):
            return "sharing_advice"

        if any(phrase in text for phrase in ["i built", "i made", "launch", "showcase"]):
            return "showcasing"

        if any(pattern in text for pattern in self.PROBLEM_CUES):
            return "seeking_help"

        return "discussing"

    def _classify_problem(self, text: str, intent: str) -> Tuple[str, str]:
        """Assign the problem label and rationale."""

        if any(pattern in text for pattern in self.RESOLVED_PATTERNS):
            return "0", "Story explains how the issue was already resolved."

        cues_found = [pattern for pattern in self.PROBLEM_CUES if pattern in text]
        filtered_cues = self._filter_negated_cues(text, cues_found)
        if intent == "showcasing" and not cues_found:
            return "0", "Showcase post without any unresolved frustration."

        if filtered_cues:
            reason = f"Found unresolved problem cues: {', '.join(sorted(set(filtered_cues))[:3])}."
            return "1", reason

        if cues_found and not filtered_cues:
            return "0", "Problem words only show up in negated phrases."

        if intent == "sharing_advice":
            return "1", "Post shares advice based on a real past problem."

        return "0", "No unresolved problem indicators detected."

    def _filter_negated_cues(self, text: str, cues: List[str]) -> List[str]:
        """Remove cues that only appear as part of negated phrases."""

        negated_cues = set()
        for phrase, blocked in self.NEGATED_PROBLEM_PATTERNS.items():
            if phrase in text:
                negated_cues.update(blocked)

        return [cue for cue in cues if cue not in negated_cues]

    def _classify_software(self, text: str, is_problem: str) -> Tuple[str, str]:
        """Determine if the issue is solvable purely through software."""

        if is_problem != "1":
            return "0", "No active problem, so this label stays at 0."

        software_hits = [cue for cue in self.SOFTWARE_CUES if cue in text]
        non_software_hits = [cue for cue in self.NON_SOFTWARE_CUES if cue in text]

        if (
            "looking for" in text
            and any(keyword in text for keyword in {"software", "app", "application", "tool", "program"})
            and not any(word in text for word in {"build", "create", "develop"})
        ):
            return "0", "Looking for an existing product rather than fixing software."

        if software_hits and not non_software_hits:
            return "1", f"Clear software troubleshooting cues: {', '.join(sorted(set(software_hits))[:3])}."

        if non_software_hits and not software_hits:
            return "0", f"Signals point to hardware or market research ({', '.join(sorted(set(non_software_hits))[:3])})."

        if "how do i" in text or "how to" in text:
            return "0", "Question is about learning, not repairing software."

        if "driver" in software_hits or "update" in software_hits:
            return "1", "Mentions drivers or updates linking problem to software."

        if software_hits and non_software_hits:
            return "0", "Mix of hardware and product cues, so not purely software solvable."

        return "0", "No strong software-related clues detected."

    def _classify_external(self, text: str, is_problem: str) -> Tuple[str, str]:
        """Determine if the user needs external coordination or vendors."""

        if is_problem != "1":
            return "0", "No active problem, so this label stays at 0."

        external_hits = [cue for cue in self.EXTERNAL_CUES if cue in text]
        prototype_hits = [cue for cue in self.PROTOTYPE_PATTERNS if cue in text]

        if "which" in text and "should i" in text:
            return "1", "Choosing between outside options, so external help is needed."

        if "job" in text and "advice" in text:
            return "0", "Career advice discussion can be handled individually."

        if any(word in text for word in {"career", "school", "college"}) and "advice" in text:
            return "0", "Education or career advice does not need outside partners."

        if external_hits:
            return "1", f"Needs outside support: {', '.join(sorted(set(external_hits))[:3])}."

        if prototype_hits:
            return "1", "Prototype work calls for external hardware or fabrication help."

        if "how do i" in text or "how to" in text:
            return "0", "User can act alone with the right guidance."

        return "0", "No signs that external coordination is required."

    def _detect_edge_cases(self, text: str) -> List[str]:
        """Return markers for patterns that warrant human review."""

        patterns = []
        if "how do i" in text or "how to" in text:
            patterns.append("learning_question")
        if "looking for" in text:
            patterns.append("seeking_existing_solution")
        if any(pattern in text for pattern in self.ADVICE_PATTERNS):
            patterns.append("advice_sharing")
        if any(pattern in text for pattern in self.RESOLVED_PATTERNS):
            patterns.append("resolved_problem")
        return patterns

    def _confidence(self, text: str, patterns: List[str]) -> float:
        """Estimate confidence based on uncertainty cues."""

        if any(word in text for word in self.UNCERTAINTY_WORDS):
            return 0.3
        if patterns:
            return 0.7
        return 1.0
