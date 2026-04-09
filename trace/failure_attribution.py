from openai import OpenAI
from config import (MODEL, DETECTOR_TEMPERATURE,
                    THETA_GROUND, THETA_LOOP, N_LOOP, RHO_THRESHOLD, THETA_H)
import re
from collections import deque

client = OpenAI()

CLAIM_MARKERS = [
    "the root cause is", "the failure is caused by", "this indicates",
    "the issue is", "i recommend", "based on the logs", "the problem is"
]

class FailureAttributionModule:
    def __init__(self, grounding_prompt, contradiction_prompt):
        self.grounding_prompt = grounding_prompt
        self.contradiction_prompt = contradiction_prompt
        self.fingerprint_history = deque(maxlen=10)
        self.detector_calls = 0   # counts LLM calls for overhead tracking

    # ── F1: Grounding Detector ───────────────────────────────────────────────
    def detect_F1(self, reasoning, window):
        claims = self._extract_claims(reasoning)
        evidence = [e.get("observation", {}) for e in window if e.get("observation")]
        evidence_text = "\n".join(str(e) for e in evidence if e)

        for claim in claims:
            llm_grounded = self._llm_grounding_check(claim, evidence_text)
            kw_overlap = self._token_overlap(claim, evidence_text)
            if not llm_grounded and kw_overlap < THETA_GROUND:
                conf = max(0, 1 - kw_overlap / THETA_GROUND)
                return True, conf, claim
        return False, 0.0, None

    def _extract_claims(self, text):
        claims = []
        sentences = re.split(r'[.!?]', text)
        for s in sentences:
            if any(m in s.lower() for m in CLAIM_MARKERS):
                claims.append(s.strip())
        return claims

    def _llm_grounding_check(self, claim, evidence):
        self.detector_calls += 1
        prompt = self.grounding_prompt.format(claim=claim, evidence=evidence)
        resp = client.chat.completions.create(
            model=MODEL, temperature=DETECTOR_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}])
        return "SUPPORTED" in resp.choices[0].message.content.upper()

    def _token_overlap(self, text1, text2):
        t1 = set(re.findall(r'\w+', text1.lower()))
        t2 = set(re.findall(r'\w+', text2.lower()))
        return len(t1 & t2)

    # ── F2: Contradiction Detector ───────────────────────────────────────────
    def detect_F2(self, reasoning, window):
        prior = [e.get("reasoning", "") for e in window[:-1] if e.get("reasoning")]
        prior_text = "\n".join(prior)
        if not prior_text:
            return False, 0.0

        self.detector_calls += 1
        prompt = self.contradiction_prompt.format(
            new_statement=reasoning, prior_statements=prior_text)
        resp = client.chat.completions.create(
            model=MODEL, temperature=DETECTOR_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}])
        contradicts = "CONTRADICTS" in resp.choices[0].message.content.upper()
        return contradicts, 1.0 if contradicts else 0.0

    # ── F3: Loop Detector ─────────────────────────────────────────────────────
    def detect_F3(self, reasoning, action):
        fp = self._fingerprint(reasoning, action)
        count = sum(1 for h in self.fingerprint_history if h == fp)
        self.fingerprint_history.append(fp)
        if count >= N_LOOP:
            return True, 1.0
        # Check similarity of recent subsequences
        if len(self.fingerprint_history) >= 4:
            recent = list(self.fingerprint_history)[-4:]
            sim = self._sequence_similarity(recent[:2], recent[2:])
            if sim >= THETA_LOOP:
                return True, sim
        return False, 0.0

    def _fingerprint(self, reasoning, action):
        action_type = action.split("_")[0] if action else "none"
        words = set(re.findall(r'\w+', reasoning.lower()))
        key_words = frozenset(list(words)[:5])
        return (action_type, key_words)

    def _sequence_similarity(self, seq1, seq2):
        if not seq1 or not seq2:
            return 0.0
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / max(len(seq1), len(seq2))

    # ── F4: Tool Anomaly Detector ─────────────────────────────────────────────
    def detect_F4(self, tool_status, rho, tool_output):
        if tool_status in ("fail", "timeout") and rho >= RHO_THRESHOLD:
            return True, 1.0
        if not tool_output or tool_output == {}:
            return True, 0.9
        return False, 0.0

    def get_detector_calls(self):
        return self.detector_calls
