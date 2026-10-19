"""Characterization tests for PROPOSED, NON-AUTHORIZED rules only."""
import copy
import unittest

from trace_gate2.measurement_decision_proposal import (
    SUCCESS_WITNESSES, TERMINAL_CLASSES, classify_episode_proposal,
    compare_declared_semantics_proposal,
)


class TestDraftClassification(unittest.TestCase):
    def test_four_classes_exact(self):
        self.assertEqual(TERMINAL_CLASSES, {
            "VALID_RECREATION", "RECREATION_FAILURE",
            "INFRASTRUCTURE_FAILURE", "PROTOCOL_VIOLATION",
        })

    def test_pending_is_not_new_terminal_class(self):
        result = classify_episode_proposal([])
        self.assertIsNone(result.terminal)
        self.assertTrue(result.missing_witnesses)

    def test_all_witnesses_cannot_authorize_live_execution(self):
        result = classify_episode_proposal(SUCCESS_WITNESSES)
        self.assertEqual(result.terminal, "VALID_RECREATION")
        self.assertFalse(result.terminal_is_authorized_for_live_experiment)

    def test_no_cleanup_barrier_cannot_be_valid(self):
        result = classify_episode_proposal(SUCCESS_WITNESSES - {"CLEANUP_BARRIER_PASSED"})
        self.assertIsNone(result.terminal)

    def test_no_final_capture_recheck_cannot_be_valid_under_proposal(self):
        result = classify_episode_proposal(SUCCESS_WITNESSES - {"FINAL_CAPTURE_MANIFESTATION_TRUE"})
        self.assertIsNone(result.terminal)

    def test_no_raw_evidence_cannot_be_valid(self):
        result = classify_episode_proposal(SUCCESS_WITNESSES - {"RAW_EVIDENCE_ADMISSIBLE"})
        self.assertIsNone(result.terminal)

    def test_missing_manifestation_is_recreation_failure(self):
        self.assertEqual(classify_episode_proposal(["MANIFESTATION_TIMEOUT"]).terminal,
                         "RECREATION_FAILURE")

    def test_benchmark_state_missing_evidence_is_recreation_failure(self):
        self.assertEqual(classify_episode_proposal(["BENCHMARK_STATE_EVIDENCE_MISSING"]).terminal,
                         "RECREATION_FAILURE")

    def test_cleanup_residue_is_recreation_failure(self):
        self.assertEqual(classify_episode_proposal(["CLEANUP_RESIDUE"]).terminal,
                         "RECREATION_FAILURE")

    def test_collector_failure_not_benchmark_failure(self):
        self.assertEqual(classify_episode_proposal(["COLLECTOR_FAILURE"]).terminal,
                         "PROTOCOL_VIOLATION")

    def test_incomplete_capture_not_valid(self):
        self.assertEqual(classify_episode_proposal(["INCOMPLETE_CAPTURE"]).terminal,
                         "PROTOCOL_VIOLATION")

    def test_watch_gap_not_valid(self):
        self.assertEqual(classify_episode_proposal(["WATCH_CONTINUITY_FAILURE"]).terminal,
                         "PROTOCOL_VIOLATION")

    def test_external_host_failure_is_infrastructure(self):
        self.assertEqual(classify_episode_proposal(["EXTERNAL_DOCKER_KIND_HOST_FAILURE"]).terminal,
                         "INFRASTRUCTURE_FAILURE")

    def test_protocol_precedes_recreation(self):
        result = classify_episode_proposal(["FROZEN_CONTRACT_BREACH", "MANIFESTATION_TIMEOUT"])
        self.assertEqual(result.terminal, "PROTOCOL_VIOLATION")
        self.assertIn("MANIFESTATION_TIMEOUT", result.observed_events)

    def test_protocol_precedes_external_infra(self):
        result = classify_episode_proposal(["COLLECTOR_FAILURE", "EXTERNAL_DOCKER_KIND_HOST_FAILURE"])
        self.assertEqual(result.terminal, "PROTOCOL_VIOLATION")

    def test_verified_recreation_precedes_subsequent_infra(self):
        result = classify_episode_proposal(["MANIFESTATION_TIMEOUT", "EXTERNAL_DOCKER_KIND_HOST_FAILURE"])
        self.assertEqual(result.terminal, "RECREATION_FAILURE")

    def test_failures_cannot_be_turned_success_by_complete_witnesses(self):
        result = classify_episode_proposal(SUCCESS_WITNESSES | {"INCOMPLETE_CAPTURE"})
        self.assertEqual(result.terminal, "PROTOCOL_VIOLATION")

    def test_unknown_event_cannot_be_guessed(self):
        with self.assertRaises(ValueError):
            classify_episode_proposal(["SOME_UNKNOWN_CODE"])

    def test_text_input_rejected(self):
        with self.assertRaises(ValueError):
            classify_episode_proposal("MANIFESTATION_TIMEOUT")

    def test_repeat_events_deduplicated_without_suppressing_fails(self):
        result = classify_episode_proposal(["CLEANUP_RESIDUE", "CLEANUP_RESIDUE"])
        self.assertEqual(result.terminal, "RECREATION_FAILURE")
        self.assertEqual(result.observed_events, ("CLEANUP_RESIDUE",))


class TestDraftComparison(unittest.TestCase):
    fields = ("count", "fault", "nested")

    def base(self):
        return {"count": 1, "fault": True, "nested": {"state": "NOAUTH", "ready": [True, False]}}

    def compare(self, left, right):
        return compare_declared_semantics_proposal(left, right, required_fields=self.fields)

    def test_equivalent_baseline(self):
        a = self.base()
        result = self.compare(a, copy.deepcopy(a))
        self.assertTrue(result.admissible and result.equivalent)
        self.assertFalse(result.authorized_for_live_experiment)

    def test_dictionary_key_order_not_material(self):
        a = self.base()
        b = {"nested": {"ready": [True, False], "state": "NOAUTH"}, "fault": True, "count": 1}
        self.assertTrue(self.compare(a, b).equivalent)

    def test_boolean_integer_not_equivalent(self):
        b = self.base()
        b["fault"] = 1
        r = self.compare(self.base(), b)
        self.assertTrue(r.admissible)
        self.assertFalse(r.equivalent)
        self.assertIn("$.fault", r.mismatch_paths)

    def test_string_boolean_not_equivalent(self):
        b = self.base()
        b["fault"] = "true"
        self.assertFalse(self.compare(self.base(), b).equivalent)

    def test_semantic_count_change_is_visible(self):
        b = self.base()
        b["count"] = 2
        self.assertEqual(self.compare(self.base(), b).mismatch_paths, ("$.count",))

    def test_nested_auth_result_change_is_visible(self):
        b = self.base()
        b["nested"]["state"] = "PONG"
        self.assertEqual(self.compare(self.base(), b).mismatch_paths, ("$.nested.state",))

    def test_list_order_remains_material_until_schema_specifies(self):
        b = self.base()
        b["nested"]["ready"] = [False, True]
        self.assertFalse(self.compare(self.base(), b).equivalent)

    def test_missing_required_field_not_treated_as_difference(self):
        b = self.base()
        b.pop("fault")
        self.assertFalse(self.compare(self.base(), b).admissible)

    def test_undeclared_semantic_field_does_not_silently_expand_gate(self):
        b = self.base()
        b["extra"] = "not-frozen"
        self.assertFalse(self.compare(self.base(), b).admissible)

    def test_null_invalid_not_defaulted(self):
        b = self.base()
        b["count"] = None
        self.assertFalse(self.compare(self.base(), b).admissible)

    def test_float_invalid_without_declared_tolerance(self):
        b = self.base()
        b["count"] = 1.0
        self.assertFalse(self.compare(self.base(), b).admissible)

    def test_negative_integer_invalid(self):
        b = self.base()
        b["count"] = -1
        self.assertFalse(self.compare(self.base(), b).admissible)

    def test_unknown_nested_type_invalid(self):
        b = self.base()
        b["nested"]["ready"] = {True, False}
        self.assertFalse(self.compare(self.base(), b).admissible)

    def test_incomplete_schema_not_accepted(self):
        r = compare_declared_semantics_proposal(self.base(), self.base(), required_fields=())
        self.assertFalse(r.admissible)

    def test_duplicate_schema_fields_not_accepted(self):
        r = compare_declared_semantics_proposal(self.base(), self.base(), required_fields=("count", "count"))
        self.assertFalse(r.admissible)

    def test_no_byte_hash_identity_used_as_capture_provenance(self):
        r = self.compare(self.base(), self.base())
        self.assertTrue(r.equivalent)
        self.assertFalse(r.authorized_for_live_experiment)


if __name__ == "__main__":
    unittest.main()
