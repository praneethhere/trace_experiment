"""Offline falsification cases for the prospective v1.32.1 status decoder."""
import copy
import unittest

from trace_gate2.deployment_status_v132 import decode_deployment_status_v132


def dep():
    return {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": "search", "namespace": "astronomy-shop", "uid": "dep-1", "generation": 4},
        "spec": {"replicas": 1},
        "status": {"observedGeneration": 4, "replicas": 0, "unavailableReplicas": 1,
                   "conditions": [{"type": "Available", "status": "False"}]},
    }


class TestDeploymentStatusV132(unittest.TestCase):
    def test_p4_zero_ready_omitted_is_valid_zero(self):
        r = decode_deployment_status_v132(dep())
        self.assertTrue(r.schema_valid, r.reasons)
        self.assertTrue(r.status_current, r.reasons)
        self.assertEqual(r.p4_replica_fields, {"desired": 1, "ready": 0, "unavailable": 1})
        self.assertIn("readyReplicas", r.omitted_zero_fields)
        self.assertFalse(r.execution_admissible)

    def test_explicit_zero_and_omitted_zero_same_semantic_with_provenance(self):
        x = dep()
        y = copy.deepcopy(x)
        y["status"]["readyReplicas"] = 0
        a, b = decode_deployment_status_v132(x), decode_deployment_status_v132(y)
        self.assertEqual(a.p4_replica_fields, b.p4_replica_fields)
        self.assertIn("readyReplicas", a.omitted_zero_fields)
        self.assertNotIn("readyReplicas", b.omitted_zero_fields)

    def test_explicit_null_cannot_be_defaulted(self):
        x = dep()
        x["status"]["readyReplicas"] = None
        self.assertFalse(decode_deployment_status_v132(x).schema_valid)

    def test_wrong_ready_type_string_fails(self):
        x = dep()
        x["status"]["readyReplicas"] = "0"
        self.assertFalse(decode_deployment_status_v132(x).schema_valid)

    def test_boolean_is_not_integer(self):
        x = dep()
        x["status"]["readyReplicas"] = False
        self.assertFalse(decode_deployment_status_v132(x).schema_valid)

    def test_negative_status_counter_fails(self):
        x = dep()
        x["status"]["unavailableReplicas"] = -1
        self.assertFalse(decode_deployment_status_v132(x).schema_valid)

    def test_missing_status_not_defaulted(self):
        x = dep()
        del x["status"]
        a = decode_deployment_status_v132(x)
        self.assertFalse(a.schema_valid)
        self.assertIsNone(a.p4_replica_fields)

    def test_stale_generation_not_semantically_usable(self):
        x = dep()
        x["status"]["observedGeneration"] = 3
        a = decode_deployment_status_v132(x)
        self.assertTrue(a.schema_valid)
        self.assertFalse(a.status_current)
        self.assertIsNone(a.p4_replica_fields)

    def test_missing_observed_generation_not_semantically_usable(self):
        x = dep()
        del x["status"]["observedGeneration"]
        a = decode_deployment_status_v132(x)
        self.assertTrue(a.schema_valid)
        self.assertFalse(a.status_current)
        self.assertIsNone(a.p4_replica_fields)

    def test_explicit_null_observed_generation_fails(self):
        x = dep()
        x["status"]["observedGeneration"] = None
        self.assertFalse(decode_deployment_status_v132(x).schema_valid)

    def test_missing_spec_replicas_remains_invalid(self):
        x = dep()
        del x["spec"]["replicas"]
        self.assertFalse(decode_deployment_status_v132(x).schema_valid)

    def test_missing_identity_is_invalid(self):
        x = dep()
        del x["metadata"]["uid"]
        self.assertFalse(decode_deployment_status_v132(x).schema_valid)

    def test_wrong_api_kind_invalid(self):
        x = dep()
        x["kind"] = "ReplicaSet"
        self.assertFalse(decode_deployment_status_v132(x).schema_valid)

    def test_does_not_mutate_raw(self):
        x = dep()
        original = copy.deepcopy(x)
        decode_deployment_status_v132(x)
        self.assertEqual(original, x)

    def test_nonzero_ready_remains_nonzero(self):
        x = dep()
        x["status"]["readyReplicas"] = 1
        x["status"]["unavailableReplicas"] = 0
        r = decode_deployment_status_v132(x)
        self.assertEqual(r.p4_replica_fields, {"desired": 1, "ready": 1, "unavailable": 0})

    def test_other_omitted_status_counts_zero_but_preserved_as_omitted(self):
        x = dep()
        del x["status"]["unavailableReplicas"]
        r = decode_deployment_status_v132(x)
        self.assertTrue(r.schema_valid)
        self.assertEqual(r.p4_replica_fields["unavailable"], 0)
        self.assertIn("unavailableReplicas", r.omitted_zero_fields)

    def test_zero_desired_is_not_rewritten_to_one(self):
        x = dep()
        x["spec"]["replicas"] = 0
        r = decode_deployment_status_v132(x)
        self.assertEqual(r.desired, 0)
        self.assertTrue(r.schema_valid)

    def test_future_generation_not_semantically_usable(self):
        x = dep()
        x["status"]["observedGeneration"] = 5
        r = decode_deployment_status_v132(x)
        self.assertTrue(r.schema_valid)
        self.assertFalse(r.status_current)
        self.assertIsNone(r.p4_replica_fields)

    def test_non_dict_rejected_without_crash(self):
        self.assertFalse(decode_deployment_status_v132(None).schema_valid)

    def test_missing_generation_rejected(self):
        x = dep()
        del x["metadata"]["generation"]
        self.assertFalse(decode_deployment_status_v132(x).schema_valid)


if __name__ == "__main__":
    unittest.main()
