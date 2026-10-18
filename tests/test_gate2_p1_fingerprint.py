"""Synthetic offline tests: no KIND, kubectl, network, or model calls."""

from __future__ import annotations

import copy
import unittest

from trace_gate2.p1 import evaluate_p1


NS = "hotel-reservation"


def fixture(token: str = "a") -> dict:
    def meta(name, uid, **extra):
        return {"namespace": NS, "name": name, "uid": uid, "resourceVersion": "42",
                "creationTimestamp": "2026-10-15T15:00:00Z", **extra}

    result = {"namespace": NS, "deployments": [], "replicasets": [],
              "pods": [], "services": [], "endpoints": []}
    for index, name in enumerate(("frontend", "search"), start=1):
        d_uid = f"{token}-deployment-{name}"
        rs_name = f"{name}-abc123-{token}"
        rs_uid = f"{token}-replicaset-{name}"
        pod_name = f"{rs_name}-xyz99"
        pod_uid = f"{token}-pod-{name}"
        result["deployments"].append({
            "metadata": meta(name, d_uid),
            "spec": {"replicas": 1, "template": {"metadata": {"labels": {
                "service-route": "frontend", "app": name}}}},
            "status": {"readyReplicas": 1},
        })
        result["replicasets"].append({
            "metadata": meta(rs_name, rs_uid, ownerReferences=[{
                "kind": "Deployment", "name": name, "uid": d_uid, "controller": True}])
        })
        result["pods"].append({
            "metadata": meta(pod_name, pod_uid, labels={"service-route": "frontend", "app": name},
                             ownerReferences=[{"kind": "ReplicaSet", "name": rs_name,
                                               "uid": rs_uid, "controller": True}]),
            "spec": {"nodeName": f"{token}-worker-1"},
            "status": {"phase": "Running", "podIP": f"10.4.0.{index}",
                       "conditions": [{"type": "Ready", "status": "True"}]},
        })

    result["services"].append({
        "metadata": meta("frontend", f"{token}-service"),
        "spec": {"selector": {"service-route": "frontend"},
                 "clusterIP": "10.96.0.15",
                 "ports": [{"port": 5000, "targetPort": 5000, "protocol": "TCP"}]},
    })
    addresses = [{"ip": f"10.4.0.{i+1}", "targetRef": {"kind": "Pod",
                  "name": pod["metadata"]["name"], "uid": pod["metadata"]["uid"]}}
                 for i, pod in enumerate(result["pods"])]
    result["endpoints"].append({"metadata": meta("frontend", f"{token}-ep"),
                                "subsets": [{"addresses": addresses, "ports": [{"port": 5000}]}]})
    return result


class TestP1Manifestation(unittest.TestCase):
    def test_positive_mixed_endpoint_fixture(self):
        r = evaluate_p1(fixture())
        self.assertTrue(r.valid_manifestation, r.errors)
        self.assertEqual(r.semantic["frontend_ready_endpoints"], 1)
        self.assertEqual(r.semantic["search_ready_endpoints"], 1)
        self.assertTrue(r.semantic["mixed_endpoint_contamination"])

    def test_ephemeral_identity_and_ips_do_not_change_semantic_hash(self):
        a, b = fixture("a"), fixture("b")
        b["services"][0]["spec"]["clusterIP"] = "10.96.255.98"
        for i, pod in enumerate(b["pods"], start=1):
            pod["status"]["podIP"] = f"10.18.8.{i}"
            b["endpoints"][0]["subsets"][0]["addresses"][i - 1]["ip"] = f"10.18.8.{i}"
        self.assertEqual(evaluate_p1(a).semantic_sha256, evaluate_p1(b).semantic_sha256)

    def test_missing_wrong_endpoint_must_fail(self):
        f = fixture()
        f["endpoints"][0]["subsets"][0]["addresses"].pop()
        result = evaluate_p1(f)
        self.assertFalse(result.valid_manifestation)
        self.assertIn("frontend Service has no ready endpoint owned by search Deployment", result.errors)

    def test_unmapped_endpoint_must_fail(self):
        f = fixture()
        f["endpoints"][0]["subsets"][0]["addresses"].append({
            "ip": "10.9.8.7", "targetRef": {"kind": "Pod", "name": "unknown"}})
        result = evaluate_p1(f)
        self.assertFalse(result.valid_manifestation)
        self.assertIn("frontend Service has unidentified ready endpoints", result.errors)

    def test_terminating_search_pod_not_counted_as_ready(self):
        f = fixture()
        f["pods"][1]["metadata"]["deletionTimestamp"] = "2026-10-15T17:00:00Z"
        result = evaluate_p1(f)
        self.assertFalse(result.valid_manifestation)
        self.assertEqual(result.semantic["search_ready_endpoints"], 0)

    def test_owner_chain_cannot_be_inferred_from_name_or_label(self):
        f = fixture()
        f["pods"][1]["metadata"]["ownerReferences"][0]["uid"] = "wrong"
        result = evaluate_p1(f)
        self.assertFalse(result.valid_manifestation)
        self.assertEqual(result.semantic["unidentified_ready_endpoints"], 1)

    def test_material_service_port_change_changes_semantic_hash(self):
        f = fixture()
        modified = copy.deepcopy(f)
        modified["services"][0]["spec"]["ports"][0]["targetPort"] = 5001
        modified["endpoints"][0]["subsets"][0]["ports"][0]["port"] = 5001
        before, after = evaluate_p1(f), evaluate_p1(modified)
        self.assertTrue(before.valid_manifestation)
        self.assertTrue(after.valid_manifestation)
        self.assertNotEqual(before.semantic_sha256, after.semantic_sha256)

    def test_not_exact_fault_selector_must_fail(self):
        f = fixture()
        f["services"][0]["spec"]["selector"]["extra"] = "anything"
        self.assertFalse(evaluate_p1(f).valid_manifestation)

    def test_missing_raw_kind_fails_closed(self):
        f = fixture()
        del f["replicasets"]
        self.assertFalse(evaluate_p1(f).valid_manifestation)

    def test_no_mutation_of_input_evidence(self):
        f = fixture()
        original = copy.deepcopy(f)
        evaluate_p1(f)
        self.assertEqual(f, original)

    def test_no_guessing_with_different_namespace(self):
        f = fixture()
        f["pods"][1]["metadata"]["namespace"] = "another-ns"
        self.assertFalse(evaluate_p1(f).valid_manifestation)

    def test_endpoint_requires_explicit_target_uid(self):
        f = fixture()
        f["endpoints"][0]["subsets"][0]["addresses"][1]["targetRef"].pop("uid")
        r = evaluate_p1(f)
        self.assertFalse(r.valid_manifestation)
        self.assertEqual(r.semantic["unidentified_ready_endpoints"], 1)

    def test_endpoint_ip_must_match_referenced_pod(self):
        f = fixture()
        f["endpoints"][0]["subsets"][0]["addresses"][1]["ip"] = "10.99.99.99"
        r = evaluate_p1(f)
        self.assertFalse(r.valid_manifestation)
        self.assertEqual(r.semantic["unidentified_ready_endpoints"], 1)

    def test_endpoint_subset_must_contain_port(self):
        f = fixture()
        f["endpoints"][0]["subsets"][0].pop("ports")
        r = evaluate_p1(f)
        self.assertFalse(r.valid_manifestation)
        self.assertIn(
            "frontend Endpoints subset lacks the frontend TCP target port",
            r.errors,
        )

    def test_endpoint_subset_port_must_match_service_target(self):
        f = fixture()
        f["endpoints"][0]["subsets"][0]["ports"][0]["port"] = 9999
        r = evaluate_p1(f)
        self.assertFalse(r.valid_manifestation)
        self.assertIn(
            "frontend Endpoints subset lacks the frontend TCP target port",
            r.errors,
        )

    def test_missing_deployment_ready_count_fails_closed(self):
        f = fixture()
        f["deployments"][0]["status"].pop("readyReplicas")
        r = evaluate_p1(f)
        self.assertFalse(r.valid_manifestation)
        self.assertIn("frontend: readyReplicas missing, zero, or invalid", r.errors)

    def test_zero_deployment_ready_count_conflicts_with_ready_pod(self):
        f = fixture()
        f["deployments"][0]["status"]["readyReplicas"] = 0
        self.assertFalse(evaluate_p1(f).valid_manifestation)

    def test_null_deployment_status_fails_closed(self):
        f = fixture()
        f["deployments"][0]["status"] = None
        self.assertFalse(evaluate_p1(f).valid_manifestation)

    def test_malformed_owner_reference_fails_closed(self):
        f = fixture()
        f["pods"][1]["metadata"]["ownerReferences"] = "not-a-list"
        self.assertFalse(evaluate_p1(f).valid_manifestation)

    def test_malformed_endpoint_target_fails_closed(self):
        f = fixture()
        f["endpoints"][0]["subsets"][0]["addresses"][1]["targetRef"] = "not-an-object"
        self.assertFalse(evaluate_p1(f).valid_manifestation)

    def test_conflicting_readiness_conditions_do_not_pass(self):
        f = fixture()
        f["pods"][1]["status"]["conditions"].append({"type": "Ready", "status": "False"})
        self.assertFalse(evaluate_p1(f).valid_manifestation)

    def test_missing_pod_uid_breaks_owner_attribution(self):
        f = fixture()
        f["replicasets"][1]["metadata"]["uid"] = "changed-without-ref-update"
        self.assertFalse(evaluate_p1(f).valid_manifestation)


if __name__ == "__main__":
    unittest.main()
