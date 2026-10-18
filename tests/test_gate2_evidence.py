"""Offline evidence contract tests. No kubectl, KIND, network, or models."""
import copy
import hashlib
import json
import unittest

from trace_gate2.evidence import (
    SCHEMA, P1_ID, PROTOCOL_SHA256, SREGYM_SHA, APPLICATIONS_SHA,
    audit_capture, check_observed_item_bracket,
)

NS = "hotel-reservation"
CAMPAIGN = "test-campaign-1"
EPISODE = "G2-R1-P1"
CLUSTER_UID = "kube-system-uid-1"
NOW = "2026-10-18T14:00:00Z"
LATER = "2026-10-18T14:00:01Z"

KIND_META = {
    "deployments": ("DeploymentList", "apps/v1", "Deployment"),
    "replicasets": ("ReplicaSetList", "apps/v1", "ReplicaSet"),
    "pods": ("PodList", "v1", "Pod"),
    "services": ("ServiceList", "v1", "Service"),
    "endpoints": ("EndpointsList", "v1", "Endpoints"),
    "endpointslices": ("EndpointSliceList", "discovery.k8s.io/v1", "EndpointSlice"),
}


def record(body, request):
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {"raw_json": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "http_status": 200, "started_at_utc": NOW, "ended_at_utc": LATER,
            "request": request}


def list_request(key):
    return {"method": "GET", "resource": key, "namespace": NS,
            "query": {"resourceVersion": "", "limit": None, "continue": "",
                      "labelSelector": "", "fieldSelector": ""}}


def make_capture():
    records = {}
    for key, (lk, api, _kind) in KIND_META.items():
        body = {"kind": lk, "apiVersion": api,
                "metadata": {"resourceVersion": "100", "continue": ""}, "items": []}
        records[key] = record(body, list_request(key))
    records["namespace"] = record(
        {"kind": "Namespace", "apiVersion": "v1",
         "metadata": {"name": NS, "uid": "ns-uid-1", "resourceVersion": "70"}},
        {"method": "GET", "path": f"/api/v1/namespaces/{NS}", "query": {}},
    )
    return {"schema": SCHEMA, "identity": {
        "campaign_id": CAMPAIGN, "episode_id": EPISODE, "cluster_uid": CLUSTER_UID,
        "protocol_sha256": PROTOCOL_SHA256, "sregym_sha": SREGYM_SHA,
        "applications_sha": APPLICATIONS_SHA, "problem_id": P1_ID,
        "namespace": NS, "namespace_uid": "ns-uid-1", "trace_commit": "90c2018",
        "collector_version": "synthetic-v0", "api_server": "https://local-fixture.invalid"},
        "records": records}


def put_item(capture, key, obj):
    rec = capture["records"][key]
    body = json.loads(rec["raw_json"])
    body["items"].append(obj)
    capture["records"][key] = record(body, rec["request"])


def make_deployment(name="frontend", uid="deploy-1", rv="17"):
    return {"kind": "Deployment", "apiVersion": "apps/v1",
            "metadata": {"namespace": NS, "name": name, "uid": uid,
                         "resourceVersion": rv, "generation": 2},
            "spec": {"replicas": 1},
            "status": {"readyReplicas": 1, "observedGeneration": 2}}


def make_pod(name="pod1", uid="pod-1", ip="10.0.0.1", rv="18"):
    return {"kind": "Pod", "apiVersion": "v1",
            "metadata": {"namespace": NS, "name": name, "uid": uid, "resourceVersion": rv},
            "status": {"phase": "Running", "podIP": ip}}


def verify(capture):
    return audit_capture(capture, expected_campaign_id=CAMPAIGN,
                         expected_episode_id=EPISODE, expected_cluster_uid=CLUSTER_UID)


class TestEvidenceContract(unittest.TestCase):
    def test_valid_recorded_format_does_not_authorize_execution(self):
        result = verify(make_capture())
        self.assertTrue(result.schema_valid, result.reasons)
        self.assertFalse(result.execution_admissible)

    def test_missing_episode_identity(self):
        c = make_capture()
        del c["identity"]["episode_id"]
        self.assertFalse(verify(c).schema_valid)

    def test_reused_snapshot_wrong_expected_episode(self):
        c = make_capture()
        result = audit_capture(c, expected_campaign_id=CAMPAIGN,
                               expected_episode_id="G2-R2-P1", expected_cluster_uid=CLUSTER_UID)
        self.assertFalse(result.schema_valid)

    def test_wrong_cluster(self):
        self.assertFalse(audit_capture(make_capture(), expected_campaign_id=CAMPAIGN,
                        expected_episode_id=EPISODE, expected_cluster_uid="different").schema_valid)

    def test_wrong_namespace_uid(self):
        c = make_capture()
        c["identity"]["namespace_uid"] = "invented-uid"
        self.assertFalse(verify(c).schema_valid)

    def test_wrong_source_revision(self):
        c = make_capture()
        c["identity"]["sregym_sha"] = "new-unpinned"
        self.assertFalse(verify(c).schema_valid)

    def test_changed_raw_body_without_hash_change(self):
        c = make_capture()
        c["records"]["pods"]["raw_json"] += " "
        self.assertFalse(verify(c).schema_valid)

    def test_wrong_resource_kind(self):
        c = make_capture()
        put_item(c, "services", {"kind": "ConfigMap", "apiVersion": "v1",
                "metadata": {"namespace": NS, "name": "frontend", "uid": "svc1", "resourceVersion": "1"}})
        self.assertFalse(verify(c).schema_valid)

    def test_missing_list_resource_version(self):
        c = make_capture()
        b = json.loads(c["records"]["pods"]["raw_json"])
        del b["metadata"]["resourceVersion"]
        c["records"]["pods"] = record(b, list_request("pods"))
        self.assertFalse(verify(c).schema_valid)

    def test_stale_list_request(self):
        c = make_capture()
        c["records"]["pods"]["request"]["query"]["resourceVersion"] = "0"
        self.assertFalse(verify(c).schema_valid)

    def test_truncated_paginated_list(self):
        c = make_capture()
        c["records"]["pods"]["request"]["query"]["limit"] = 1
        self.assertFalse(verify(c).schema_valid)

    def test_continue_token_in_response(self):
        c = make_capture()
        b = json.loads(c["records"]["pods"]["raw_json"])
        b["metadata"]["continue"] = "next-page-token"
        c["records"]["pods"] = record(b, list_request("pods"))
        self.assertFalse(verify(c).schema_valid)

    def test_selector_filtered_list_rejected(self):
        c = make_capture()
        c["records"]["pods"]["request"]["query"]["labelSelector"] = "app=frontend"
        self.assertFalse(verify(c).schema_valid)

    def test_missing_endpoint_slices(self):
        c = make_capture()
        del c["records"]["endpointslices"]
        self.assertFalse(verify(c).schema_valid)

    def test_duplicate_uid(self):
        c = make_capture()
        put_item(c, "pods", make_pod())
        put_item(c, "pods", make_pod(name="pod2", ip="10.0.0.2"))
        self.assertFalse(verify(c).schema_valid)

    def test_duplicate_running_pod_ip(self):
        c = make_capture()
        put_item(c, "pods", make_pod())
        put_item(c, "pods", make_pod(name="pod2", uid="pod-2"))
        self.assertFalse(verify(c).schema_valid)

    def test_hostnetwork_pods_on_same_node_may_share_node_ip(self):
        c = make_capture()
        a = make_pod(name="host-a", uid="host-a-uid", ip="192.0.2.10")
        b = make_pod(name="host-b", uid="host-b-uid", ip="192.0.2.10")
        for pod in (a, b):
            pod["spec"] = {"hostNetwork": True, "nodeName": "worker-1"}
            put_item(c, "pods", pod)
        result = verify(c)
        self.assertTrue(result.schema_valid, result.reasons)
        self.assertFalse(result.execution_admissible)

    def test_same_ip_on_distinct_hostnetwork_nodes_is_rejected(self):
        c = make_capture()
        a = make_pod(name="host-a", uid="host-a-uid", ip="192.0.2.10")
        b = make_pod(name="host-b", uid="host-b-uid", ip="192.0.2.10")
        a["spec"] = {"hostNetwork": True, "nodeName": "worker-1"}
        b["spec"] = {"hostNetwork": True, "nodeName": "worker-2"}
        put_item(c, "pods", a)
        put_item(c, "pods", b)
        self.assertFalse(verify(c).schema_valid)

    def test_no_default_replica_count(self):
        c = make_capture()
        d = make_deployment()
        del d["spec"]["replicas"]
        put_item(c, "deployments", d)
        self.assertFalse(verify(c).schema_valid)

    def test_stale_generation_recorded_as_unstable(self):
        c = make_capture()
        d = make_deployment()
        d["status"]["observedGeneration"] = 1
        put_item(c, "deployments", d)
        result = verify(c)
        self.assertTrue(result.schema_valid, result.reasons)
        self.assertTrue(result.unstable_reasons)
        self.assertFalse(result.execution_admissible)

    def test_readiness_discrepancy_not_silently_normalized(self):
        c = make_capture()
        d = make_deployment()
        d["status"]["readyReplicas"] = 999
        put_item(c, "deployments", d)
        result = verify(c)
        self.assertTrue(result.schema_valid, result.reasons)
        self.assertTrue(result.unstable_reasons)
        self.assertFalse(result.execution_admissible)

    def test_bracket_detects_resource_version_change(self):
        first, second = make_capture(), make_capture()
        put_item(first, "pods", make_pod())
        put_item(second, "pods", make_pod(rv="19"))
        b = check_observed_item_bracket(first, second)
        self.assertFalse(b.same_observed_items)
        self.assertFalse(b.covers_intervening_events)

    def test_bracket_equal_is_not_atomic_snapshot(self):
        first, second = make_capture(), make_capture()
        put_item(first, "pods", make_pod())
        put_item(second, "pods", make_pod())
        b = check_observed_item_bracket(first, second)
        self.assertTrue(b.same_observed_items)
        self.assertFalse(b.covers_intervening_events)
        self.assertFalse(b.cross_kind_atomic)

if __name__ == "__main__":
    unittest.main()
