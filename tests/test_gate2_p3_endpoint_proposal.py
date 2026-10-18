"""Pure synthetic adversarial tests; no Kubernetes, network, or agent calls."""
from __future__ import annotations

import copy
import unittest

from trace_gate2.p3_endpoint_proposal import (
    EndpointNormalizationError, normalize_p3_endpoints, typed_exact_equal,
)

NS = "astronomy-shop"


def service(name, port):
    return {"apiVersion": "v1", "kind": "Service",
            "metadata": {"namespace": NS, "name": name, "uid": "svc-" + name},
            "spec": {"ports": [{"name": name, "port": port, "protocol": "TCP"}]}}


def slice_for(name, port, *, address="10.2.3.4", conditions=None, suffix="1", family="IPv4"):
    return {"apiVersion": "discovery.k8s.io/v1", "kind": "EndpointSlice",
            "metadata": {"namespace": NS, "name": f"{name}-{suffix}", "uid": f"slice-{name}-{suffix}",
                         "labels": {"kubernetes.io/service-name": name}},
            "addressType": family,
            "ports": [{"name": name, "port": port, "protocol": "TCP"}],
            "endpoints": [{"addresses": [address], "conditions": conditions if conditions is not None else {},
                           "targetRef": {"kind": "Pod", "name": f"{name}-pod", "uid": f"pod-{name}"}}]}


def evidence():
    return [service("cart", 8080), service("valkey-cart", 6379)], [
        slice_for("cart", 8080, address="10.2.3.4"),
        slice_for("valkey-cart", 6379, address="10.2.3.5")]


def norm(services, slices):
    return normalize_p3_endpoints(services=services, endpoint_slices=slices, namespace=NS)


class TestP3EndpointProposal(unittest.TestCase):
    def test_expected_cart_and_valkey_dependencies(self):
        services, slices = evidence()
        out = norm(services, slices)
        self.assertEqual(set(out), {"cart", "valkey-cart"})
        self.assertEqual(out["valkey-cart"]["endpoint_histogram"][0]["port"]["number"], 6379)
        self.assertEqual(out["cart"]["endpoint_histogram"][0]["count"], 1)

    def test_same_network_endpoint_duplicate_across_slices_not_double_counted(self):
        services, slices = evidence()
        slices.append(copy.deepcopy(slices[1]))
        slices[-1]["metadata"]["name"] = "valkey-cart-2"
        slices[-1]["metadata"]["uid"] = "slice-2"
        self.assertEqual(norm(services, slices)["valkey-cart"]["endpoint_histogram"][0]["count"], 1)

    def test_order_and_ephemeral_ip_or_uid_do_not_change_fingerprint(self):
        services, slices = evidence()
        baseline = norm(services, slices)
        changed = copy.deepcopy(slices)
        for sl in changed:
            sl["metadata"]["uid"] += "-fresh"
            sl["endpoints"][0]["targetRef"]["uid"] += "-fresh"
            sl["endpoints"][0]["addresses"][0] = "10.20.20.20" if sl["metadata"]["name"].startswith("cart") else "10.20.20.21"
        self.assertTrue(typed_exact_equal(baseline, norm(list(reversed(services)), list(reversed(changed)))))

    def test_omitted_ready_defaults_true(self):
        services, slices = evidence()
        self.assertEqual(norm(services, slices)["cart"]["endpoint_histogram"][0]["ready"], True)

    def test_omitted_serving_follows_false_ready_in_kubernetes_1321(self):
        services, slices = evidence()
        slices[0]["endpoints"][0]["conditions"] = {"ready": False}
        hist = norm(services, slices)["cart"]["endpoint_histogram"][0]
        self.assertEqual((hist["ready"], hist["serving"], hist["terminating"]),
                         (False, False, False))

    def test_explicit_nil_serving_follows_false_ready(self):
        services, slices = evidence()
        slices[0]["endpoints"][0]["conditions"] = {"ready": False, "serving": None}
        self.assertFalse(norm(services, slices)["cart"]["endpoint_histogram"][0]["serving"])

    def test_absent_terminating_defaults_false(self):
        services, slices = evidence()
        self.assertIs(norm(services, slices)["cart"]["endpoint_histogram"][0]["terminating"], False)

    def test_condition_triple_distinguishes_all_eight_states(self):
        seen = set()
        for i in range(8):
            services, slices = evidence()
            condition = {"ready": bool(i & 1), "serving": bool(i & 2), "terminating": bool(i & 4)}
            slices[0]["endpoints"][0]["conditions"] = condition
            result = norm(services, slices)
            seen.add(str(result["cart"]["endpoint_histogram"]))
        self.assertEqual(len(seen), 8)

    def test_ready_true_with_serving_false_is_not_collapsed(self):
        services, slices = evidence()
        slices[0]["endpoints"][0]["conditions"] = {"ready": True, "serving": False}
        record = norm(services, slices)["cart"]["endpoint_histogram"][0]
        self.assertTrue(record["ready"])
        self.assertFalse(record["serving"])

    def test_publish_not_ready_is_semantically_visible(self):
        services, slices = evidence()
        baseline = norm(services, slices)
        services[0]["spec"]["publishNotReadyAddresses"] = True
        changed = norm(services, slices)
        self.assertFalse(typed_exact_equal(baseline, changed))

    def test_conflicting_duplicate_conditions_fail(self):
        services, slices = evidence()
        dup = copy.deepcopy(slices[0]); dup["metadata"]["name"] = "duplicate"
        dup["endpoints"][0]["conditions"] = {"ready": False}
        slices.append(dup)
        with self.assertRaisesRegex(EndpointNormalizationError, "conflicting duplicate"):
            norm(services, slices)

    def test_conflicting_duplicate_pod_target_fails(self):
        services, slices = evidence()
        dup = copy.deepcopy(slices[0]); dup["metadata"]["name"] = "duplicate"
        dup["endpoints"][0]["targetRef"]["uid"] = "different-pod"
        slices.append(dup)
        with self.assertRaisesRegex(EndpointNormalizationError, "conflicting duplicate"):
            norm(services, slices)

    def test_irrelevant_other_service_slice_does_not_change_p3(self):
        services, slices = evidence()
        baseline = norm(services, slices)
        other = slice_for("frontend", 8080, address="10.2.3.99")
        self.assertEqual(baseline, norm(services, slices + [other]))

    def test_wrong_namespace_is_rejected(self):
        services, slices = evidence()
        slices[0]["metadata"]["namespace"] = "other"
        with self.assertRaises(EndpointNormalizationError):
            norm(services, slices)

    def test_relevant_slice_without_service_label_is_rejected(self):
        services, slices = evidence()
        slices[0]["metadata"]["labels"] = {}
        with self.assertRaises(EndpointNormalizationError):
            norm(services, slices)

    def test_missing_service_is_rejected(self):
        services, slices = evidence()
        with self.assertRaises(EndpointNormalizationError):
            norm(services[:1], slices)

    def test_empty_valid_slice_is_distinct_from_missing_capture_validation(self):
        services, slices = evidence()
        slices[0]["endpoints"] = []
        self.assertEqual(norm(services, slices)["cart"]["endpoint_histogram"], [])
        # This is purely a normalization result, NOT admissibility evidence.

    def test_string_boolean_rejected(self):
        services, slices = evidence()
        slices[0]["endpoints"][0]["conditions"] = {"ready": "false"}
        with self.assertRaises(EndpointNormalizationError):
            norm(services, slices)

    def test_port_difference_is_visible(self):
        services, slices = evidence()
        baseline = norm(services, slices)
        slices[0]["ports"][0]["port"] = 8181
        self.assertFalse(typed_exact_equal(baseline, norm(services, slices)))

    def test_address_family_is_visible(self):
        services, slices = evidence()
        baseline = norm(services, slices)
        slices[0]["addressType"] = "IPv6"
        slices[0]["endpoints"][0]["addresses"] = ["2001:db8::3"]
        self.assertFalse(typed_exact_equal(baseline, norm(services, slices)))

    def test_invalid_family_address_is_rejected(self):
        services, slices = evidence()
        slices[0]["addressType"] = "IPv6"  # still has v4 address
        with self.assertRaises(EndpointNormalizationError):
            norm(services, slices)

    def test_does_not_mutate_inputs(self):
        services, slices = evidence()
        snapshot = copy.deepcopy((services, slices))
        norm(services, slices)
        self.assertEqual((services, slices), snapshot)

    def test_typed_comparison_does_not_conflate_boolean_and_integer(self):
        self.assertFalse(typed_exact_equal({"x": True}, {"x": 1}))
        self.assertFalse(typed_exact_equal({"x": [True]}, {"x": [1]}))

    def test_typed_comparison_ignores_dictionary_key_order_only(self):
        self.assertTrue(typed_exact_equal({"a": 1, "b": [2, 3]}, {"b": [2, 3], "a": 1}))
        self.assertFalse(typed_exact_equal([1, 2], [2, 1]))

    def test_floats_require_a_separate_frozen_policy(self):
        with self.assertRaises(ValueError):
            typed_exact_equal({"p": 0.5}, {"p": 0.5})


if __name__ == "__main__":
    unittest.main()
