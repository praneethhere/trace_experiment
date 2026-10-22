# TRACE Gate 2 — offline P4 status projection (review proposal)

**Status:** proposal only; not frozen, not wired into the collector or any episode grader. No SREGym incident data observed.

**Motivation.** Kubernetes v1.32.1 `apps/v1.DeploymentStatus` uses non-pointer int32 fields annotated `json:",omitempty"`, including `readyReplicas`, `availableReplicas`, `unavailableReplicas`, `updatedReplicas` and `replicas`. A zero can legitimately be omitted from the raw API JSON. P4's injected quota failure can produce zero ready search replicas. Treating that omission as invalid evidence can create a spurious candidate failure.

**Source pin:** https://github.com/kubernetes/kubernetes/blob/v1.32.1/staging/src/k8s.io/api/apps/v1/types.go (DeploymentStatus). Parent TRACE protocol SHA-256: `a5e77dbbc08869f07fdd1f2e5c2b4a7e343e44fe970913ae6e908611c3b0d562`.

**Narrow behavior:** only an **absent** optional `Deployment.status` count is decoded as zero. Explicit JSON `null`, strings, booleans, floats or negative values remain malformed. The original raw object is unchanged, and `omitted_zero_fields` records the decoding provenance. The `status` object itself must exist; `metadata.generation` must be a positive integer and `status.observedGeneration` must equal it before candidate-relevant fields can be exposed. Stale/unobserved status is not graded as a recreation failure here.

**Limitations:** This module does not prove the JSON originated from Kubernetes or came from an unfiltered complete LIST; it does not establish consistency across Pod, ReplicaSet, Event and ResourceQuota resources, nor that quota admission failed. `spec.replicas` must be explicit for this *candidate-specific* decoder even though general API defaulting can have different semantics. A complete P4 semantic fingerprint remains unimplemented. No derived value authorizes experiment execution.

**Review requirement:** reconcile with the evidence collector's raw-response provenance and structural admissibility, then prospectively specify typed P1–P4 schemas, source-aware omission policies, phase-scoped classification and fixed retry bounds. This is only the first narrow correction.
