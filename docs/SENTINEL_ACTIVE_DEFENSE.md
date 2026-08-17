# Koschei Sentinel Active Defense Doctrine

Koschei Sentinel is designed as a human-protective, attacker-denying cyber defense intelligence system. It is not limited to alert generation. Its operating goal is to understand an attack as a connected system event, interrupt hostile progression inside defended environments, preserve evidence, verify outcomes, and learn from the incident.

## Core principle

**Protect humans. Deny attackers. Preserve evidence. Maintain control.**

Sentinel treats telemetry as a cyber state, not as isolated alerts. Identity, endpoint, process, credential, repository, CI/CD, artifact, cloud, wallet, transaction, protocol, malware, network and threat-actor observations are represented in a Cyber State Graph. Relations are explicitly classified as OBSERVED, INFERRED, PREDICTED or DISPROVED.

A prediction is never silently promoted into an observation. A disproved path cannot advance the attack state.

## Attack progression engine

The deterministic Attack Progression Engine converts supported graph relations into attack-stage evidence. It can recognize reconnaissance, initial access, execution, persistence, privilege escalation, credential access, discovery, lateral movement, collection, command-and-control, supply-chain compromise, signer or wallet access, exfiltration and impact.

For every active incident it produces:

- evidence-backed active stages,
- a current credible stage,
- explicitly marked predicted next transitions,
- progression confidence,
- active and disproved relation IDs, and
- ranked defensive cut points inside the protected graph.

A defensive cut point is a protected entity where containment can break a large part of the hostile path. Candidate controls include credential revocation, endpoint isolation, malicious process termination, workload quarantine, pipeline pause, signer freeze, transaction hold and IOC blocking. Cut-point ranking combines downstream reach, relation confidence and observed-evidence support.

## Defense modes

### Guard

Used when evidence is incomplete or attack confidence remains below the active-containment threshold. Sentinel gathers evidence, observes progression and may perform low-impact defensive blocking. Higher-impact cut points remain withheld.

### Combat

Used for a corroborated attack when hostile progression is active or protected assets are at risk. Sentinel may contain the attack within defended systems by revoking credentials, terminating sessions, killing malicious processes, isolating endpoints, quarantining workloads, pausing pipelines, freezing signers and holding transactions.

### Siege

Used for a high-confidence, active attack against critical assets with broad blast-radius risk. Sentinel may activate pre-defined emergency defensive policy in addition to Combat actions.

## Deterministic authority bridge

The Active Defense Planner connects progression analysis to defense authority:

`Cyber State Graph -> Attack Progression -> Critical Asset Risk -> Attack Assessment -> Guard/Combat/Siege -> Permitted Cut Points`

The reasoning model may propose hypotheses and enrich the graph, but it cannot bypass this authority bridge. The selected defense mode determines which containment actions are permitted. A lower-confidence incident can therefore expose a useful cut point while still withholding the corresponding higher-impact action until the evidence threshold is met.

The command-line planner is:

`sentinel-active-defense-plan --graph <graph.json> --critical <entity-id>`

`--critical` may be repeated for protected critical entities such as production pipelines, treasury signers, control-plane workloads or other assets represented in the graph.

## Interception sequencing

The Interception Planner converts authorized cut points into a containment sequence. It never promotes a withheld cut point into an executable step.

When impact is predicted at the edge of the defended system, immediate-impact controls take precedence over broad graph centrality. For example, a pending hostile transaction or exposed signer can be held or frozen before the planner continues backward through the root path to compromised credentials, pipelines, endpoints or workloads.

This produces a two-part containment strategy:

1. **Stop the imminent effect.** Prevent the hostile state transition that is about to damage a protected asset.
2. **Collapse the attack path.** Remove the credentials, sessions, processes, pipelines, endpoints or workloads that allowed the attacker to reach that point.

Every interception step requires post-action verification. A high-effect cut point may end the immediate sequence when verification proves that the hostile progression has been broken, but Sentinel must continue hunting for displaced or alternate attacker paths.

## Defense boundary

Active defense applies to systems, identities, endpoints, networks, cloud resources, wallets, signers, pipelines and protocols that the defender is authorized to protect. Sentinel is not designed to retaliate against or compromise external systems. The objective is rapid containment and recovery, not hack-back.

## Evidence and execution

A model assertion alone does not become an execution fact. Authorized defense execution must be linked to precondition evidence, and every material action should produce post-action evidence so Sentinel can verify whether containment succeeded, failed or displaced the attacker to another path.

The intended loop is:

`SEE -> CORRELATE -> UNDERSTAND -> PREDICT -> VERIFY -> FIND CUT POINT -> INTERCEPT -> VERIFY OUTCOME -> COLLAPSE PATH -> HUNT -> RECOVER -> LEARN`

This doctrine is the execution counterpart to the Cyber Corpus v3 knowledge plane and the future Sentinel Cyber World Model.
