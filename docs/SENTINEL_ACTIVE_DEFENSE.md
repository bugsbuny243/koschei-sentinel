# Koschei Sentinel Active Defense Doctrine

Koschei Sentinel is designed as a human-protective, attacker-denying cyber defense intelligence system. It is not limited to alert generation. Its operating goal is to understand an attack as a connected system event, interrupt hostile progression inside defended environments, preserve evidence, verify outcomes, and learn from the incident.

## Core principle

**Protect humans. Deny attackers. Preserve evidence. Maintain control.**

Sentinel treats telemetry as a cyber state, not as isolated alerts. Identity, endpoint, process, credential, repository, CI/CD, artifact, cloud, wallet, transaction, protocol, malware, network and threat-actor observations are represented in a Cyber State Graph. Relations are explicitly classified as OBSERVED, INFERRED, PREDICTED or DISPROVED.

## Defense modes

### Guard

Used when evidence is incomplete or attack confidence remains below the active-containment threshold. Sentinel gathers evidence, observes progression and may perform low-impact defensive blocking.

### Combat

Used for a corroborated attack when hostile progression is active or protected assets are at risk. Sentinel may contain the attack within defended systems by revoking credentials, terminating sessions, killing malicious processes, isolating endpoints, quarantining workloads, pausing pipelines, freezing signers and holding transactions.

### Siege

Used for a high-confidence, active attack against critical assets with broad blast-radius risk. Sentinel may activate pre-defined emergency defensive policy in addition to Combat actions.

## Defense boundary

Active defense applies to systems, identities, endpoints, networks, cloud resources, wallets, signers, pipelines and protocols that the defender is authorized to protect. Sentinel is not designed to retaliate against or compromise external systems. The objective is rapid containment and recovery, not hack-back.

## Evidence and execution

A model assertion alone does not become an execution fact. Authorized defense execution must be linked to precondition evidence, and every material action should produce post-action evidence so Sentinel can verify whether containment succeeded, failed or displaced the attacker to another path.

The intended loop is:

`SEE -> CORRELATE -> UNDERSTAND -> PREDICT -> VERIFY -> CONTAIN -> VERIFY OUTCOME -> HUNT -> RECOVER -> LEARN`

This doctrine is the execution counterpart to the Cyber Corpus v3 knowledge plane and the future Sentinel Cyber World Model.
