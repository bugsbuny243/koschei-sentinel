"""Offline bundle boundary tests; the compiler and model are never loaded."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "src" / "koschei_sentinel" / "commercial_bundle_v1.py"
spec = importlib.util.spec_from_file_location("bundle_contract_under_test", MODULE_PATH)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class CommercialBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.value = {
            "schema_version": "koschei.commercial-bundle.v1",
            "bundle_id": "synthetic-test-bundle",
            "bundle_version": "0.1.0-test",
            "components": [],
        }
        for product in ("koschei-lang", "koschei-sentinel"):
            content = f"SYNTHETIC TEST ARTIFACT: {product}".encode()
            name = f"{product}.fixture"
            (self.root / name).write_bytes(content)
            self.value["components"].append({
                "product": product, "version": "0.1.0-test", "artifact": name,
                "sha256": hashlib.sha256(content).hexdigest(),
            })

    def test_both_artifacts_verified_without_release_or_execution_authority(self):
        report = bundle.check_bundle(self.value, self.root)
        self.assertEqual(report["status"], "integrity_verified")
        self.assertEqual(report["verified_artifacts"], ["koschei-lang", "koschei-sentinel"])
        self.assertFalse(report["release_approved"])
        self.assertFalse(report["execution_authorized"])

    def test_metadata_check_does_not_claim_artifact_verification(self):
        report = bundle.check_bundle(self.value)
        self.assertEqual(report["status"], "manifest_valid")
        self.assertEqual(report["verified_artifacts"], [])

    def test_missing_duplicate_or_foreign_product_is_rejected(self):
        for products in ([], self.value["components"][:1], [self.value["components"][0]] * 2):
            with self.subTest(products=len(products)), self.assertRaises(ValueError):
                bundle.check_bundle(dict(self.value, components=products))
        self.value["components"][1]["product"] = "external-model"
        with self.assertRaises(ValueError):
            bundle.check_bundle(self.value)

    def test_tampered_or_missing_artifact_cannot_complete_integrity_check(self):
        path = self.root / self.value["components"][1]["artifact"]
        path.write_bytes(b"tampered synthetic data")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            bundle.check_bundle(self.value, self.root)
        path.unlink()
        with self.assertRaises(OSError):
            bundle.check_bundle(self.value, self.root)

    def test_paths_urls_and_case_collisions_are_rejected(self):
        for artifact in ("../escape", "/absolute", "https://host/file", "a\\b", ".", "a\n"):
            value = copy.deepcopy(self.value)
            value["components"][0]["artifact"] = artifact
            with self.subTest(artifact=artifact), self.assertRaises(ValueError):
                bundle.check_bundle(value)
        self.value["components"][1]["artifact"] = self.value["components"][0]["artifact"].upper()
        with self.assertRaisesRegex(ValueError, "distinct"):
            bundle.check_bundle(self.value)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support required")
    def test_artifact_symlink_is_rejected(self):
        artifact = self.root / self.value["components"][0]["artifact"]
        target = self.root / "real-artifact"
        artifact.rename(target)
        artifact.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            bundle.check_bundle(self.value, self.root)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO support required")
    def test_special_file_is_rejected_without_reading(self):
        artifact = self.root / self.value["components"][0]["artifact"]
        artifact.unlink()
        os.mkfifo(artifact)
        with self.assertRaisesRegex(ValueError, "regular"):
            bundle.check_bundle(self.value, self.root)

    def test_unknown_fields_and_forged_readiness_are_rejected(self):
        for field in ("release_approved", "execution_authorized", "secret"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                bundle.check_bundle(dict(self.value, **{field: True}))

    def test_duplicate_json_fields_and_oversized_manifest_are_rejected(self):
        path = self.root / "manifest.json"
        path.write_text('{"schema_version":"one","schema_version":"two"}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bundle.load_bundle(path)
        path.write_bytes(b" " * (bundle.MAX_MANIFEST_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "too large"):
            bundle.load_bundle(path)

    def test_manifest_digest_binds_versions_and_ignores_component_order(self):
        original = bundle.check_bundle(self.value)["manifest_sha256"]
        self.value["components"].reverse()
        self.assertEqual(bundle.check_bundle(self.value)["manifest_sha256"], original)
        self.value["components"][0]["version"] = "0.1.1-test"
        self.assertNotEqual(bundle.check_bundle(self.value)["manifest_sha256"], original)

    def test_cli_verifies_files_and_reports_failure_without_partial_success(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.value))
        command = [sys.executable, str(MODULE_PATH), str(path), "--artifacts-dir", str(self.root)]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "integrity_verified")
        (self.root / self.value["components"][1]["artifact"]).unlink()
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, "")
        self.assertEqual(json.loads(process.stderr)["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
