#!/usr/bin/env python3
"""
SonarQube container lifecycle, mode switching (Standard <-> MQR), rule catalog dump,
and full issue + Quality Gate extraction client for D04-Plus.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


class SonarQubeInstance:
    def __init__(
        self,
        label: str,
        image: str,
        port: int,
        scanner_bin: Path,
        mode: Optional[str] = None,  # None (9.9/10.3), "STANDARD_EXPERIENCE", or "MQR"
        profile_name: Optional[str] = None,  # Optional built-in profile override, e.g., "Sonar way comprehensive"
        cpus: str = "4.5",
        memory: str = "6.5g",
    ) -> None:
        self.label = label
        self.image = image
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.container_name = f"d04p_{label}"
        self.scanner_bin = scanner_bin
        self.mode = mode
        self.profile_name = profile_name
        self.cpus = cpus
        self.memory = memory
        self.admin_pw = "D04Plus_LocalOnly_2026!"
        self.auth: Tuple[str, str] = ("admin", "admin")
        self.token: str = ""
        self.valid_metrics: set[str] = set()
        self._py_rule_keys: List[str] = []

    def start_clean(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            f"--cpus={self.cpus}",
            f"--memory={self.memory}",
            "-p",
            f"127.0.0.1:{self.port}:9000",
            "-e",
            "SONAR_ES_BOOTSTRAP_CHECKS_DISABLE=true",
            "-e",
            "SONAR_WEB_JAVAOPTS=-Xmx1024m -Xms512m",
            "-e",
            "SONAR_CE_JAVAOPTS=-Xmx3328m -Xms512m",
            "-e",
            "SONAR_SEARCH_JAVAOPTS=-Xmx1536m -Xms1024m",
            self.image,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        self._wait_until_up()
        self._init_credentials_and_mode()

    def stop_and_remove(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)

    def _wait_until_up(self, timeout_sec: int = 240) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/api/system/status", timeout=5)
                if r.status_code == 200:
                    st = r.json().get("status")
                    if st == "UP":
                        return
            except requests.RequestException:
                pass
            time.sleep(3)
        logs = subprocess.run(
            ["docker", "logs", "--tail", "80", self.container_name],
            capture_output=True,
            text=True,
            errors="replace",
        ).stdout
        raise RuntimeError(f"SonarQube {self.label} failed to reach UP within {timeout_sec}s:\n{logs}")

    def _init_credentials_and_mode(self) -> None:
        # Change default admin password if fresh
        r = requests.post(
            f"{self.base_url}/api/users/change_password",
            auth=("admin", "admin"),
            data={
                "login": "admin",
                "previousPassword": "admin",
                "password": self.admin_pw,
            },
            timeout=15,
        )
        if r.status_code in (200, 204):
            self.auth = ("admin", self.admin_pw)
        else:
            # Verify which password works
            chk = requests.get(f"{self.base_url}/api/authentication/validate", auth=("admin", self.admin_pw), timeout=10)
            if chk.ok and chk.json().get("valid"):
                self.auth = ("admin", self.admin_pw)
            else:
                self.auth = ("admin", "admin")

        # Generate analysis token
        tok_name = f"tok_{int(time.time())}"
        tr = requests.post(
            f"{self.base_url}/api/user_tokens/generate",
            auth=self.auth,
            data={"name": tok_name},
            timeout=15,
        )
        tr.raise_for_status()
        self.token = tr.json()["token"]

        # Switch MQR / Standard Experience mode if requested (SonarQube 10.8+ / 25.x / 26.x)
        if self.mode is not None:
            cur_r = requests.get(f"{self.base_url}/api/v2/clean-code-policy/mode", auth=self.auth, timeout=15)
            cur_mode = cur_r.json().get("mode") if cur_r.ok else None
            if cur_mode != self.mode:
                candidates = [self.mode]
                if self.mode == "STANDARD_EXPERIENCE":
                    candidates.append("STANDARD")
                switched = False
                last_err = ""
                for cand in candidates:
                    mr = requests.patch(
                        f"{self.base_url}/api/v2/clean-code-policy/mode",
                        auth=self.auth,
                        json={"mode": cand},
                        headers={"Content-Type": "application/json"},
                        timeout=20,
                    )
                    if mr.status_code in (200, 204):
                        switched = True
                        break
                    last_err = f"{mr.status_code} {mr.text}"
                if not switched:
                    raise RuntimeError(f"Failed setting mode={self.mode} on {self.label}: {last_err}")

        # Set global New Code Period to PREVIOUS_VERSION so sequential release scans compute exact deltas
        requests.post(
            f"{self.base_url}/api/new_code_periods/set",
            auth=self.auth,
            data={"type": "PREVIOUS_VERSION"},
            timeout=15,
        )

        # Optional override of default Python quality profile (e.g. "Sonar way comprehensive" on SQ 26.11)
        if self.profile_name:
            pr_set = requests.post(
                f"{self.base_url}/api/qualityprofiles/set_default",
                auth=self.auth,
                data={"language": "py", "qualityProfile": self.profile_name},
                timeout=15,
            )
            if pr_set.status_code not in (200, 204):
                raise RuntimeError(
                    f"Failed setting Python default profile={self.profile_name} on {self.label}: {pr_set.status_code} {pr_set.text}"
                )

        # Discover supported metric keys on this exact SonarQube version
        p = 1
        while True:
            mr = requests.get(f"{self.base_url}/api/metrics/search", auth=self.auth, params={"ps": 500, "p": p}, timeout=15)
            if not mr.ok:
                break
            d = mr.json()
            batch = d.get("metrics", [])
            for m in batch:
                if "key" in m:
                    self.valid_metrics.add(m["key"])
            if p * 500 >= d.get("total", 0) or not batch:
                break
            p += 1

    def get_mode_and_gates_metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"label": self.label, "image": self.image, "profile_name": self.profile_name}
        st = requests.get(f"{self.base_url}/api/system/status", auth=self.auth, timeout=10).json()
        meta["sq_version"] = st.get("version")
        mr = requests.get(f"{self.base_url}/api/v2/clean-code-policy/mode", auth=self.auth, timeout=10)
        if mr.ok:
            meta["clean_code_mode"] = mr.json()
        pr = requests.get(
            f"{self.base_url}/api/qualityprofiles/search",
            auth=self.auth,
            params={"language": "py"},
            timeout=15,
        )
        if pr.ok:
            meta["py_profiles"] = pr.json().get("profiles", [])
        qgl = requests.get(f"{self.base_url}/api/qualitygates/list", auth=self.auth, timeout=10)
        if qgl.ok:
            gates = qgl.json().get("qualitygates", [])
            meta["quality_gates"] = []
            for g in gates:
                gid = g.get("id") or g.get("name")
                gs = requests.get(
                    f"{self.base_url}/api/qualitygates/show",
                    auth=self.auth,
                    params={"name": g["name"]},
                    timeout=10,
                )
                meta["quality_gates"].append(gs.json() if gs.ok else g)
        return meta

    def fetch_active_python_rules(self) -> List[Dict[str, Any]]:
        # Find active Python quality profile key (matching self.profile_name or isDefault)
        pr = requests.get(
            f"{self.base_url}/api/qualityprofiles/search",
            auth=self.auth,
            params={"language": "py"},
            timeout=15,
        )
        pr.raise_for_status()
        profiles = pr.json().get("profiles", [])
        prof_key = None
        if self.profile_name:
            for p in profiles:
                if p.get("name") == self.profile_name:
                    prof_key = p["key"]
                    break
        if not prof_key:
            for p in profiles:
                if p.get("isDefault"):
                    prof_key = p["key"]
                    break
        if not prof_key and profiles:
            prof_key = profiles[0]["key"]

        rules: List[Dict[str, Any]] = []
        page = 1
        while True:
            params: Dict[str, Any] = {"languages": "py", "ps": 500, "p": page}
            if prof_key:
                params["qprofile"] = prof_key
                params["activation"] = "true"
            rr = requests.get(f"{self.base_url}/api/rules/search", auth=self.auth, params=params, timeout=20)
            rr.raise_for_status()
            data = rr.json()
            batch = data.get("rules", [])
            rules.extend(batch)
            total = data.get("total", 0)
            if page * 500 >= total or not batch:
                break
            page += 1
        return rules

    def ensure_project_previous_version_newcode(self, project_key: str, project_name: str) -> None:
        requests.post(
            f"{self.base_url}/api/projects/create",
            auth=self.auth,
            data={"project": project_key, "name": project_name},
            timeout=15,
        )
        if self.profile_name:
            requests.post(
                f"{self.base_url}/api/qualityprofiles/add_project",
                auth=self.auth,
                data={"language": "py", "project": project_key, "qualityProfile": self.profile_name},
                timeout=15,
            )
        # Set New Code Period to PREVIOUS_VERSION
        requests.post(
            f"{self.base_url}/api/new_code_periods/set",
            auth=self.auth,
            data={"project": project_key, "type": "PREVIOUS_VERSION"},
            timeout=15,
        )

    def scan_snapshot_and_collect(
        self,
        wt_dir: Path,
        project_key: str,
        release_tag: str,
        project_date_iso: str,
    ) -> Dict[str, Any]:
        work_tmp = wt_dir / ".scannerwork_d04p"
        auth_arg = (
            f"-Dsonar.login={self.token}"
            if ("9.9" in self.image or "10.3" in self.image)
            else f"-Dsonar.token={self.token}"
        )
        cmd = [
            str(self.scanner_bin),
            f"-Dsonar.host.url={self.base_url}",
            auth_arg,
            f"-Dsonar.projectKey={project_key}",
            f"-Dsonar.projectName={project_key}",
            f"-Dsonar.projectVersion={release_tag}",
            "-Dsonar.sources=.",
            "-Dsonar.inclusions=**/*.py",
            "-Dsonar.exclusions=sympy/integrals/rubi/**",
            "-Dsonar.sourceEncoding=UTF-8",
            "-Dsonar.scm.disabled=true",
            f"-Dsonar.working.directory={work_tmp}",
            "-Dsonar.cpd.exclusions=**/*",
        ]
        env = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "SONAR_SCANNER_OPTS": "-Xmx3072m"}
        res = subprocess.run(cmd, cwd=str(wt_dir), env=env, capture_output=True, text=True, errors="replace", timeout=2400)
        if res.returncode != 0:
            raise RuntimeError(
                f"sonar-scanner failed on {project_key}@{release_tag} ({self.label}):\nSTDOUT:\n{res.stdout[-1200:]}\nSTDERR:\n{res.stderr[-1200:]}"
            )

        # Wait for Compute Engine task from report-task.txt
        report_task = work_tmp / "report-task.txt"
        ce_task_id = ""
        if report_task.exists():
            for line in report_task.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("ceTaskId="):
                    ce_task_id = line.split("=", 1)[1].strip()
                    break
        if ce_task_id:
            self._wait_ce_task(ce_task_id)
        else:
            time.sleep(3)

        # Collect Quality Gate status, component measures, and all OPEN issues (plus isNew via inNewCodePeriod!)
        qg = requests.get(
            f"{self.base_url}/api/qualitygates/project_status",
            auth=self.auth,
            params={"projectKey": project_key},
            timeout=15,
        ).json()

        desired_metrics = [
            "ncloc",
            "files",
            "violations",
            "new_violations",
            "bugs",
            "new_bugs",
            "vulnerabilities",
            "new_vulnerabilities",
            "code_smells",
            "new_code_smells",
            "blocker_violations",
            "critical_violations",
            "major_violations",
            "minor_violations",
            "info_violations",
            "sqale_index",
            "new_technical_debt",
            "sqale_debt_ratio",
            "new_sqale_debt_ratio",
            "reliability_rating",
            "new_reliability_rating",
            "security_rating",
            "new_security_rating",
            "sqale_rating",
            "new_maintainability_rating",
            "software_quality_maintainability_issues",
            "software_quality_reliability_issues",
            "software_quality_security_issues",
            "software_quality_blocker_issues",
            "software_quality_high_issues",
            "software_quality_medium_issues",
            "software_quality_low_issues",
            "software_quality_info_issues",
            "new_software_quality_maintainability_rating",
            "new_software_quality_reliability_rating",
            "new_software_quality_security_rating",
        ]
        active_keys = [k for k in desired_metrics if (not self.valid_metrics or k in self.valid_metrics)]
        metric_keys = ",".join(active_keys)
        meas_resp = requests.get(
            f"{self.base_url}/api/measures/component",
            auth=self.auth,
            params={"component": project_key, "metricKeys": metric_keys},
            timeout=15,
        )
        measures = meas_resp.json() if meas_resp.ok else {}

        all_issues = self._fetch_issues(project_key, in_new_code=False)
        new_issue_keys = {
            iss["key"] for iss in self._fetch_issues(project_key, in_new_code=True)
        }
        compact_issues = []
        prefix = f"{project_key}:"
        for iss in all_issues:
            comp = iss.get("component", "")
            rpath = comp[len(prefix) :] if comp.startswith(prefix) else comp
            compact_issues.append(
                {
                    "key": iss.get("key"),
                    "rule": iss.get("rule"),
                    "file": rpath,
                    "line": iss.get("line") or (iss.get("textRange") or {}).get("startLine") or 0,
                    "hash": iss.get("hash") or "",
                    "type": iss.get("type"),
                    "severity": iss.get("severity"),
                    "cleanCodeAttribute": iss.get("cleanCodeAttribute"),
                    "cleanCodeAttributeCategory": iss.get("cleanCodeAttributeCategory"),
                    "impacts": iss.get("impacts") or [],
                    "effort": iss.get("effort") or "",
                    "is_new": iss.get("key") in new_issue_keys,
                }
            )

        return {
            "quality_gate": qg.get("projectStatus", {}),
            "measures": measures.get("component", {}),
            "issues": compact_issues,
        }

    def _wait_ce_task(self, ce_task_id: str, timeout_sec: int = 1200) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"{self.base_url}/api/ce/task",
                    auth=self.auth,
                    params={"id": ce_task_id},
                    timeout=20,
                )
            except requests.RequestException:
                time.sleep(1.5)
                continue
            if r.ok:
                task = r.json().get("task", {})
                st = task.get("status")
                if st == "SUCCESS":
                    return
                if st in ("FAILED", "CANCELED"):
                    raise RuntimeError(f"CE task {ce_task_id} ended with status={st}: {task}")
            time.sleep(0.8)
        raise RuntimeError(f"CE task {ce_task_id} timed out after {timeout_sec}s")

    def _paginate_all_pages(self, params: Dict[str, Any], seen: set[str], out: List[Dict[str, Any]]) -> int:
        r1 = requests.get(f"{self.base_url}/api/issues/search", auth=self.auth, params={**params, "p": 1}, timeout=25)
        if not r1.ok:
            return 0
        d1 = r1.json()
        tot = int(d1.get("total", 0))
        if tot == 0:
            return 0
        if tot > 9500:
            return tot  # Caller must sub-shard
        for it in d1.get("issues", []):
            if it["key"] not in seen:
                seen.add(it["key"])
                out.append(it)
        p = 2
        while (p - 1) * 500 < tot and p <= 20:
            rp = requests.get(f"{self.base_url}/api/issues/search", auth=self.auth, params={**params, "p": p}, timeout=25)
            if not rp.ok:
                break
            batch = rp.json().get("issues", [])
            if not batch:
                break
            for it in batch:
                if it["key"] not in seen:
                    seen.add(it["key"])
                    out.append(it)
            p += 1
        return tot

    def _fetch_issues(self, project_key: str, in_new_code: bool = False) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        base_params: Dict[str, Any] = {
            "componentKeys": project_key,
            "statuses": "OPEN",
            "ps": 500,
        }
        if in_new_code:
            if "9.9" in self.image:
                base_params["sinceLeakPeriod"] = "true"
            else:
                base_params["inNewCodePeriod"] = "true"

        r1 = requests.get(f"{self.base_url}/api/issues/search", auth=self.auth, params={**base_params, "p": 1}, timeout=25)
        if not r1.ok and in_new_code:
            alt_key = "sinceLeakPeriod" if "inNewCodePeriod" in base_params else "inNewCodePeriod"
            base_params.pop("inNewCodePeriod", None)
            base_params.pop("sinceLeakPeriod", None)
            base_params[alt_key] = "true"
            r1 = requests.get(f"{self.base_url}/api/issues/search", auth=self.auth, params={**base_params, "p": 1}, timeout=25)
        r1.raise_for_status()
        d1 = r1.json()
        total = int(d1.get("total", 0))
        if total <= 9500:
            issues.extend(d1.get("issues", []))
            p = 2
            while (p - 1) * 500 < total:
                rp = requests.get(f"{self.base_url}/api/issues/search", auth=self.auth, params={**base_params, "p": p}, timeout=25)
                rp.raise_for_status()
                batch = rp.json().get("issues", [])
                if not batch:
                    break
                issues.extend(batch)
                p += 1
            return issues

        # Hierarchical sharding (>9500 issues on large monorepos like transformers / sympy):
        # Level 1: severity tier; Level 2: active rule chunks; Level 3: single rule x top-level directory
        seen: set[str] = set()
        shard_field = "impactSeverities" if ("26." in self.image and self.mode == "MQR") else "severities"
        shard_vals = (
            ["INFO", "LOW", "MEDIUM", "HIGH", "BLOCKER"]
            if shard_field == "impactSeverities"
            else ["INFO", "MINOR", "MAJOR", "CRITICAL", "BLOCKER"]
        )
        if not self._py_rule_keys:
            self._py_rule_keys = [r["key"] for r in self.fetch_active_python_rules()]

        for sv in shard_vals:
            sv_params = {**base_params, shard_field: sv}
            sv_tot = self._paginate_all_pages(sv_params, seen, issues)
            if sv_tot <= 9500:
                continue
            # Sub-shard by rule batches of 20
            rkeys = self._py_rule_keys
            for i in range(0, len(rkeys), 20):
                chunk = rkeys[i : i + 20]
                ch_params = {**sv_params, "rules": ",".join(chunk)}
                ch_tot = self._paginate_all_pages(ch_params, seen, issues)
                if ch_tot <= 9500:
                    continue
                # Split down to individual rules
                for rk in chunk:
                    rk_params = {**sv_params, "rules": rk}
                    rk_tot = self._paginate_all_pages(rk_params, seen, issues)
                    if rk_tot <= 9500:
                        continue
                    # Split single hyper-prolific rule by all paginated repository directories + root files
                    dirs: List[str] = []
                    dp_page = 1
                    while True:
                        tr = requests.get(
                            f"{self.base_url}/api/components/tree",
                            auth=self.auth,
                            params={"component": project_key, "qualifiers": "DIR", "ps": 500, "p": dp_page},
                            timeout=25,
                        )
                        if not tr.ok:
                            break
                        td = tr.json()
                        comps = td.get("components", [])
                        dirs.extend(c["path"] for c in comps if c.get("path"))
                        if dp_page * 500 >= int(td.get("paging", {}).get("total", 0)) or not comps:
                            break
                        dp_page += 1
                    for dp in dirs:
                        d_params = {**rk_params, "directories": dp}
                        self._paginate_all_pages(d_params, seen, issues)
                    self._paginate_all_pages({**rk_params, "directories": "/"}, seen, issues)
        return issues
