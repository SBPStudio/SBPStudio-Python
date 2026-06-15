"""
test_cli_smoke.py — Smoke tests that run each CLI subcommand via subprocess.

All commands must exit 0 and produce the expected output files.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _run(args: list, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "sbp_studio.cli.main"] + args
    return subprocess.run(
        cmd, capture_output=True, text=True,
        check=check, timeout=120,
    )


class TestCLISmoke:
    def test_info(self, simple_segy):
        result = _run(["info", simple_segy])
        assert result.returncode == 0
        assert "Traces" in result.stdout or "trazas" in result.stdout.lower()

    def test_info_json(self, simple_segy):
        import json
        result = _run(["info", "--json", simple_segy])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert data[0]["n_traces"] == 50

    def test_reproject(self, simple_segy, tmp_path):
        src = str(tmp_path / "rep_src.sgy")
        shutil.copy(simple_segy, src)
        result = _run(["reproject", src,
                        "--src", "EPSG:4326",
                        "--dst", "EPSG:32630",
                        "--out-dir", str(tmp_path)])
        assert result.returncode == 0
        out_files = list(tmp_path.glob("*_REPROY*.sgy"))
        assert len(out_files) >= 1

    def test_export_image_png(self, simple_segy, tmp_path):
        out = str(tmp_path / "out.png")
        result = _run(["export-image", simple_segy,
                        "--format", "png", "--out", out])
        assert result.returncode == 0
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_spectrum(self, spectrum_segy, tmp_path):
        out = str(tmp_path / "spectrum.png")
        result = _run(["spectrum", spectrum_segy, "--out", out])
        assert result.returncode == 0
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_navline_geojson(self, simple_segy, tmp_path):
        out = str(tmp_path / "nav.geojson")
        result = _run(["navline", simple_segy,
                        "--format", "geojson", "--out", out])
        assert result.returncode == 0
        assert os.path.exists(out)

    def test_navline_csv(self, simple_segy, tmp_path):
        out = str(tmp_path / "nav.csv")
        result = _run(["navline", simple_segy,
                        "--format", "csv", "--out", out])
        assert result.returncode == 0
        assert os.path.exists(out)

    def test_navline_shp(self, simple_segy, tmp_path):
        base = str(tmp_path / "nav")
        result = _run(["navline", simple_segy,
                        "--format", "shp", "--out", base + ".shp"])
        assert result.returncode == 0
        assert os.path.exists(base + ".shp")

    def test_fix_geojson(self, simple_segy, tmp_path):
        out = str(tmp_path / "fix.geojson")
        result = _run(["fix", simple_segy,
                        "--interval", "1",
                        "--format", "geojson",
                        "--out", out])
        assert result.returncode == 0

    def test_join_chain(self, chain_pair, tmp_path):
        path1, path2 = chain_pair
        src1, src2 = str(tmp_path / "c1.sgy"), str(tmp_path / "c2.sgy")
        shutil.copy(path1, src1); shutil.copy(path2, src2)
        out = str(tmp_path / "joined.sgy")
        result = _run(["join-chain", src1, src2,
                        "--src", "EPSG:4326",
                        "--dst", "EPSG:4326",    # identity transform = pure join
                        "--gap-km", "1.0",
                        "--out", out])
        assert result.returncode == 0
        assert os.path.exists(out)

    def test_invalid_crs_nonzero_exit(self, simple_segy, tmp_path):
        src = str(tmp_path / "bad.sgy")
        shutil.copy(simple_segy, src)
        result = _run(["reproject", src,
                        "--src", "NOT_A_CRS",
                        "--dst", "EPSG:32630"],
                       check=False)
        assert result.returncode != 0
