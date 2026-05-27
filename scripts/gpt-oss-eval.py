"""Evaluate gpt-oss through the same SMG/OpenAI path used by CI.

This is a local POC wrapper around the CI eval shape for MI355. It starts
``ts serve`` as a subprocess, waits for readiness, then runs EvalScope against
the OpenAI-compatible endpoint.  It intentionally avoids common ports such as
8000 and 21000 so it can run next to other local serving experiments.

Equivalent CI-style commands:

    CI=true ts serve \
      --model amd/gpt-oss-120b-w-mxfp4-a-fp8 \
      --attn-tp-size 2 \
      --moe-tp-size 2 \
      --max-model-len 80000 \
      --trust-remote-code \
      --reasoning-parser base \
      --disable-kvstore \
      --no-enable-prefix-caching \
      --sampling-backend greedy \
      --enable-cache-report \
      --host 127.0.0.1 \
      --port 21513

    /tmp/evalscope-perf/bin/evalscope eval \
      --model amd/gpt-oss-120b-w-mxfp4-a-fp8 \
      --api-url http://127.0.0.1:21513/v1 \
      --api-key EMPTY_TOKEN \
      --datasets gpqa_diamond \
      --eval-batch-size 16

Edit ``EvalConfig`` below to change the hardcoded values.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalConfig:
    model: str = "amd/gpt-oss-120b-w-mxfp4-a-fp8"
    host: str = "127.0.0.1"
    port: int = 21513
    attn_tp_size: int = 2
    moe_tp_size: int = 2
    max_model_len: int = 80000
    dataset: str = "gpqa_diamond"
    eval_batch_size: int = 16
    sampling_backend: str = "greedy"
    api_key: str = "EMPTY_TOKEN"
    evalscope_venv: str = "/tmp/evalscope-perf"
    install_evalscope: bool = True
    limit: int | None = None
    work_dir: str | None = None
    score_threshold: float | None = 0.7
    startup_timeout_s: int = 1800
    readiness_interval_s: int = 10
    shutdown_timeout_s: int = 30


def _print_section_header(title: str, fill: str = "=") -> None:
    print(f"{title:{fill}^70}", flush=True)


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def build_serve_cmd(config: EvalConfig) -> list[str]:
    return [
        "ts",
        "serve",
        "--model",
        config.model,
        "--attn-tp-size",
        str(config.attn_tp_size),
        "--moe-tp-size",
        str(config.moe_tp_size),
        "--max-model-len",
        str(config.max_model_len),
        "--trust-remote-code",
        "--reasoning-parser",
        "base",
        "--disable-kvstore",
        "--no-enable-prefix-caching",
        "--sampling-backend",
        config.sampling_backend,
        "--enable-cache-report",
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]


def evalscope_bin(config: EvalConfig) -> str:
    if env_bin := os.environ.get("EVALSCOPE_BIN"):
        return env_bin
    return str(Path(config.evalscope_venv) / "bin" / "evalscope")


def build_evalscope_cmd(config: EvalConfig) -> list[str]:
    cmd = [
        evalscope_bin(config),
        "eval",
        "--model",
        config.model,
        "--api-url",
        f"http://{config.host}:{config.port}/v1",
        "--api-key",
        config.api_key,
        "--datasets",
        config.dataset,
        "--eval-batch-size",
        str(config.eval_batch_size),
    ]
    if config.limit is not None:
        cmd.extend(["--limit", str(config.limit)])
    if config.work_dir is not None:
        cmd.extend(["--work-dir", config.work_dir])
    return cmd


def run_checked(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(_format_cmd(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def install_evalscope(config: EvalConfig) -> None:
    if not config.install_evalscope:
        return

    python_bin = str(Path(config.evalscope_venv) / "bin" / "python")
    _print_section_header(" Installing EvalScope ")
    run_checked(
        [
            "python3",
            "-m",
            "uv",
            "venv",
            "--seed",
            "--clear",
            config.evalscope_venv,
        ]
    )
    run_checked(
        [
            "python3",
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            python_bin,
            "evalscope[perf]",
        ]
    )


class StreamedProcess:
    def __init__(
        self,
        cmd: list[str],
        *,
        prefix: str,
        env: dict[str, str] | None = None,
    ) -> None:
        self.cmd = cmd
        self.prefix = prefix
        self.output: list[str] = []
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        self._thread = threading.Thread(target=self._stream_output, daemon=True)
        self._thread.start()

    def _stream_output(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.output.append(line)
            print(f"[{self.prefix}] {line}", end="", flush=True)

    def wait(self) -> int:
        return_code = self.process.wait()
        self._thread.join(timeout=5)
        return return_code

    def terminate(self, timeout_s: int) -> None:
        if self.process.poll() is not None:
            self._thread.join(timeout=5)
            return

        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            self._thread.join(timeout=5)
            return
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
        finally:
            self._thread.join(timeout=5)

    @property
    def stdout_text(self) -> str:
        return "".join(self.output)


def wait_for_readiness(config: EvalConfig, server: StreamedProcess) -> None:
    url = f"http://{config.host}:{config.port}/readiness"
    deadline = time.time() + config.startup_timeout_s
    last_error: Exception | None = None

    print(f"Waiting for readiness: {url}", flush=True)
    while time.time() < deadline:
        return_code = server.process.poll()
        if return_code is not None:
            raise RuntimeError(f"server exited before readiness: code={return_code}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    print("Server is ready.", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(config.readiness_interval_s)

    raise TimeoutError(
        f"server did not become ready within {config.startup_timeout_s}s: "
        f"last_error={last_error!r}"
    )


def _iter_evalscope_scores(value):
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in {
                "score",
                "accuracy",
                "acc",
                "averageaccuracy",
            } and isinstance(item, (int, float)):
                yield float(item)
            else:
                yield from _iter_evalscope_scores(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_evalscope_scores(item)


def _extract_evalscope_table(output: str, marker: str) -> str | None:
    marker_index = output.find(marker)
    if marker_index < 0:
        return None

    table_lines: list[str] = []
    for line in output[marker_index + len(marker) :].splitlines():
        stripped = line.strip()
        if not stripped:
            if table_lines:
                break
            continue
        table_lines.append(line.rstrip())

    return "\n".join(table_lines) if table_lines else None


def _parse_evalscope_report_table(report_table: str) -> float | None:
    score_index: int | None = None
    score: float | None = None
    for line in report_table.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("|", "│")):
            continue
        separator = "│" if stripped.startswith("│") else "|"
        cells = [cell.strip() for cell in stripped.strip(separator).split(separator)]
        if any(set(cell) <= {"=", "-"} for cell in cells if cell):
            continue
        normalized = [cell.lower() for cell in cells]
        if "score" in normalized:
            score_index = normalized.index("score")
            continue
        if score_index is None or len(cells) <= score_index:
            continue
        try:
            score = float(cells[score_index])
        except ValueError:
            continue
    return score


def _parse_evalscope_stdout(output: str) -> float | None:
    report_table = _extract_evalscope_table(output, "Overall report table:")
    if report_table:
        return _parse_evalscope_report_table(report_table)
    return None


def _load_evalscope_score(config: EvalConfig, output: str) -> float | None:
    scores: list[float] = []
    if config.work_dir is not None:
        reports_dir = Path(config.work_dir) / "reports"
        if reports_dir.is_dir():
            for path in reports_dir.rglob("*.json"):
                try:
                    scores.extend(_iter_evalscope_scores(json.loads(path.read_text())))
                except (OSError, json.JSONDecodeError):
                    continue

    if scores:
        return sum(scores) / len(scores)
    return _parse_evalscope_stdout(output)


def run_evalscope(config: EvalConfig, env: dict[str, str]) -> str:
    cmd = build_evalscope_cmd(config)
    _print_section_header(" Running EvalScope ")
    print(_format_cmd(cmd), flush=True)
    proc = StreamedProcess(cmd, prefix="evalscope", env=env)
    try:
        return_code = proc.wait()
    except BaseException:
        proc.terminate(config.shutdown_timeout_s)
        raise
    if return_code != 0:
        raise RuntimeError(f"evalscope failed: code={return_code}")
    return proc.stdout_text


def validate_score(config: EvalConfig, evalscope_output: str) -> None:
    score = _load_evalscope_score(config, evalscope_output)
    if score is None:
        message = "No EvalScope score found in output."
        if config.score_threshold is None:
            print(message, flush=True)
            return
        raise RuntimeError(message)

    print(f"EvalScope score: {score:g}", flush=True)
    if config.score_threshold is not None and score < config.score_threshold:
        raise RuntimeError(
            f"EvalScope score {score:g} is below threshold "
            f"{config.score_threshold:g}"
        )


def main() -> None:
    config = EvalConfig()
    env = os.environ.copy()
    env["CI"] = "true"

    install_evalscope(config)

    serve_cmd = build_serve_cmd(config)
    _print_section_header(" Starting Server ")
    print(_format_cmd(serve_cmd), flush=True)
    server = StreamedProcess(serve_cmd, prefix="serve", env=env)
    try:
        wait_for_readiness(config, server)
        evalscope_output = run_evalscope(config, env)
        validate_score(config, evalscope_output)
    finally:
        _print_section_header(" Stopping Server ")
        server.terminate(config.shutdown_timeout_s)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
