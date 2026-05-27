"""Profile gpt-oss with TokenSpeed's in-process Engine API.

This is a workaround for the SMG-fronted ``tokenspeed serve`` path where the
old TokenSpeed ``/start_profile`` and ``/stop_profile`` HTTP controls are not
publicly exposed. It reproduces the requested random benchmark shape in-process:

    tokenspeed serve \
      --model amd/gpt-oss-120b-w-mxfp4-a-fp8 \
      --host 127.0.0.1 \
      --port 21000 \
      --world-size 1 \
      --disable-kvstore \
      --no-enable-prefix-caching

    tokenspeed bench \
      --backend tokenspeed \
      --host 127.0.0.1 \
      --port 21000 \
      --model amd/gpt-oss-120b-w-mxfp4-a-fp8 \
      --dataset-name random \
      --random-input-len 8192 \
      --random-output-len 1024 \
      --random-range-ratio 0.0 \
      --num-prompts 80 \
      --max-concurrency 8 \
      --request-rate inf \
      --seed 1

Edit ``BenchmarkConfig`` below to change the hardcoded benchmark values.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from tokenspeed.bench import RandomDataset, get_tokenizer
from tokenspeed.runtime.entrypoints.engine import Engine
from tokenspeed.runtime.utils.server_args import prepare_server_args


@dataclass
class BenchmarkConfig:
    model: str = "amd/gpt-oss-120b-w-mxfp4-a-fp8"
    host: str = "127.0.0.1"
    port: int = 21000
    world_size: int = 1
    enforce_eager: bool = True
    random_input_len: int = 8192
    random_output_len: int = 1024
    random_range_ratio: float = 0.0
    random_prefix_len: int = 0
    num_prompts: int = 16
    max_concurrency: int = 8
    seed: int = 1
    warmup_requests: int = 8
    profile_dir: str = "./gpt-oss-profile"
    profile_id: str | None = None
    activities: tuple[str, ...] = ("CPU", "GPU")
    profile_start_step: int | None = None
    profile_num_steps: int = 32
    profile_by_stage: bool = False
    record_shapes: bool = True
    with_stack: bool = True
    profile_enabled: bool = True


@dataclass
class RequestResult:
    success: bool
    latency: float
    ttft: float | None
    chunks: int
    prompt_len: int
    expected_output_len: int
    error: str | None = None


def build_server_argv(config: BenchmarkConfig) -> list[str]:
    # Keep these defaults aligned with the serve command in the prompt.
    argv = [
        "--model",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--world-size",
        str(config.world_size),
        "--disable-kvstore",
        "--no-enable-prefix-caching",
    ]
    if config.enforce_eager:
        argv.append("--enforce-eager")
    return argv


def build_requests(config: BenchmarkConfig):
    tokenizer = get_tokenizer(config.model)
    dataset = RandomDataset(random_seed=config.seed)
    requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=config.num_prompts,
        request_id_prefix="bench-",
        input_len=config.random_input_len,
        output_len=config.random_output_len,
        range_ratio=config.random_range_ratio,
        prefix_len=config.random_prefix_len,
    )
    return tokenizer, requests


async def run_one(
    engine: Engine, request: Any, sem: asyncio.Semaphore
) -> RequestResult:
    async with sem:
        start = time.perf_counter()
        ttft = None
        chunks = 0
        try:
            stream = await engine.async_generate(
                prompt=request.prompt,
                sampling_params={
                    "max_new_tokens": request.expected_output_len,
                    "ignore_eos": True,
                    "repetition_penalty": 1.0,
                },
                stream=True,
                user_rid=request.request_id,
            )
            async for _chunk in stream:
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
            return RequestResult(
                success=True,
                latency=time.perf_counter() - start,
                ttft=ttft,
                chunks=chunks,
                prompt_len=request.prompt_len,
                expected_output_len=request.expected_output_len,
            )
        except Exception as exc:  # noqa: BLE001
            return RequestResult(
                success=False,
                latency=time.perf_counter() - start,
                ttft=ttft,
                chunks=chunks,
                prompt_len=request.prompt_len,
                expected_output_len=request.expected_output_len,
                error=repr(exc),
            )


async def run_requests(
    engine: Engine,
    requests,
    *,
    max_concurrency: int,
) -> list[RequestResult]:
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [asyncio.create_task(run_one(engine, request, sem)) for request in requests]
    return await asyncio.gather(*tasks)


async def maybe_start_profile(config: BenchmarkConfig, engine: Engine) -> None:
    if not config.profile_enabled:
        return
    os.environ["TOKENSPEED_PROFILER_DIR"] = config.profile_dir
    activities = list(config.activities)
    print(
        "Starting profiler: "
        f"dir={config.profile_dir}, activities={activities}, "
        f"profile_id={config.profile_id or '<auto>'}"
    )
    await engine.tokenizer_manager.start_profile(
        output_dir=config.profile_dir,
        activities=activities,
        profile_id=config.profile_id,
        start_step=config.profile_start_step,
        num_steps=config.profile_num_steps,
        profile_by_stage=config.profile_by_stage,
        with_stack=config.with_stack,
        record_shapes=config.record_shapes,
    )


async def maybe_stop_profile(config: BenchmarkConfig, engine: Engine) -> None:
    if not config.profile_enabled:
        return
    try:
        await engine.tokenizer_manager.stop_profile()
    except RuntimeError as exc:
        # Step-limited or stage profiling can auto-stop before the benchmark
        # driver reaches this point.
        if "Profiling is not in progress" not in str(exc):
            raise
    print(f"Profiler output dir: {config.profile_dir}")


def print_summary(results: list[RequestResult], duration: float) -> None:
    completed = sum(1 for result in results if result.success)
    failed = len(results) - completed
    total_input = sum(result.prompt_len for result in results)
    expected_output = sum(result.expected_output_len for result in results)
    successful = [result for result in results if result.success]
    mean_latency = (
        sum(result.latency for result in successful) / completed if completed else 0.0
    )
    ttfts = [result.ttft for result in successful if result.ttft is not None]
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0

    print("=" * 50)
    print("GPT-OSS Engine Benchmark Result")
    print("=" * 50)
    print(f"Successful requests: {completed}")
    print(f"Failed requests: {failed}")
    print(f"Benchmark duration (s): {duration:.2f}")
    if duration > 0:
        print(f"Request throughput (req/s): {completed / duration:.2f}")
        print(
            f"Expected output token throughput (tok/s): {expected_output / duration:.2f}"
        )
        print(
            f"Total token throughput (tok/s): {(total_input + expected_output) / duration:.2f}"
        )
    print(f"Total input tokens: {total_input}")
    print(f"Expected output tokens: {expected_output}")
    print(f"Mean latency (ms): {mean_latency * 1000:.2f}")
    print(f"Mean TTFT (ms): {mean_ttft * 1000:.2f}")
    for idx, result in enumerate(results):
        if not result.success:
            print(f"Request {idx} failed: {result.error}")


async def main() -> None:
    config = BenchmarkConfig()
    if config.max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if config.profile_start_step is not None and config.profile_start_step <= 0:
        raise ValueError("profile_start_step must be positive")
    if config.profile_num_steps <= 0:
        raise ValueError("profile_num_steps must be positive")

    server_argv = build_server_argv(config)
    print("Engine args:", " ".join(server_argv))
    _, requests = build_requests(config)

    server_args = prepare_server_args(server_argv)
    engine = Engine(server_args=server_args)
    try:
        warmup_count = min(config.warmup_requests, len(requests))
        if warmup_count > 0:
            print(f"Running {warmup_count} warmup request(s)...")
            await run_requests(
                engine,
                requests[:warmup_count],
                max_concurrency=min(config.max_concurrency, warmup_count),
            )

        await maybe_start_profile(config, engine)
        print(
            f"Running {len(requests)} benchmark requests "
            f"with max_concurrency={config.max_concurrency}..."
        )
        start = time.perf_counter()
        results = await run_requests(
            engine,
            requests,
            max_concurrency=config.max_concurrency,
        )
        duration = time.perf_counter() - start
        await maybe_stop_profile(config, engine)
        print_summary(results, duration)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
