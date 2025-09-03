from __future__ import annotations
from time import sleep
import json
from loguru import logger
import orjson
from collections.abc import Iterable
from contextlib import contextmanager
from pydantic import TypeAdapter
from itertools import product
from collections import ChainMap
from concurrent.futures import ProcessPoolExecutor
import base64
from io import TextIOBase
import gzip

from collections import deque
from collections.abc import Sequence
from pathlib import Path

import click

from collections.abc import Iterator
from pydantic import BaseModel

import time
from functools import wraps


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info("{}.{} : {}".format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


@contextmanager
def timeblock(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        logger.info("{} : {}".format(label, end - start))


@click.group()
def cli():
    pass


class Edge(BaseModel):
    depth: int  # depth of stack frame
    caller: Func
    callee: Func
    file: str
    line: int


class Func(BaseModel):
    file: str
    module: str | None
    func: str

    def __str__(self):
        return f"{self.module}.{self.func}"

    def __eq__(self, other):
        return self.file == other.file and self.func == other.func

    def __hash__(self):
        return hash((self.file, self.func))


class Call(BaseModel):
    callee: Func
    file: str
    line: int
    uid: int

    def __eq__(self, other: object):
        if not isinstance(other, Call):
            return False
        return (
            self.callee == other.callee
            and self.file == other.file
            and self.line == other.line
            and self.uid == other.uid
        )

    def __hash__(self):
        return hash((self.callee, self.file, self.line))


class Stack(BaseModel):
    calls: Sequence[Call]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stack):
            return False

        return len(self.calls) == len(other.calls) and all(
            x == y for x, y in zip(self.calls, other.calls)
        )

    def __iter__(self) -> Iterator[Call]:  # type:ignore
        return iter(self.calls)

    def __len__(self) -> int:
        return len(self.calls)

    def __hash__(self) -> int:
        return hash(tuple(self.calls))

    def __getitem__(self, x):
        return self.calls[x]


class Graph(BaseModel):
    edges: Sequence[Edge]

    def __iter__(self) -> Iterator[Edge]:  # type:ignore
        return iter(self.edges)

    def __len__(self) -> int:
        return len(self.edges)


class ReconstructCallstacks:
    @timethis
    @staticmethod
    def run(trace: Iterable[Edge]) -> Iterable[Stack]:
        stack: deque[Call] = deque()
        depths: deque[int] = deque()
        last_callee = None
        for uid, edge in enumerate(trace):
            depth, caller, callee = edge.depth, edge.caller, edge.callee
            last_depth = depths[-1] if depths else -1
            if not stack:
                stack.append(Call(callee=caller, file="", line=-1, uid=-1))

            if depth > last_depth:
                assert last_callee is None or (caller == last_callee), (
                    depth,
                    caller,
                    callee,
                )
                stack.append(
                    Call(callee=edge.callee, file=edge.file, line=edge.line, uid=uid)
                )
                depths.append(depth)
            else:
                assert depths
                assert last_callee

                yield Stack(calls=stack.copy())

                while depths and depth < depths[-1]:
                    stack.pop()
                    depths.pop()
                assert (not depths) or (depth == depths[-1])

                if depths:
                    stack.pop()
                    depths.pop()
                assert (not stack) or stack[-1].callee == caller
                stack.append(
                    Call(callee=callee, file=edge.file, line=edge.line, uid=uid)
                )
                depths.append(depth)

            last_callee = callee

        if stack:
            yield Stack(calls=stack)


def _is_strict_prefix(shorter: Sequence[Call], longer: Sequence[Call]) -> bool:
    if len(shorter) >= len(longer):
        return False

    return all(x == y for x, y in zip(shorter, longer, strict=False))


def filter_stack(stack: Stack) -> Stack:
    return Stack(
        calls=[
            Call(
                callee=Func(
                    file=call.callee.file.removeprefix("/testbed/"),
                    module=call.callee.module,
                    func=call.callee.func,
                ),
                file=call.file,
                line=call.line,
                uid=call.uid,
            )
            for call in stack.calls
            if is_relevant_file(call.callee.file)
        ]
    )


def is_relevant_file(file: str) -> bool:
    return (
        "/tests/" not in file
        and "/site-packages/" not in file
        and "miniconda" not in file
        and "/root/trace.py" not in file
        and "<frozen importlib" not in file
    )


@logger.catch()
def main():
    dev_funcs_map = load_dev_funcs_map()

    with ProcessPoolExecutor(1) as executor:
        future_map = {}
        while True:
            task_dirs = Path(
                "run_instance_swt_logs/assertflip/pred_pre__AssertFlip"
            ).glob("*/")
            task_dirs = [t for t in task_dirs if t.joinpath("test_output.txt").exists()]

            new_task_dirs = sorted(set(task_dirs) - set(future_map))
            new_task_ids = [t.name for t in new_task_dirs]
            new_dev_funcs = [dev_funcs_map.get(t) for t in new_task_ids]
            new_out_dirs = [
                Path("assertflip_analysis", "fresh_results", t) for t in new_task_ids
            ]

            for task_dir, out_dir, dev_func in zip(
                new_task_dirs, new_out_dirs, new_dev_funcs
            ):
                future = executor.submit(process_task_dir, task_dir, out_dir, dev_func)
                future_map[task_dir] = future

            sleep(10)


def load_dev_funcs_map() -> dict[str, set[Func]]:
    with open("assertflip_analysis/norm.dev_locations.jsonl") as f:
        dev_locs_map = ChainMap(*map(json.loads, f))

    def dev_loc_to_func(dev_loc: dict) -> Func:
        if dev_loc["start"] is None:
            func_name = "<module>"
        elif dev_loc["class_name"]:
            func_name = f"{dev_loc['class_name']}.{dev_loc['method_name']}"
        else:
            func_name = dev_loc["method_name"]
        return Func(file=dev_loc["rel_file_path"], module="", func=func_name)

    return {k: set(map(dev_loc_to_func, v)) for k, v in dev_locs_map.items()}


def process_task_dir(
    task_dir: Path, out_dir: Path, dev_funcs: set[Func] | None = None
) -> None:
    logger.add(out_dir / "fresh_analysis.log")

    logger.info("Processing {}", str(task_dir))

    test_output_file = task_dir / "test_output.txt"
    if not test_output_file.exists():
        logger.error("test output file not found")
        return

    with open(test_output_file) as f, timeblock("extract graph b64 from log"):
        b64 = extract_graph_b64(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("graph.b64").write_text(b64)

    process_graph_b64(b64, out_dir, dev_funcs)


def extract_graph_b64(f: TextIOBase) -> str:
    for line in f:
        if "Call stacks json base64:" in line:
            return next(f).rstrip()
    raise RuntimeError("call stack b64 not found in test output")


def process_graph_b64(
    b64_str: str, out_dir: Path, dev_funcs: set[Func] | None = None
) -> None:
    graph = decode_object(b64_str, Graph)
    assert isinstance(graph, Graph)

    # dump clean graph
    clean_edges = [e for e in graph if is_relevant_file(e.callee.file)]
    lines = (
        f"{e.file},{e.line},{e.caller!s},{e.callee!s},{e.depth}\n" for e in clean_edges
    )
    with timeblock("compute & dump lines"), open(out_dir / "clean.trace.csv", "w") as f:
        f.write("file,line,caller,callee,depth\n")
        f.writelines(lines)

    # reconstruct stacks from call graph
    stacks = list(ReconstructCallstacks.run(graph.edges))
    logger.info("found {} stacks", len(stacks))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stacks.b64l", "w") as f:
        with ProcessPoolExecutor(10) as executor, timeblock("encode stacks"):
            stack_strings = list(executor.map(encode_object, stacks, chunksize=1000))

        with timeblock("write stacks"):
            for s in stack_strings:
                f.write(s)
                f.write("\n")

    # remove irrelevant functions from stacks
    clean_stacks = map(filter_stack, stacks)
    clean_stacks = filter(None, clean_stacks)
    with timeblock("filter stacks"):
        clean_stacks = list(clean_stacks)

    logger.info("found {} clean stacks", len(clean_stacks))

    with open(out_dir / "clean.stacks.b64l", "w") as f:
        with ProcessPoolExecutor(10) as executor, timeblock("encode clean stacks"):
            stack_strings = list(
                executor.map(encode_object, clean_stacks, chunksize=1000)
            )

        for s in stack_strings:
            f.write(s)
            f.write("\n")

    dev_funcs = dev_funcs or set()
    if not dev_funcs:
        logger.warning("no dev funcs found; exiting")
        raise SystemExit

    def is_relevant(stack: Stack) -> bool:
        return any(dev_f == call.callee for dev_f, call in product(dev_funcs, stack))

    relevant_stacks = list(filter(is_relevant, clean_stacks))
    logger.info("found {} relevant clean stacks before dedup", len(relevant_stacks))

    with (
        open(out_dir / "relevant.clean.stacks.b64l", "w") as f,
        timeblock("encode & dump relevant stacks"),
    ):
        for stack in relevant_stacks:
            f.write(encode_object(stack))
            f.write("\n")

    with open(out_dir / "relevant.clean.stacks.json", "wb") as f:
        adapter = TypeAdapter(list[Stack])
        f.write(adapter.dump_json(relevant_stacks, indent=4))

    # deduplicate relevant stacks
    uniq_relevant_stacks = list(dict.fromkeys(relevant_stacks))
    logger.info("found {} relevant clean stacks after dedup", len(uniq_relevant_stacks))

    uniq_relevant_stacks = [
        x
        for x in uniq_relevant_stacks
        if not any(_is_strict_prefix(x.calls, y.calls) for y in uniq_relevant_stacks)
    ]
    logger.info(
        "found {} relevant clean stacks after dedup & subsuming",
        len(uniq_relevant_stacks),
    )

    with (
        open(out_dir / "uniq.relevant.clean.stacks.b64l", "w") as f,
        timeblock("encode & dump relevant stacks"),
    ):
        for stack in uniq_relevant_stacks:
            f.write(encode_object(stack))
            f.write("\n")

    with open(out_dir / "uniq.relevant.clean.stacks.json", "wb") as f:
        adapter = TypeAdapter(list[Stack])
        f.write(adapter.dump_json(uniq_relevant_stacks, indent=4))


def encode_object(model: BaseModel) -> str:
    json_str = model.model_dump_json()
    compressed = gzip.compress(json_str.encode("utf-8"), compresslevel=1)
    b64_str = base64.b64encode(compressed).decode("ascii")
    return b64_str


@timethis
def decode_object(b64_str: str, model_type: type[BaseModel] | None = None):
    compressed = base64.b64decode(b64_str)
    json_bytes = gzip.decompress(compressed)
    if model_type is not None:
        return model_type.model_validate_json(json_bytes.decode("utf-8"))
    else:
        return orjson.loads(json_bytes)


def test_process_graph_b64():
    b64 = "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"
    process_graph_b64(b64, Path("/tmp/foo"))


def test_reconstruct_callstacks():
    f, g, h, t, y = (Func(file="foo.py", module="foo", func=name) for name in "fghty")
    cf, cg, ch, ct, cy = (
        Call(callee=func, file="foo.py", line=line, uid=-1)
        for func, line in ((f, -1), (g, 2), (h, 10), (t, 11), (y, 3))
    )
    cf.file = ""
    trace = [
        Edge(depth=1, caller=f, callee=g, file="foo.py", line=2),
        Edge(depth=2, caller=g, callee=h, file="foo.py", line=10),
        Edge(depth=2, caller=g, callee=t, file="foo.py", line=11),
        Edge(depth=1, caller=f, callee=y, file="foo.py", line=3),
    ]
    expected_callstacks = [
        Stack(calls=[cf, cg, ch]),
        Stack(calls=[cf, cg, ct]),
        Stack(calls=[cf, cy]),
    ]
    try:
        result = ReconstructCallstacks.run(trace)
        assert list(result) == expected_callstacks
    except Exception:
        logger.exception("exception when reconstructing")


if __name__ == "__main__":
    main()
