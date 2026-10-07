"""Read-only evidence recovery for the approved image; no model/GPU execution."""
import argparse
import json
from pathlib import Path
import subprocess
from b_common import digest, require, write_json

IMAGE = "dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04"
FILES = ["srt/models/qwen3_moe.py", "srt/models/qwen2_moe.py", "srt/models/utils.py",
         "srt/layers/communicator.py", "srt/layers/layernorm.py", "srt/layers/rotary_embedding/base.py",
         "srt/layers/attention/aiter_backend.py", "srt/layers/vocab_parallel_embedding.py",
         "srt/layers/quantization/unquant.py", "srt/mem_cache/memory_pool.py",
         "srt/model_executor/model_runner.py", "srt/model_executor/cuda_graph_runner.py",
         "srt/model_executor/forward_batch_info.py", "srt/managers/scheduler.py",
         "srt/managers/schedule_batch.py", "srt/managers/scheduler_profiler_mixin.py",
         "srt/server_args.py", "srt/entrypoints/http_server.py", "srt/managers/io_struct.py"]


def recover(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    commands = []

    def command(argv, filename):
        commands.append(argv)
        p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        (output / filename).write_text(p.stdout)
        (output / (filename + ".stderr")).write_text(p.stderr)
        require(p.returncode == 0, f"inspection command failed ({p.returncode}): {argv[0]}; see {output}")
        return p.stdout

    try:
        info = json.loads(command(["docker", "image", "inspect", "--format",
                                   '{{json .Id}} {{json .RepoDigests}}', IMAGE], "image.txt").split(" ", 1)[0])
        # Use immutable local image ID for subsequent reads, even if tag moves.
        prefix = ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp", "--entrypoint", "python3", info, "-c"]
        script = """import importlib.util, importlib.metadata, json, pathlib
spec=importlib.util.find_spec('sglang')
root=pathlib.Path(spec.origin).parent
versions={}
for p in ['sglang','torch','aiter']:
 try: versions[p]=importlib.metadata.version(p)
 except importlib.metadata.PackageNotFoundError: versions[p]=None
print(json.dumps({'root':str(root),'versions':versions}))
"""
        metadata = json.loads(command(prefix + [script], "versions.json"))
        for filename in FILES:
            script = "from pathlib import Path; print(Path(" + repr(metadata["root"] + "/" + filename) + ").read_text(), end='')"
            # Missing source is evidence, retained without silently substituting upstream.
            try:
                command(prefix + [script], filename.replace("/", "__"))
            except ValueError:
                continue
        command(["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp",
                 "--entrypoint", "bash", info, "-lc",
                 "command -v rocprofv3; rocprofv3 --version; rocprofv3 --help; ldconfig -p | grep roctx; dpkg-query -W '*rocprof*' '*rocm*' '*hipblaslt*' '*rccl*'"], "tools.txt")
    finally:
        write_json(output / "commands.json", commands)
        write_json(output / "sha256.json", {p.name: digest(p) for p in output.iterdir() if p.is_file()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", help="new evidence directory on an accessible host")
    a = p.parse_args()
    recover(a.output)


if __name__ == "__main__":
    main()
