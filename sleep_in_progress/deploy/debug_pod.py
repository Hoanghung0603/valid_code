"""Debug Pod: find files anywhere in container."""
from beam import Image, Pod, Volume

image = Image(python_version="python3.11")
dv = Volume(name="r2ai-data", mount_path="/data")

pod = Pod(
    name="debug-vol4",
    image=image,
    volumes=[dv],
    cpu=1,
    memory="2Gi",
    entrypoint=["bash", "-c", "find / -name queries.jsonl 2>/dev/null; echo ===; find / -name 'chunkdocs*' 2>/dev/null; echo ===; ls -laR /data; echo DONE"],
    keep_warm_seconds=60,
)
