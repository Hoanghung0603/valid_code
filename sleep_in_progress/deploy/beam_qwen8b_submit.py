"""Submit/monitor Qwen3-Reranker-8B rerank tasks on Beam.

Usage:
  python deploy/beam_qwen8b_submit.py bench
  python deploy/beam_qwen8b_submit.py full
  python deploy/beam_qwen8b_submit.py status
  python deploy/beam_qwen8b_submit.py stop <task_id>
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"

    if mode == "bench":
        from beam_qwen8b import rerank_shard
        t = rerank_shard.put(
            cands_file="cands_bench50.json",
            out_file="qrr8b/bench/qwen8b_bench.json",
            resume_dir="qrr8b/bench/parts",
            max_len=1024,
            batch=8,
            dtype="fp16",
            limit_queries=0,
        )
        print(f"bench task: {t}")
        if t and hasattr(t, "task_id"):
            print(f"task_id: {t.task_id}")
            print(f"status: {t.status()}")
        return

    if mode == "full":
        from beam_qwen8b import rerank_shard
        ids = []
        for i in range(4):
            t = rerank_shard.put(
                cands_file=f"cands_shard{i}.json",
                out_file=f"qrr8b/shard{i}/qwen8b_top100.json",
                resume_dir=f"qrr8b/shard{i}/parts",
                max_len=1024,
                batch=8,
                dtype="fp16",
                limit_queries=0,
            )
            tid = t.task_id if t and hasattr(t, "task_id") else str(t)
            ids.append(tid)
            print(f"shard{i}: {tid}")
        print("task_ids:", ids)
        return

    if mode == "status":
        from beta9.client.task import TaskClient
        from beam_qwen8b import rerank_shard
        print("Use: beam task list")
        return

    print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
