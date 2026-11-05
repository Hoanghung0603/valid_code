"""Submission exporter for formatting JSON, compression, and strict file size safety."""

import json
import logging
import os
import zipfile
from typing import Any, Dict, List

from src.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class SubmissionExporter:
    """Exports retrieval results to competition JSON and compressed ZIP files (<100MB)."""

    def __init__(self, max_zip_mb: float = 100.0, target_safe_mb: float = 98.0):
        self.max_zip_mb = max_zip_mb
        self.target_safe_mb = target_safe_mb

    def export(
        self,
        results: List[RetrievalResult],
        out_json: str,
        out_zip: str,
        initial_top_chunks: int = 98
    ) -> float:
        """
        Dumps results to JSON, compresses to ZIP with maximum compression (level 9),
        and automatically trims chunks if size approaches or exceeds competition limits.
        Returns final ZIP size in MB.
        """
        submission_data = [res.to_dict() for res in results]

        # 1. Initial JSON write
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(submission_data, f, ensure_ascii=False, separators=(",", ":"))

        # 2. Level 9 Deflate ZIP
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(out_json, arcname="submission.json")

        zip_sz_mb = os.path.getsize(out_zip) / (1024 * 1024)

        # 3. Dynamic Chunk Trimming if approaching safety threshold
        if zip_sz_mb >= (self.max_zip_mb - 1.0):
            logger.warning(
                f"ZIP size is {zip_sz_mb:.2f} MB (nearing {self.max_zip_mb} MB threshold). "
                f"Auto-adjusting top_chunks to guarantee compliance..."
            )
            max_c = initial_top_chunks - 2
            while zip_sz_mb >= self.target_safe_mb and max_c > 10:
                for item in submission_data:
                    item["relevant_chunks"] = item["relevant_chunks"][:max_c]

                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(submission_data, f, ensure_ascii=False, separators=(",", ":"))

                with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                    zf.write(out_json, arcname="submission.json")

                zip_sz_mb = os.path.getsize(out_zip) / (1024 * 1024)
                logger.info(f"Adjusted to top_chunks={max_c}: zip size={zip_sz_mb:.2f} MB")
                max_c -= 2

        logger.info(f"Exported JSON: {out_json}")
        logger.info(f"Exported ZIP: {out_zip} ({zip_sz_mb:.2f} MB)")
        if zip_sz_mb < self.max_zip_mb:
            logger.info("Compliant with competition size limit (<100 MB).")
        else:
            logger.error(f"WARNING: File size exceeds {self.max_zip_mb} MB limit!")

        return zip_sz_mb
