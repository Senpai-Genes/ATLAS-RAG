
import shutil
import threading
import uuid

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from indexing.ingestion_manager import IngestionManager


class IndexingJobManager:
    def __init__(
        self,
        max_workers=1,
        collections_root="data/collections",
    ):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers
        )

        self.collections_root = Path(
            collections_root
        )

        self.jobs = {}
        self.active_collections = set()
        self.lock = threading.Lock()

    def create_job(
        self,
        pdf_path,
        collection_name,
        cleanup_dir=None,
    ):
        job_id = str(uuid.uuid4())

        job = {
            "job_id": job_id,
            "collection": collection_name,
            "status": "queued",
            "progress": 0,
            "error": None,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        with self.lock:
            collection_path = (
                self.collections_root /
                collection_name
            )

            if (
                collection_name in self.active_collections
                or collection_path.exists()
            ):
                raise FileExistsError(
                    f"Collection '{collection_name}' "
                    "already exists or is being indexed."
                )

            self.active_collections.add(
                collection_name
            )

            self.jobs[job_id] = job

        try:
            self.executor.submit(
                self._run_job,
                job_id,
                str(pdf_path),
                collection_name,
                str(cleanup_dir)
                if cleanup_dir
                else None,
            )
        except Exception:
            with self.lock:
                self.active_collections.discard(
                    collection_name
                )
                self.jobs.pop(job_id, None)
            raise

        return job.copy()

    def _update(self, job_id, **changes):
        with self.lock:
            self.jobs[job_id].update(
                changes
            )

    def _run_job(
        self,
        job_id,
        pdf_path,
        collection_name,
        cleanup_dir,
    ):
        self._update(
            job_id,
            status="processing",
            progress=10,
        )

        try:
            manager = IngestionManager(
                collections_root=str(
                    self.collections_root
                )
            )

            manager.ingest(
                pdf_path=pdf_path,
                collection_name=collection_name,
            )

            self._update(
                job_id,
                status="completed",
                progress=100,
            )

        except Exception as error:
            self._update(
                job_id,
                status="failed",
                progress=0,
                error=str(error),
            )

        finally:
            if cleanup_dir:
                shutil.rmtree(
                    cleanup_dir,
                    ignore_errors=True,
                )

            with self.lock:
                self.active_collections.discard(
                    collection_name
                )

    def get_job(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)

            return (
                job.copy()
                if job is not None
                else None
            )


indexing_jobs = IndexingJobManager()
