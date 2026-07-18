import os
import shutil
from storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.getenv("LOCAL_STORAGE_DIR", "./data/photos")
        os.makedirs(self.base_dir, exist_ok=True)

    def upload(self, local_file_path: str, dest_name: str) -> str:
        dest_path = os.path.join(self.base_dir, dest_name)
        shutil.copy2(local_file_path, dest_path)
        return dest_path

    def get_url(self, storage_ref: str) -> str:
        return storage_ref

    def delete(self, storage_ref: str) -> None:
        if os.path.exists(storage_ref):
            os.remove(storage_ref)
