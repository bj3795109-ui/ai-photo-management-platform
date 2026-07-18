import os
from storage.local_backend import LocalStorageBackend


def get_storage_backend():
    backend = os.getenv("STORAGE_BACKEND", "local")
    if backend == "local":
        return LocalStorageBackend()
    if backend == "drive":
        from storage.drive_backend import DriveStorageBackend
        # credentials, folder_id = load_drive_credentials()
        # return DriveStorageBackend(credentials, folder_id)
        raise NotImplementedError("Set up Drive credentials before using this backend")
    raise ValueError(f"Unknown STORAGE_BACKEND: {backend}")
