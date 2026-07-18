from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Common interface every storage backend must implement."""

    @abstractmethod
    def upload(self, local_file_path: str, dest_name: str) -> str:
        """Upload a file, return a storage_ref (drive file_id or local path)."""
        raise NotImplementedError

    @abstractmethod
    def get_url(self, storage_ref: str) -> str:
        """Return a URL/path the client can use to fetch the file."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, storage_ref: str) -> None:
        raise NotImplementedError
