from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from storage.base import StorageBackend

# Requires: google-api-python-client, google-auth-oauthlib
# Reuses the same OAuth flow/credentials as the Google Photos ingestion source.


class DriveStorageBackend(StorageBackend):
    def __init__(self, credentials, folder_id: str):
        self.credentials = credentials
        self.folder_id = folder_id
        self.service = build("drive", "v3", credentials=credentials)

    def upload(self, local_file_path: str, dest_name: str) -> str:
        media = MediaFileUpload(local_file_path, resumable=True)
        file = self.service.files().create(
            body={"name": dest_name, "parents": [self.folder_id]},
            media_body=media, fields="id"
        ).execute()
        return file["id"]
        raise NotImplementedError("Wire up Drive API client here")

    def get_url(self, storage_ref: str) -> str:
        return f"https://drive.google.com/uc?id={storage_ref}"

    def delete(self, storage_ref: str) -> None:
        self.service.files().delete(fileId=storage_ref).execute()
        raise NotImplementedError("Wire up Drive API client here")
