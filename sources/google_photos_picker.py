"""
Google Photos integration via the Picker API.

As of April 2025, Google restricted the Photos Library API so that new/unverified
apps can no longer list a user's entire existing photo library (`photoslibrary.readonly`
requires a formal Google security verification process). The Picker API is Google's
current recommended approach: the user explicitly selects photos through Google's own
picker UI, and the app receives access only to those selected items.

Flow:
1. create_picker_session()      -> returns a pickerUri to show the user
2. poll_session_until_ready()   -> waits until the user finishes picking
3. list_picked_media_items()    -> lists what they picked
4. download_picked_item()       -> downloads one item's bytes to a local temp file
5. delete_session()             -> cleans up the session (best practice, avoids limits)
"""

import time
import tempfile
import requests

PICKER_BASE_URL = "https://photospicker.googleapis.com/v1"


def _auth_headers(creds) -> dict:
    return {"Authorization": f"Bearer {creds.token}"}


def create_picker_session(creds) -> dict:
    """Creates a new picker session. Returns the session dict, including 'pickerUri'."""
    response = requests.post(
        f"{PICKER_BASE_URL}/sessions",
        headers=_auth_headers(creds),
        json={},
    )
    response.raise_for_status()
    return response.json()


def get_session(creds, session_id: str) -> dict:
    """Fetches current session status. Check 'mediaItemsSet' to know if picking is done."""
    response = requests.get(
        f"{PICKER_BASE_URL}/sessions/{session_id}",
        headers=_auth_headers(creds),
    )
    response.raise_for_status()
    return response.json()


def poll_session_until_ready(creds, session_id: str, timeout_seconds: int = 300) -> dict:
    """Polls the session until the user finishes picking, or timeout is reached."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        session = get_session(creds, session_id)
        if session.get("mediaItemsSet"):
            return session

        poll_interval = session.get("pollingConfig", {}).get("pollInterval", "5s")
        wait_seconds = float(poll_interval.rstrip("s")) if isinstance(poll_interval, str) else 5
        time.sleep(wait_seconds)

    raise TimeoutError(f"User did not finish picking photos within {timeout_seconds}s")


def list_picked_media_items(creds, session_id: str) -> list[dict]:
    """Lists all media items the user picked in this session."""
    items = []
    page_token = None

    while True:
        params = {"sessionId": session_id}
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(
            f"{PICKER_BASE_URL}/mediaItems",
            headers=_auth_headers(creds),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        items.extend(data.get("mediaItems", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return items


def download_picked_item(creds, media_item: dict) -> str:
    """Downloads a single picked media item's bytes to a local temp file.

    Returns the local file path, ready to be passed into ingest_photo().
    """
    base_url = media_item["mediaFile"]["baseUrl"]
    filename = media_item["mediaFile"].get("filename", "photo.jpg")
    suffix = "." + filename.split(".")[-1] if "." in filename else ".jpg"

    download_url = f"{base_url}=d"

    response = requests.get(download_url, headers=_auth_headers(creds))
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        return tmp.name


def delete_session(creds, session_id: str) -> None:
    """Deletes the session. Best practice to avoid hitting session limits."""
    requests.delete(
        f"{PICKER_BASE_URL}/sessions/{session_id}",
        headers=_auth_headers(creds),
    )