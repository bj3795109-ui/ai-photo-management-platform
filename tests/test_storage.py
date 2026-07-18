import os
import shutil
from storage.local_backend import LocalStorageBackend

def test_local_backend_upload_and_get_url(tmp_path):
    backend = LocalStorageBackend(base_dir=str(tmp_path))
    
    src_file = tmp_path / "source.txt"
    src_file.write_text("test content")
    
    ref = backend.upload(str(src_file), "dest.txt")
    assert os.path.exists(ref)
    assert backend.get_url(ref) == ref
    
def test_local_backend_delete(tmp_path):
    backend = LocalStorageBackend(base_dir=str(tmp_path))
    
    src_file = tmp_path / "source.txt"
    src_file.write_text("test content")
    
    ref = backend.upload(str(src_file), "dest.txt")
    assert os.path.exists(ref)
    
    backend.delete(ref)
    assert not os.path.exists(ref)