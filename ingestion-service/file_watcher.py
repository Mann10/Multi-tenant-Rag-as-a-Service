import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .database import add_detected_file

DATA_DIR = os.getenv("DATA_DIR", "/data")


class TenantFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        path = event.src_path
        rel_path = os.path.relpath(path, DATA_DIR)
        parts = rel_path.split(os.sep)

        if len(parts) >= 2 and parts[1].lower() == "raw":
            tenant_id = parts[0]
            filename = os.path.basename(path)
            add_detected_file(tenant_id, filename, path)
            print(f"[Watcher] Detected {rel_path} for tenant {tenant_id}")


def start_watcher():
    os.makedirs(DATA_DIR, exist_ok=True)
    observer = Observer()
    observer.schedule(TenantFileHandler(), DATA_DIR, recursive=True)
    observer.start()
    print(f"[Watcher] Watching {DATA_DIR}")
    return observer