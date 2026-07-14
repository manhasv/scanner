from pathlib import Path
import shutil

OUTPUT = "../output/"
def clear_folder(folder=OUTPUT):
    folder = Path(folder)

    if not folder.exists():
        folder.mkdir(parents=True)
        return

    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

clear_folder("output")