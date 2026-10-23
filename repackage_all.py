"""
========================================================================================
DLKR-LJP V5.1 RE-PACKAGING AND SUBMISSION BUNDLER
========================================================================================
Synchronizes and regenerates all distribution zip archives:
1. DLKR_LJP_V5_Presentation_Package.zip
2. DLKR_LJP_V5_Submission_Clean.zip
3. ../final.zip
========================================================================================
"""

import os
import sys
import zipfile
from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent
PARENT_DIR = FINAL_DIR.parent

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".DS_Store",
    "*.pyc",
    "*.zip"  # Do not include existing zip files inside zip files
]

def should_exclude(file_path: Path) -> bool:
    parts = file_path.parts
    for p in parts:
        if p in ["__pycache__", ".git", ".pytest_cache"]:
            return True
    if file_path.suffix in [".pyc", ".zip"]:
        return True
    return False

def make_zip(source_dir: Path, output_zip_path: Path, prefix_in_zip: str = ""):
    print(f"Creating archive: {output_zip_path.name}...")
    file_count = 0
    total_bytes = 0
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", ".pytest_cache", "runtime_data"]]
            for f in files:
                f_path = Path(root) / f
                if should_exclude(f_path):
                    continue
                # Calculate relative path
                rel_path = f_path.relative_to(source_dir)
                archive_name = f"{prefix_in_zip}/{rel_path}".lstrip("/")
                zf.write(f_path, archive_name)
                file_count += 1
                total_bytes += f_path.stat().st_size
    print(f"  -> Added {file_count} files ({total_bytes / (1024*1024):.2f} MB uncompressed). Output size: {output_zip_path.stat().st_size / 1024:.1f} KB")

def main():
    print("=" * 65)
    print("DLKR-LJP V5.1 BUNDLE SYNCHRONIZATION")
    print("=" * 65)

    # 1. DLKR_LJP_V5_Presentation_Package.zip
    pkg_zip = FINAL_DIR / "DLKR_LJP_V5_Presentation_Package.zip"
    make_zip(FINAL_DIR, pkg_zip)

    # 2. DLKR_LJP_V5_Submission_Clean.zip
    clean_zip = FINAL_DIR / "DLKR_LJP_V5_Submission_Clean.zip"
    make_zip(FINAL_DIR, clean_zip)

    # 3. Parent final.zip (d:\mh\final.zip)
    root_zip = PARENT_DIR / "final.zip"
    make_zip(FINAL_DIR, root_zip, prefix_in_zip="final")

    print("=" * 65)
    print("ALL 3 DISTRIBUTION ARCHIVES REPACKAGED & SYNCHRONIZED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
