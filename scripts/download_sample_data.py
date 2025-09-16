#!/usr/bin/env python
"""Download sample data from Zenodo for omero-screen-plots examples.

This script downloads a sample CSV file from Zenodo and places it in the
omero-screen-plots examples data directory for testing and demonstration purposes.

Usage:
    python scripts/download_sample_data.py [--force]
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


def download_sample_data(force: bool = False) -> int:
    """Download sample plate data from Zenodo."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = (
        project_root / "packages" / "omero-screen-plots" / "examples" / "data"
    )

    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Data directory: {data_dir}")

    # Zenodo URL for the sample data
    zenodo_url = "https://zenodo.org/records/16636600/files/sample_plate_data.csv?download=1"

    # Target file path
    target_file = data_dir / "sample_plate_data.csv"

    # Check if file already exists
    if target_file.exists() and not force:
        file_size = target_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"✅ File already exists: {target_file.name} ({size_mb:.2f} MB)")
        print("   Use --force to re-download")
        return 0

    # Download the file
    print("🌐 Downloading from Zenodo...")
    print(f"   URL: {zenodo_url}")

    try:
        # Download with progress indication
        def download_progress(
            block_num: int, block_size: int, total_size: int
        ) -> None:
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            progress = int(50 * percent / 100)
            sys.stdout.write(
                f"\r   Progress: [{'=' * progress}{' ' * (50 - progress)}] {percent:.1f}%"
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(
            zenodo_url, target_file, reporthook=download_progress
        )
        print()  # New line after progress bar

        # Check file size
        file_size = target_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"✅ Successfully downloaded: {target_file.name}")
        print(f"   Size: {size_mb:.2f} MB")

        return 0

    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error {e.code}: {e.reason}")
        return 1
    except urllib.error.URLError as e:
        print(f"❌ URL Error: {e.reason}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download sample data from Zenodo"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Sample Data Downloader for omero-screen-plots")
    print("=" * 60)

    exit_code = download_sample_data(force=args.force)

    if exit_code == 0:
        print("\n📊 Sample data ready for use in examples!")
        print("   You can now run the example notebooks in:")
        print("   packages/omero-screen-plots/examples/")
    else:
        print(
            "\n⚠️  Download failed. Please check your internet connection and try again."
        )

    sys.exit(exit_code)
