#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import concurrent.futures
import threading

# Create a lock to prevent output interleaving
print_lock = threading.Lock()

def sync_single_folder(path, index, total):
    """
    Syncs a single wandb path
    """
    with print_lock:
        print(f"[{index}/{total}] Starting sync for {path}...")

    if not os.path.exists(path):
        with print_lock:
            print(f"Error: path {path} does not exist")
        return path, False, "Path does not exist"

    try:
        result = subprocess.run(["wandb", "sync", path], check=True)

        return path, True, None
    except subprocess.CalledProcessError as e:
        with print_lock:
            print(f"Error syncing {path}: {e}")
        return path, False, str(e)

def sync_wandb_folders(paths, max_workers=None):
    """
    Runs wandb sync for each of the specified paths in parallel
    """
    successful = []
    failed = []

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create list of tasks
        future_to_path = {
            executor.submit(sync_single_folder, path, i+1, len(paths)): path
            for i, path in enumerate(paths)
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_path):
            path, success, error = future.result()
            if success:
                successful.append(path)
            else:
                failed.append((path, error))

    # Print summary
    print("\n--- Sync Summary ---")
    print(f"Successfully synced: {len(successful)}/{len(paths)}")

    if failed:
        print(f"Failed to sync: {len(failed)}/{len(paths)}")
        print("List of failed syncs:")
        for path, error in failed:
            print(f"- {path}: {error}")

    return successful, failed

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Parallel sync of offline wandb runs"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to offline wandb runs (multiple paths can be specified)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Maximum number of parallel processes (default: number of CPUs)"
    )

    args = parser.parse_args()

    # Run sync
    successful, failed = sync_wandb_folders(args.paths, max_workers=args.workers)

    # Exit with non-zero code if there were errors
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
