#!/usr/bin/env python3
"""
GCS Bulk Inventory Patch for Snakemake Job Scheduler - FIXED VERSION

The key insight: We can't just patch update_input_sizes() because job_selector 
calls it via async_run(). We need to intercept BEFORE async_run executes it.

Solution: Patch async_run() to detect when update_input_sizes is being called,
do our bulk inventory FIRST, then let the original method run with cached data.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any
import inspect

# Import dependencies
from google.cloud import storage
from snakemake.io import _IOFile

# ============================================================================
# CONFIGURATION
# ============================================================================

GCS_PROJECT = os.environ.get('GCS_PROJECT', 'lasagna-199723')
GCS_BUCKET = os.environ.get('GCS_BUCKET', 'scale1')
GCS_PREFIX = os.environ.get('GCS_PREFIX', 'qinling/')

PATCH_DISABLED = os.environ.get('GCS_JOB_SCHEDULER_PATCH_DISABLE', '0') == '1'

if PATCH_DISABLED:
    print("⚠️  GCS Job Scheduler Patch is DISABLED", file=sys.stderr, flush=True)
    sys.exit(0)

# ============================================================================
# GLOBAL STATE
# ============================================================================

_bulk_cache: Dict[str, Dict[str, Any]] = {}
_cache_stats = {
    'inventory_calls': 0,
    'files_cached': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'inventory_time': 0.0
}

_original_iofile_exists = None
_original_iofile_size = None
_inventory_done = False

# ============================================================================
# BULK INVENTORY
# ============================================================================

def bulk_inventory_gcs() -> Dict[str, Dict[str, Any]]:
    """
    Perform ONE bulk GCS list_blobs() operation to cache all files.
    """
    print(f"\n{'='*70}", file=sys.stderr, flush=True)
    print(f"[GCS Bulk Inventory] Starting for gs://{GCS_BUCKET}/{GCS_PREFIX}*",
          file=sys.stderr, flush=True)

    start_time = datetime.now()
    cache = {}

    try:
        client = storage.Client(project=GCS_PROJECT)
        bucket = client.bucket(GCS_BUCKET)
        blobs = bucket.list_blobs(prefix=GCS_PREFIX)

        count = 0
        for blob in blobs:
            file_info = {
                'exists': True,
                'size': blob.size,
                'mtime': blob.updated.timestamp() if blob.updated else None
            }

            # Store under multiple path formats
            gcs_uri = f"gs://{GCS_BUCKET}/{blob.name}"
            snakemake_path = f".snakemake/storage/gcs/{GCS_BUCKET}/{blob.name}"
            local_path = f"/mnt/data/blainey/qinling-analysis/gcs/{GCS_BUCKET}/{blob.name}"

            cache[gcs_uri] = file_info
            cache[snakemake_path] = file_info
            cache[local_path] = file_info

            count += 1

            if count % 10000 == 0:
                print(f"[GCS Bulk Inventory] Processed {count:,} files...",
                      file=sys.stderr, flush=True)

        elapsed = (datetime.now() - start_time).total_seconds()

        _cache_stats['inventory_calls'] += 1
        _cache_stats['files_cached'] = count
        _cache_stats['inventory_time'] = elapsed

        print(f"[GCS Bulk Inventory] ✅ Cached {count:,} files in {elapsed:.2f}s",
              file=sys.stderr, flush=True)
        print(f"{'='*70}\n", file=sys.stderr, flush=True)

        return cache

    except Exception as e:
        print(f"[GCS Bulk Inventory] ❌ ERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr, flush=True)
        return {}

# ============================================================================
# PATCHED IOFILE METHODS
# ============================================================================

async def patched_iofile_exists(self):
    """Patched _IOFile.exists() that checks bulk cache first."""
    path_str = str(self)

    if path_str in _bulk_cache:
        _cache_stats['cache_hits'] += 1
        return _bulk_cache[path_str]['exists']

    _cache_stats['cache_misses'] += 1
    return await _original_iofile_exists(self)


async def patched_iofile_size(self):
    """Patched _IOFile.size() that checks bulk cache first."""
    path_str = str(self)

    if path_str in _bulk_cache:
        return _bulk_cache[path_str]['size']

    return await _original_iofile_size(self)

# ============================================================================
# THE KEY: INTERCEPT async_run()
# ============================================================================

def apply_patch():
    """Apply the patch by intercepting async_run()."""
    global _original_iofile_exists, _original_iofile_size, _inventory_done

    print("\n" + "="*70, file=sys.stderr, flush=True)
    print("🔧 Applying GCS Bulk Inventory Patch to Snakemake",
          file=sys.stderr, flush=True)
    print("="*70, file=sys.stderr, flush=True)

    # Import here to avoid circular dependencies
    from snakemake.common import async_run as original_async_run

    # Save original _IOFile methods
    _original_iofile_exists = _IOFile.exists
    _original_iofile_size = _IOFile.size

    # Apply _IOFile patches
    _IOFile.exists = patched_iofile_exists
    _IOFile.size = patched_iofile_size

    print(f"✓ Patched _IOFile.exists()", file=sys.stderr, flush=True)
    print(f"✓ Patched _IOFile.size()", file=sys.stderr, flush=True)

    # NOW THE CRITICAL PART: Patch async_run()
    def patched_async_run(coroutine):
        """
        Intercept async_run() to detect update_input_sizes() calls.
        Do bulk inventory BEFORE the coroutine runs.
        """
        global _bulk_cache, _inventory_done

        # Check if this coroutine is update_input_sizes
        # We can detect this by inspecting the coroutine
        coro_name = coroutine.__name__ if hasattr(coroutine, '__name__') else str(coroutine)
        
        print(f"\n[async_run] Called with: {coro_name}", file=sys.stderr, flush=True)
        
        # Check if this is the update_input_sizes call
        if 'update_input_sizes' in str(coroutine):
            print(f"[async_run] 🎯 Detected update_input_sizes call!", file=sys.stderr, flush=True)
            
            if not _inventory_done:
                print(f"[async_run] 📦 Doing bulk inventory BEFORE update_input_sizes...",
                      file=sys.stderr, flush=True)
                
                # Do the bulk inventory NOW
                _bulk_cache = bulk_inventory_gcs()
                _inventory_done = True
                
                print(f"[async_run] ✅ Bulk inventory complete, cache has {len(_bulk_cache)} entries",
                      file=sys.stderr, flush=True)
                print(f"[async_run] 🚀 Now running update_input_sizes with cached data...",
                      file=sys.stderr, flush=True)
            else:
                print(f"[async_run] ✓ Using existing inventory cache", file=sys.stderr, flush=True)

        # Run the original async_run with the coroutine
        return original_async_run(coroutine)

    # Replace async_run in the snakemake.common module
    import snakemake.common
    snakemake.common.async_run = patched_async_run

    print(f"✓ Patched async_run() to intercept update_input_sizes", file=sys.stderr, flush=True)
    
    # CRITICAL: Also patch job_selector to trigger inventory BEFORE it runs
    # This solves the case where the hang happens before update_input_sizes is called
    from snakemake.scheduling.job_scheduler import JobScheduler
    original_job_selector = JobScheduler.job_selector
    
    def early_inventory_job_selector(self, jobs):
        global _bulk_cache, _inventory_done
        
        print(f"\n{'='*70}", file=sys.stderr, flush=True)
        print(f"[job_selector] 🎯 ENTERED", file=sys.stderr, flush=True)
        print(f"[job_selector] Number of jobs: {len(list(jobs)) if hasattr(jobs, '__len__') else '?'}", 
              file=sys.stderr, flush=True)
        sys.stderr.flush()
        
        # DO BULK INVENTORY RIGHT NOW - before anything else
        if not _inventory_done:
            print(f"[job_selector] 📦 Doing bulk inventory FIRST...", file=sys.stderr, flush=True)
            sys.stderr.flush()
            
            _bulk_cache = bulk_inventory_gcs()
            _inventory_done = True
            
            print(f"[job_selector] ✅ Inventory complete: {len(_bulk_cache)} cache entries", 
                  file=sys.stderr, flush=True)
            sys.stderr.flush()
        else:
            print(f"[job_selector] ✓ Using existing cache: {len(_bulk_cache)} entries", 
                  file=sys.stderr, flush=True)
            sys.stderr.flush()
        
        print(f"[job_selector] Calling original job_selector with cached data...", 
              file=sys.stderr, flush=True)
        sys.stderr.flush()
        
        result = original_job_selector(self, jobs)
        
        print(f"[job_selector] ✅ Selected {len(result) if result else 0} jobs", 
              file=sys.stderr, flush=True)
        print(f"{'='*70}\n", file=sys.stderr, flush=True)
        sys.stderr.flush()
        
        return result
    
    JobScheduler.job_selector = early_inventory_job_selector
    print(f"✓ Patched JobScheduler.job_selector to do early inventory", file=sys.stderr, flush=True)

    print(f"\nConfiguration:", file=sys.stderr, flush=True)
    print(f"  GCS Project: {GCS_PROJECT}", file=sys.stderr, flush=True)
    print(f"  GCS Bucket:  {GCS_BUCKET}", file=sys.stderr, flush=True)
    print(f"  GCS Prefix:  {GCS_PREFIX}", file=sys.stderr, flush=True)
    print("="*70 + "\n", file=sys.stderr, flush=True)


def print_statistics():
    """Print cache statistics at exit."""
    if _cache_stats['inventory_calls'] == 0:
        return

    print("\n" + "="*70, file=sys.stderr, flush=True)
    print("📊 GCS Job Scheduler Patch Statistics", file=sys.stderr, flush=True)
    print("="*70, file=sys.stderr, flush=True)
    print(f"Inventory operations:  {_cache_stats['inventory_calls']}",
          file=sys.stderr, flush=True)
    print(f"Files cached:          {_cache_stats['files_cached']:,}",
          file=sys.stderr, flush=True)
    print(f"Inventory time:        {_cache_stats['inventory_time']:.2f}s",
          file=sys.stderr, flush=True)
    print(f"Cache hits:            {_cache_stats['cache_hits']:,}",
          file=sys.stderr, flush=True)
    print(f"Cache misses:          {_cache_stats['cache_misses']:,}",
          file=sys.stderr, flush=True)

    total_checks = _cache_stats['cache_hits'] + _cache_stats['cache_misses']
    if total_checks > 0:
        hit_rate = (_cache_stats['cache_hits'] / total_checks) * 100
        print(f"Cache hit rate:        {hit_rate:.1f}%", file=sys.stderr, flush=True)

        if hit_rate > 95:
            print(f"\n✅ Excellent cache performance!", file=sys.stderr, flush=True)
            print(f"   Avoided ~{_cache_stats['cache_hits']:,} GCS API calls",
                  file=sys.stderr, flush=True)

    print("="*70 + "\n", file=sys.stderr, flush=True)


# ============================================================================
# AUTO-APPLY ON IMPORT
# ============================================================================

import atexit

apply_patch()
atexit.register(print_statistics)

__all__ = ['apply_patch', 'print_statistics', '_cache_stats']