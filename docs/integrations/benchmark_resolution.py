#!/usr/bin/env python
"""
Benchmark script to test LeNet5 segmentation at different resolutions
and compare with HEST baseline.
"""

import time
import os
import subprocess
import sys

def run_segmentation_test(segmenter, target_mag, output_suffix):
    """Run segmentation test and measure time."""

    slide_path = "/mnt/hd12tb/experiments/archive/all_slides_laaff/fibpen/E2021000606D1-a-1_HE_20210527_110251.tiff"
    job_dir = f"/tmp/benchmark_{segmenter}_{output_suffix}"

    # Ensure we're in the right directory and construct the command
    cmd = [
        "bash", "-c",
        f"cd /home/tobechanged/mirrored_folder/benchmark2025/TRIDENT && "
        f"conda run -n trident python run_single_slide.py "
        f"--slide_path {slide_path} "
        f"--job_dir {job_dir} "
        f"--segmenter {segmenter} "
        f"--seg_conf_thresh 0.5 "
        f"--mpp 0.25 "
        f"--mag 10 "
        f"--patch_size 256"
    ]

    print(f"\n🔄 Testing {segmenter} at target_mag={target_mag} (resolution ~{10/target_mag:.2f} µm/pixel)")
    print(f"   Output: {job_dir}")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print(f"✅ Success in {duration:.1f}s")
            return duration, True, ""
        else:
            print(f"❌ Failed in {duration:.1f}s")
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return duration, False, error_msg

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after 600s")
        return 600, False, "Timeout"
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"💥 Exception in {duration:.1f}s: {e}")
        return duration, False, str(e)

def update_lenet5_target_mag(target_mag):
    """Update the target_mag in LeNet5Segmenter."""

    load_py_path = "/home/tobechanged/mirrored_folder/benchmark2025/TRIDENT/trident/segmentation_models/load.py"

    # Read the file
    with open(load_py_path, 'r') as f:
        content = f.read()

    # Replace the target_mag line
    import re
    pattern = r'(self\.target_mag = )[0-9.]+(\s*# .*)?'
    replacement = f'\\g<1>{target_mag}  # Benchmark test resolution'

    new_content = re.sub(pattern, replacement, content)

    # Write back
    with open(load_py_path, 'w') as f:
        f.write(new_content)

    print(f"📝 Updated LeNet5 target_mag to {target_mag}")

def main():
    """Run the resolution vs performance benchmark."""

    print("🚀 Starting LeNet5 Resolution vs Performance Benchmark")
    print("=" * 60)

    # Test configurations
    target_mags = [2.5, 5, 10, 15, 20]  # From low to high resolution

    results = []

    # First, test HEST as baseline (once)
    print("\n📊 BASELINE: Testing HEST segmentation")
    duration, success, error = run_segmentation_test("hest", 10, "baseline")
    results.append(("HEST (baseline)", 10, 10/10, duration, success, error))

    # Test LeNet5 at different resolutions
    print("\n📊 TESTING: LeNet5 at different resolutions")
    for target_mag in target_mags:
        # Update LeNet5 target_mag
        update_lenet5_target_mag(target_mag)

        # Run test
        duration, success, error = run_segmentation_test("lenet5", target_mag, f"mag{target_mag}")
        resolution_um = 10 / target_mag
        results.append(("LeNet5", target_mag, resolution_um, duration, success, error))

        # Short break between tests
        time.sleep(2)

    # Print summary
    print("\n" + "=" * 80)
    print("📈 BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Target Mag':<12} {'Resolution':<12} {'Time (s)':<10} {'Status':<8} {'Error'}")
    print("-" * 80)

    for model, mag, res_um, duration, success, error in results:
        status = "✅ OK" if success else "❌ FAIL"
        error_short = error[:30] + "..." if len(error) > 30 else error
        print(f"{model:<15} {mag:<12} {res_um:.2f} µm/px  {duration:<10.1f} {status:<8} {error_short}")

    # Analysis
    successful_lenet5 = [(mag, res, dur) for model, mag, res, dur, success, _ in results
                         if model == "LeNet5" and success]

    if successful_lenet5:
        print(f"\n🔍 ANALYSIS:")
        print(f"   - HEST baseline: {results[0][3]:.1f}s at {results[0][2]:.2f} µm/pixel")

        fastest_lenet5 = min(successful_lenet5, key=lambda x: x[2])
        slowest_lenet5 = max(successful_lenet5, key=lambda x: x[2])

        print(f"   - LeNet5 fastest: {fastest_lenet5[2]:.1f}s at {fastest_lenet5[1]:.2f} µm/pixel (mag={fastest_lenet5[0]})")
        print(f"   - LeNet5 slowest: {slowest_lenet5[2]:.1f}s at {slowest_lenet5[1]:.2f} µm/pixel (mag={slowest_lenet5[0]})")

        # Speed comparison
        hest_time = results[0][3]
        for mag, res, dur in successful_lenet5:
            speedup = hest_time / dur if dur > 0 else float('inf')
            print(f"   - LeNet5 mag={mag}: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than HEST")

if __name__ == "__main__":
    main()
