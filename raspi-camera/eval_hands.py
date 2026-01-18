#!/usr/bin/env python3
"""
Evaluation runner for hand gesture detection.
Loads test cases and reports accuracy metrics.

Usage:
    python3 eval_hands.py              # Run all tests
    python3 eval_hands.py --verbose    # Show all results, not just failures
"""

import os
import json
import argparse
from gesture_hands import detect_hand_gesture, dict_to_landmarks


def load_test_cases(test_dir):
    """Load all test cases from the test data directory."""
    cases = []

    if not os.path.exists(test_dir):
        return cases

    for case_name in sorted(os.listdir(test_dir)):
        case_path = os.path.join(test_dir, case_name, "case.json")
        if os.path.exists(case_path):
            with open(case_path) as f:
                case = json.load(f)
                case['_dir'] = case_name
                cases.append(case)

    return cases


def run_eval(cases, verbose=False):
    """Run evaluation on all test cases."""
    if not cases:
        print("No test cases found!")
        print("Capture some test cases first:")
        print("  python3 local_hands.py --local --capture")
        return

    results = []

    for case in cases:
        # Convert landmarks from dict to Landmark objects
        landmarks = dict_to_landmarks(case['landmarks'])

        # Run detection
        predicted, _ = detect_hand_gesture(landmarks, case['handedness'])
        expected = case['expected_gesture']
        correct = (predicted == expected)

        results.append({
            'id': case['id'],
            'expected': expected,
            'predicted': predicted,
            'correct': correct,
            'handedness': case['handedness'],
        })

    # Calculate summary stats
    total = len(results)
    correct_count = sum(r['correct'] for r in results)
    accuracy = correct_count / total if total > 0 else 0

    # Print results
    print(f"\n{'='*50}")
    print(f"Hand Gesture Evaluation Results")
    print(f"{'='*50}")
    print(f"\nTest cases: {total}")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{total})")
    print()

    # Group by gesture type
    gesture_stats = {}
    for r in results:
        g = r['expected']
        if g not in gesture_stats:
            gesture_stats[g] = {'total': 0, 'correct': 0}
        gesture_stats[g]['total'] += 1
        if r['correct']:
            gesture_stats[g]['correct'] += 1

    print("Per-gesture accuracy:")
    for gesture, stats in sorted(gesture_stats.items()):
        g_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        status = "OK" if g_acc == 1.0 else "FAIL" if g_acc == 0 else "PARTIAL"
        print(f"  {gesture:15} {stats['correct']:2}/{stats['total']:<2} ({g_acc:5.1%}) {status}")
    print()

    # Print failures (or all if verbose)
    failures = [r for r in results if not r['correct']]

    if verbose:
        print("All results:")
        for r in results:
            status = "PASS" if r['correct'] else "FAIL"
            print(f"  [{status}] {r['id']}: expected '{r['expected']}', got '{r['predicted']}'")
    elif failures:
        print(f"Failures ({len(failures)}):")
        for r in failures:
            print(f"  {r['id']}: expected '{r['expected']}', got '{r['predicted']}'")
    else:
        print("All tests passed!")

    print()
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate hand gesture detection")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all results, not just failures')
    args = parser.parse_args()

    # Find test data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "test_data", "hands")

    print(f"Loading test cases from: {test_dir}")
    cases = load_test_cases(test_dir)
    print(f"Found {len(cases)} test cases")

    run_eval(cases, verbose=args.verbose)


if __name__ == '__main__':
    main()
