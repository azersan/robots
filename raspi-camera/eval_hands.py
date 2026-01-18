#!/usr/bin/env python3
"""
Evaluation runner for hand gesture detection.
Loads test cases and reports accuracy metrics.
Tracks results over time in eval_history.json.

Usage:
    python3 eval_hands.py              # Run all tests
    python3 eval_hands.py --verbose    # Show all results, not just failures
    python3 eval_hands.py --history    # Show evaluation history
    python3 eval_hands.py --no-save    # Run without saving to history
"""

import os
import json
import argparse
import subprocess
from datetime import datetime
from gesture_hands import detect_hand_gesture, dict_to_landmarks

HISTORY_FILE = "eval_history.json"


def get_git_info():
    """Get current git commit hash and message."""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        commit_msg = subprocess.check_output(
            ['git', 'log', '-1', '--format=%s'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit_hash, commit_msg
    except Exception:
        return None, None


def load_history(script_dir):
    """Load evaluation history from file."""
    history_path = os.path.join(script_dir, HISTORY_FILE)
    if os.path.exists(history_path):
        with open(history_path) as f:
            return json.load(f)
    return []


def save_history(script_dir, history):
    """Save evaluation history to file."""
    history_path = os.path.join(script_dir, HISTORY_FILE)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def show_history(history, limit=10):
    """Display evaluation history."""
    if not history:
        print("No evaluation history yet.")
        return

    print(f"\n{'='*60}")
    print("Evaluation History (recent first)")
    print(f"{'='*60}\n")

    for entry in reversed(history[-limit:]):
        timestamp = entry.get('timestamp', 'unknown')
        accuracy = entry.get('accuracy', 0)
        total = entry.get('total', 0)
        correct = entry.get('correct', 0)
        commit = entry.get('commit_hash', 'unknown')
        commit_msg = entry.get('commit_msg', '')[:40]

        print(f"{timestamp}  {accuracy:5.1%} ({correct}/{total})  [{commit}] {commit_msg}")

        # Show per-gesture if available
        if 'per_gesture' in entry:
            for gesture, stats in sorted(entry['per_gesture'].items()):
                g_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"    {gesture:15} {stats['correct']}/{stats['total']} ({g_acc:.0%})")
        print()


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

    # Return detailed results for history tracking
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct_count,
        'per_gesture': gesture_stats,
        'failures': [{'id': r['id'], 'expected': r['expected'], 'predicted': r['predicted']}
                     for r in failures]
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate hand gesture detection")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all results, not just failures')
    parser.add_argument('--history', action='store_true',
                       help='Show evaluation history')
    parser.add_argument('--no-save', action='store_true',
                       help='Run without saving to history')
    args = parser.parse_args()

    # Find directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "test_data", "hands")

    # Show history if requested
    if args.history:
        history = load_history(script_dir)
        show_history(history)
        return

    print(f"Loading test cases from: {test_dir}")
    cases = load_test_cases(test_dir)
    print(f"Found {len(cases)} test cases")

    results = run_eval(cases, verbose=args.verbose)

    if results and not args.no_save:
        # Get git info
        commit_hash, commit_msg = get_git_info()

        # Create history entry
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'accuracy': results['accuracy'],
            'total': results['total'],
            'correct': results['correct'],
            'per_gesture': results['per_gesture'],
            'commit_hash': commit_hash,
            'commit_msg': commit_msg,
        }

        # Load existing history and append
        history = load_history(script_dir)
        history.append(entry)
        save_history(script_dir, history)
        print(f"Results saved to {HISTORY_FILE}")


if __name__ == '__main__':
    main()
