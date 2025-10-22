import argparse
import json
from pymot.pymot import MOTEvaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--groundtruth', required=True)
    parser.add_argument('-b', '--hypothesis', required=True)
    parser.add_argument('-c', '--check_format', action="store_true", default=True)
    parser.add_argument('-v', '--visual_debug_file')
    parser.add_argument('-i', '--iou', default=0.2, type=float, help='iou threshold')
    args = parser.parse_args()

    # Load ground truth according to format
    # Assume MOT format, if non-json
    gt = open(args.groundtruth) # gt file
    if args.groundtruth.endswith(".json"):
        groundtruth = json.load(gt)[0]
    else:
        groundtruth = MOT_groundtruth_import(gt.readlines())
    gt.close()

    # Load MOT format files
    hypo = open(args.hypothesis) # hypo file
    if args.hypothesis.endswith(".json"):
        hypotheses = json.load(hypo)[0]
    else:
        hypotheses = MOT_hypo_import(hypo.readlines())
    hypo.close()

    evaluator = MOTEvaluation(groundtruth, hypotheses, args.iou)

    if args.check_format:
        formatChecker = FormatChecker(groundtruth, hypotheses)
        success = formatChecker.checkForExistingIDs()
        success |= formatChecker.checkForAmbiguousIDs()
        success |= formatChecker.checkForCompleteness()

        if not success:
            write_stderr_red("Error:", "Stopping. Fix ids first. Evaluating with broken data does not make sense!\n    File: %s" % args.groundtruth)
            sys.exit()

    evaluator.evaluate()
    print("Track statistics")
    evaluator.printTrackStatistics()
    print()
    print("Results")
    evaluator.printResults()
    print("Absolute Statistics")
    abs_stats = evaluator.getAbsoluteStatistics()
    for key, value in abs_stats.items():
        print(f"{key}: {value}")
    print("Relative Statistics")
    rel_stats = evaluator.getRelativeStatistics()
    for key, value in rel_stats.items():
        print(f"{key}: {value}")

    if args.visual_debug_file:
        with open(args.visual_debug_file, 'w') as fp:
            json.dump(evaluator.getVisualDebug(), fp, indent=4)
