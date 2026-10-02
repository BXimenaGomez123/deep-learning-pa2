from pa2.tracking_analysis import analyze_identity_errors


def test_identity_errors_count_switches_and_bounded_miss_runs():
    box = (20, 20, 20, 20)
    results = analyze_identity_errors(
        gt_boxes=[[box], [box], [box]],
        gt_ids=[[7], [7], [7]],
        pred_boxes=[[box], [], [box]],
        pred_ids=[[3], [], [8]],
        frame_numbers=[1, 2, 3],
        image_sizes=[(100, 100)] * 3,
    )

    identity = results[0]
    assert identity.id_switches == 1
    assert identity.fragmentations == 1
    assert identity.misses == 1
    assert identity.error_score == 2
    assert identity.failure_frames == [3]


def test_gt_annotation_gap_is_not_counted_as_tracker_fragmentation():
    box = (20, 20, 20, 20)
    results = analyze_identity_errors(
        gt_boxes=[[box], [], [box]],
        gt_ids=[[7], [], [7]],
        pred_boxes=[[box], [], [box]],
        pred_ids=[[3], [], [3]],
        frame_numbers=[1, 20, 21],
        image_sizes=[(100, 100)] * 3,
    )

    identity = results[0]
    assert identity.id_switches == 0
    assert identity.fragmentations == 0
    assert identity.longest_annotation_gap == 19


def test_overlapping_people_at_an_identity_failure_is_flagged():
    target = (20, 20, 20, 20)
    overlapping = (30, 20, 20, 20)
    results = analyze_identity_errors(
        gt_boxes=[[target, overlapping], [target, overlapping]],
        gt_ids=[[7, 9], [7, 9]],
        pred_boxes=[[target, overlapping], [target, overlapping]],
        pred_ids=[[3, 4], [5, 4]],
        frame_numbers=[1, 2],
        image_sizes=[(100, 100)] * 2,
    )

    target_result = next(identity for identity in results if identity.identity_id == 7)
    assert target_result.id_switches == 1
    assert target_result.crossing_events == 1
