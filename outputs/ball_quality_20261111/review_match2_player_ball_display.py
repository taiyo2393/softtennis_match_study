"""Apply visual identity and contact reviews to the match2 pilot display."""
import build_match2_player_ball_review as pilot
import build_044_spatiotemporal_pilot as common


DEST = common.HERE / 'match2_first_three_player_ball_v2'
# Stable labels use point 1 as reference: N1 burgundy, N2 red and black,
# F1 red shoes, F2 dark teal shoes. The detector's slots are retained separately.
NEAR_SWAP = {'near_front': 'near_rear', 'near_rear': 'near_front'}
FAR_SWAP = {'far_front': 'far_rear', 'far_rear': 'far_front'}
IDENTITY = {1: {}, 2: NEAR_SWAP, 3: {**NEAR_SWAP, **FAR_SWAP}}
# Visual estimates from native frame sequences. Hit coordinates use the hitter's
# box-bottom proxy. Bounce image coordinates use the ball's ground-contact area.
ANCHORS = [
    (1, 'P1_serve', 12327, 'far_rear', 12411, 1212, 666),
    (1, 'P1_final', 12469, 'far_front', 12571, 1144, 838),
    (2, 'P2_serve', 13657, 'far_rear', 13712, 730, 652),
    (3, 'P3_late', 15461, 'far_front', 15477, 1014, 691),
]
OCCLUSION = {(f, 'far_front') for f in range(14023, 14044)}


def main():
    DEST.mkdir(exist_ok=True)
    original = pilot.OUT / 'player_court_proxies.csv'
    protocol = common.load(pilot.OUT / 'protocol.json')
    calibration = common.load(pilot.PILOT / 'protocol.json')
    homography = calibration['image_to_court_homography_canonical']
    positions = common.read(original)
    by_frame = {(int(r['source_frame']), r['player_slot']): r for r in positions}
    routes = {point: [] for point in pilot.SCOPES}
    for point, name, first, slot, bounce, px, py in ANCHORS:
        player = by_frame[first, slot]
        assert int(player['point']) == point and (first, slot) not in OCCLUSION
        u, v = common.project(homography, px * 2, py * 2)
        routes[point].append(dict(
            id=name, point=point, first_frame=first, bounce_frame=bounce,
            hitter_display_id=common.NAMES[IDENTITY[point].get(slot, slot)],
            start_x=float(player['court_proxy_x']), start_y=float(player['court_proxy_y']),
            end_x=u, end_y=v, bounce_image_x=px, bounce_image_y=py,
            start_evidence='visual hitter candidate and projected box-bottom proxy',
            end_evidence='visual bounce candidate and image ground-plane projection',
            route_type='straight schematic ground-plane connector between approximate endpoints'))

    stable = []
    for row in positions:
        point, f, slot = int(row['point']), int(row['source_frame']), row['player_slot']
        masked = (f, slot) in OCCLUSION
        stable.append(dict(row, display_id=common.NAMES[IDENTITY[point].get(slot, slot)],
                           court_position_display='hidden_feet_occluded' if masked else 'proxy_shown',
                           shown_court_x='' if masked else row['court_proxy_x'],
                           shown_court_y='' if masked else row['court_proxy_y']))
    common.write(DEST / 'reviewed_player_display.csv', stable)
    common.write(DEST / 'schematic_ball_routes.csv', [r for group in routes.values() for r in group])
    common.dump(DEST / 'review_protocol.json', {
        'source_tracking_sha256': common.sha(original),
        'source_ball_path_sha256': protocol['ball_path_sha256'],
        'point_1_id_cues': {'N1': 'burgundy', 'N2': 'red and black',
                            'F1': 'red shoes', 'F2': 'dark teal shoes'},
        'changed_slot_display_ids': IDENTITY,
        'identity_evidence': 'visual spot review across all three points; no independent ID labels',
        'point_2_F2_foot_occlusion_review': [14023, 14043],
        'route_evidence': 'four visually selected shots with approximate bounce anchors',
        'source_core_denominators_unchanged': True,
        'point_1_extra_frame': 'F12571 shown with an approximately 0.8 second hold for the bounce',
    })
    pilot.render(dict(destination=DEST, routes=routes, stable_names=IDENTITY, masked_positions=OCCLUSION))


if __name__ == '__main__':
    main()
