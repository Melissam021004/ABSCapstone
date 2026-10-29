"""
abs_zone.py — the 2026 ABS strike zone, exactly as Week 5 (script 08) builds it.

Week 9's predict(state) has to turn a raw pitch location into the same
features the model was trained on, so the geometry lives here once.
tests/test_predict_interface.py checks that these functions reproduce the
`zone_dist_in` column of the Week 5 table to the last decimal.

    width    17 inches, centred on the plate
    top      53.5% of the batter's height
    bottom   27%   of the batter's height
    strike   if ANY part of the ball touches the zone (so the ball's
             1.45-inch radius is subtracted from the distance)
"""

import numpy as np

PLATE_HALF_WIDTH_FT = 8.5 / 12
BALL_RADIUS_FT = (2.9 / 2) / 12
ZONE_TOP_PCT = 0.535
ZONE_BOT_PCT = 0.27


def zone_from_height(height_ft):
    """(top, bottom) of the ABS zone in feet for a batter of this height."""
    h = np.asarray(height_ft, float)
    return h * ZONE_TOP_PCT, h * ZONE_BOT_PCT


def zone_distance_ft(plate_x, plate_z, top, bot):
    """Signed distance from the ball's surface to the ABS rectangle, in feet.
    Negative => some part of the ball overlaps the zone => ABS strike.
    (Identical to scripts/08_build_challenge_inventory.py.)"""
    plate_x = np.asarray(plate_x, float)
    plate_z = np.asarray(plate_z, float)
    top = np.asarray(top, float)
    bot = np.asarray(bot, float)

    dx = np.abs(plate_x) - PLATE_HALF_WIDTH_FT
    dz = np.maximum(bot - plate_z, plate_z - top)

    inside = (dx <= 0) & (dz <= 0)
    corner = (dx > 0) & (dz > 0)
    d = np.where(inside, np.maximum(dx, dz),
                 np.where(corner,
                          np.hypot(np.maximum(dx, 0), np.maximum(dz, 0)),
                          np.maximum(dx, dz)))
    return d - BALL_RADIUS_FT


def zone_distance_in(plate_x, plate_z, top, bot):
    return zone_distance_ft(plate_x, plate_z, top, bot) * 12
