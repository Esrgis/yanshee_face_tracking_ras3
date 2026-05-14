#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time

import YanAPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--servo", default="NeckLR")
    ap.add_argument("--duration", type=int, default=1200)
    ap.add_argument("--angles", default="90,45,135,90")
    ap.add_argument("--list-api", action="store_true")
    args = ap.parse_args()

    if args.list_api:
        names = [name for name in dir(YanAPI) if "servo" in name.lower() or "motion" in name.lower()]
        print("[YanAPI] servo/motion functions:")
        for name in sorted(names):
            print("  " + name)

    init_resp = YanAPI.yan_api_init(args.ip)
    print("[TEST] yan_api_init({}) -> {}".format(args.ip, init_resp))

    prev_angle = None
    for raw in args.angles.split(","):
        angle = int(raw.strip())
        if prev_angle is None:
            direction = "center/start"
        elif angle < prev_angle:
            direction = "angle decreasing"
        elif angle > prev_angle:
            direction = "angle increasing"
        else:
            direction = "same angle"
        print("[TEST] set {}={} duration={} ({})".format(
            args.servo, angle, args.duration, direction))
        try:
            resp = YanAPI.set_servos_angles({args.servo: angle}, args.duration)
            print("[TEST] response -> {}".format(resp))
        except Exception as e:
            print("[TEST] exception -> {}".format(e))
        time.sleep(max(0.3, args.duration / 1000.0 + 0.2))
        prev_angle = angle


if __name__ == "__main__":
    main()
