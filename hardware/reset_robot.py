#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time

import YanAPI


def call_if_exists(name, *args):
    func = getattr(YanAPI, name, None)
    if func is None:
        print("[RESET] {}: not available".format(name))
        return None
    try:
        resp = func(*args)
        print("[RESET] {}{} -> {}".format(name, args, resp))
        return resp
    except Exception as e:
        print("[RESET] {}{} failed: {}".format(name, args, e))
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.31.239")
    ap.add_argument("--servo", default="NeckLR")
    ap.add_argument("--center", type=int, default=90)
    ap.add_argument("--duration", type=int, default=1000)
    ap.add_argument("--shutdown", action="store_true",
                    help="Try robot shutdown/poweroff API if available")
    args = ap.parse_args()

    init_resp = YanAPI.yan_api_init(args.ip)
    print("[RESET] yan_api_init({}) -> {}".format(args.ip, init_resp))

    call_if_exists("stop_play_motion")
    call_if_exists("stop_subscribe_motion")
    call_if_exists("stop_subscribe_motion_gait")
    call_if_exists("exit_motion_gait")

    print("[RESET] Centering {} to {} deg".format(args.servo, args.center))
    resp = YanAPI.set_servos_angles({args.servo: args.center}, args.duration)
    print("[RESET] set_servos_angles -> {}".format(resp))
    time.sleep(max(0.5, args.duration / 1000.0 + 0.2))

    call_if_exists("get_servos_angles", [args.servo])
    call_if_exists("get_servos_mode", [args.servo])

    if args.shutdown:
        # Different YanAPI builds expose different names; try conservative common ones.
        for name in ("shutdown", "power_off", "poweroff", "robot_shutdown"):
            if getattr(YanAPI, name, None) is not None:
                call_if_exists(name)
                break
        else:
            print("[RESET] No known shutdown API found. Use OS command if you need Pi shutdown.")

    print("[RESET] Done.")


if __name__ == "__main__":
    main()
