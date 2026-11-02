"""Rosbridge websocket smoke test: subscribes to /dgt/status and checks each message is DGT_DS JSON.

Usage:
    python scripts/test_rosbridge_client.py --host <jetson-ip> [--port 9090] [--seconds 10]

Requires: pip install websockets
"""
import argparse
import asyncio
import json
import time

MSG_TYPE = "std_msgs/String"


async def main(host: str, port: int, seconds: float, topic: str):
    import websockets

    url = f"ws://{host}:{port}"
    count = 0
    bad = 0
    last = None

    async with websockets.connect(url) as ws:
        print(f"Connected to {url}")
        await ws.send(json.dumps({"op": "subscribe", "topic": topic, "type": MSG_TYPE}))
        print(f"Subscribed to {topic}")

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                break
            data = json.loads(raw)
            if data.get("op") != "publish" or data.get("topic") != topic:
                continue
            count += 1
            try:
                message = json.loads(data["msg"]["data"])
                last = message["metadata"]["sequence_number"]
            except (KeyError, TypeError, ValueError):
                bad += 1

    print(f"\nResults over {seconds:.0f}s:")
    print(f"  {topic}: {count} msgs ({count / seconds:.1f} Hz), "
          f"{bad} not DGT_DS, last sequence_number: {last}")

    if count == 0:
        print("\nNo messages received. Check: dgt active? rosbridge active? "
              "ROS_DOMAIN_ID / ROS_LOCALHOST_ONLY the same in both units? "
              "QoS mismatch (try ROS_QOS_BEST_EFFORT=false)?")
        raise SystemExit(1)
    if bad:
        raise SystemExit(1)
    print("\nROSBRIDGE TEST PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Jetson IP or hostname (no default — deployment specific)")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--topic", default="/dgt/status", help="{ROS_TOPIC_PREFIX}/status")
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port, args.seconds, args.topic))
