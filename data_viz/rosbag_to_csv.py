#!/usr/bin/env python3
"""
Convert ROS2 bag directories to CSVs — no ROS installation required.

Install dependency:
    pip install rosbags

Usage:
    python3 rosbag_to_csv.py [--input DIR] [--output DIR] [--topics TOPIC [TOPIC ...]]

Defaults:
    --input   ./ros_bags      (scans for subdirs containing metadata.yaml)
    --output  ./csv_output
    --topics  (all topics in the bag)

Each topic is written to its own CSV file named:
    <bag_name>__<topic_sanitized>.csv
"""

import argparse
import csv
import os
import sys

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    sys.exit("Missing dependency. Run: pip install rosbags")


def _flatten(msg, prefix=""):
    """Recursively flatten a rosbags message object into {field_path: value}."""
    fields = {}
    for field in msg.__dataclass_fields__:
        val = getattr(msg, field)
        key = f"{prefix}.{field}" if prefix else field
        if hasattr(val, "__dataclass_fields__"):
            fields.update(_flatten(val, prefix=key))
        elif isinstance(val, (list, tuple, bytes)):
            if isinstance(val, (list, tuple)) and len(val) <= 16 and all(
                not hasattr(v, "__dataclass_fields__") for v in val
            ):
                for i, v in enumerate(val):
                    fields[f"{key}[{i}]"] = v
            else:
                fields[key] = str(val)
        else:
            fields[key] = val
    return fields


def bag_to_csv(bag_dir: str, output_dir: str, topics: list[str] | None = None):
    """Convert one ROS2 bag directory to a single CSV with all topics as columns."""
    os.makedirs(output_dir, exist_ok=True)
    bag_name = os.path.basename(bag_dir.rstrip("/"))
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with Reader(bag_dir) as reader:
        available = [conn.topic for conn in reader.connections]
        selected_topics = set(t for t in available if topics is None or t in topics)

        if not selected_topics:
            print(f"  [skip] No matching topics in {bag_dir}")
            return

        connections = [c for c in reader.connections if c.topic in selected_topics]

        # Collect all rows across all topics, prefixing each field with the topic name
        all_rows: list[dict] = []
        all_columns: dict[str, None] = {"timestamp": None, "topic": None}

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            prefix = conn.topic.lstrip("/").replace("/", ".")
            flat = _flatten(msg, prefix=prefix)
            row = {"timestamp": timestamp / 1e9, "topic": conn.topic}
            row.update(flat)
            all_rows.append(row)
            for col in flat:
                all_columns.setdefault(col, None)

        if not all_rows:
            return

        all_rows.sort(key=lambda r: r["timestamp"])
        headers = list(all_columns.keys())

        csv_path = os.path.join(output_dir, f"{bag_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  wrote {len(all_rows):>6} rows, {len(headers)} columns → {csv_path}")


def find_bag_dirs(input_dir: str) -> list[str]:
    """Find ROS2 bag directories (contain metadata.yaml) inside input_dir."""
    bags = []
    # Check if input_dir itself is a bag
    if os.path.isfile(os.path.join(input_dir, "metadata.yaml")):
        return [input_dir]
    # Otherwise scan one level deep
    for entry in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, entry)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "metadata.yaml")):
            bags.append(path)
    return bags


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Convert ROS2 bags to CSV files.")
    parser.add_argument(
        "--input",
        default=os.path.join(script_dir, "ros_bags"),
        help="Directory containing ROS2 bag folders (default: ./ros_bags)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "csv_output"),
        help="Directory to write CSV files into (default: ./csv_output)",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Topics to extract (default: all topics)",
    )
    args = parser.parse_args()

    bag_dirs = find_bag_dirs(args.input)

    if not bag_dirs:
        sys.exit(f"No ROS2 bag directories found in {args.input}")

    print(f"Found {len(bag_dirs)} bag(s) in {args.input}")
    print(f"Output directory: {args.output}\n")

    for bag_dir in bag_dirs:
        print(f"Processing: {os.path.basename(bag_dir)}")
        try:
            bag_to_csv(bag_dir, args.output, topics=args.topics)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
