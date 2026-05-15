import argparse
import duckdb
import json
from loguru import logger

FILE_PATH = "local-data/ED-2025-OPE-0944_2026-03-30.json.gz"


def read_json(input_path: str) -> dict:
    """Load JSON into memory."""
    try:
        with open(input_path) as f:
            data = json.load(f)
    except Exception:
        logger.exception("Could not load data from {}", input_path)
        raise
    return data


def write_json(data: dict, output_path: str, key: str | None = None) -> None:
    """Dump dictionary to disk as json, optionally subsetting to a key."""
    with open(output_path, "w") as f:
        if key:
            f.write(json.dumps(data[key]))
        else:
            f.write(json.dumps(data))


def load_comments_json_to_duck(comments_json) -> None:
    """Load comments json into duckdb"""
    con = duckdb.connect("comment_db.db")
    try:
        con.sql(
            f"""
            CREATE TABLE comments AS
                SELECT *
                FROM read_json('{comments_json}');
            """
        )
    except Exception:
        con.close()


def main(args):
    data = read_json(args.input_path)
    write_json(data, "local-data/comments-only.json", key="comments")
    load_comments_json_to_duck("local-data/comments-only.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=FILE_PATH, required=True, help="Path to zipped json data")
    args = parser.parse_args()

    main(args)
