import json
import sys
import traceback

from benchmarks.compilation_time.cases import registry

def main():
    spec = None

    try:
        spec = json.load(sys.stdin)

        runner = spec["runner"]
        result = registry[runner](spec)

        print(json.dumps(result))
    except Exception:
        print(
            json.dumps(
                {
                    "error": "worker failed",
                    "runner": spec.get("runner") if isinstance(spec, dict) else None,
                    "traceback": traceback.format_exc(),
                }
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()