"""Classify one SMS or run a simple interactive terminal demo."""

import argparse
from pathlib import Path
import joblib
from sms_classifier import predict_sms, setup_nltk

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="?", help="Quote the complete SMS message.")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models" / "sms_spam_model.joblib")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("Model missing. Run python train.py first.")
    setup_nltk()
    model = joblib.load(args.model)
    if args.message is not None:
        try:
            print(predict_sms(model, args.message))
        except ValueError as exc:
            parser.error(str(exc))
        return
    print("SMS classifier — type exit to stop.")
    while True:
        try:
            text = input("Message: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"exit", "quit"}:
            break
        try:
            print(predict_sms(model, text))
        except ValueError as exc:
            print(exc)


if __name__ == "__main__":
    main()
