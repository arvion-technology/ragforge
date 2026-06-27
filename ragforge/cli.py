"""Command line entry point: ``python -m ragforge <command>``.

Commands mirror the library so the CLI is a way to check the kit end to end
without writing code, and a way to see the same numbers the demo shows.
"""

from __future__ import annotations

import argparse
import json
import sys

from .answer import format_answer
from .evaluate import evaluate_at
from .ingest import IngestError
from .providers import complete_with_fallback, provider_status
from .retriever import Retriever, default_corpus_path, load_eval_labels


def _retriever(args) -> Retriever:
    return Retriever.from_corpus_file(
        args.corpus,
        size=args.size,
        overlap=args.overlap,
        strategy=args.strategy,
    )


def cmd_stats(args) -> int:
    retriever = _retriever(args)
    stats = retriever.stats()
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    print(f"corpus      {args.corpus or default_corpus_path()}")
    print(f"documents   {stats['index']['documents']}")
    print(f"chunks      {stats['index']['chunks']}")
    print(f"terms       {stats['index']['terms']}")
    print(f"avg chunk   {stats['chunks']['avg_tokens']} tokens")
    print(f"bm25        k1={stats['index']['k1']} b={stats['index']['b']}")
    return 0


def cmd_ask(args) -> int:
    retriever = _retriever(args)
    answer = retriever.ask(" ".join(args.question), k=args.k)
    if args.json:
        print(json.dumps(answer, indent=2))
        return 0
    print(format_answer(answer))
    if args.provider:
        from .answer import build_prompt

        generation = complete_with_fallback(build_prompt(answer["question"], answer), provider=args.provider)
        print(f"\nprovider: {generation['provider']}" + (" (degraded)" if generation["degraded"] else ""))
        print(generation["reason"] or generation["text"])
    return 0


def cmd_eval(args) -> int:
    retriever = _retriever(args)
    labels = load_eval_labels(args.labels)
    report = evaluate_at(retriever, labels, cutoffs=tuple(args.cutoffs))
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"queries: {report['detail']['queries']}")
    print(f"{'k':>3}  {'hit':>6}  {'mrr':>6}  {'recall':>6}")
    for row in report["rows"]:
        print(f"{row['k']:>3}  {row['hit_rate']:>6.3f}  {row['mrr']:>6.3f}  {row['recall']:>6.3f}")
    misses = report["detail"]["misses"]
    if misses:
        print("\nmisses")
        for row in misses:
            print(f"  expected {', '.join(row['relevant']):<24} got {', '.join(row['retrieved'][:3]) or '-'}")
    return 0


def cmd_providers(args) -> int:
    for status in provider_status():
        flag = "ready" if status["available"] else "not configured"
        print(f"{status['name']:<10} {status['model']:<28} {flag}")
        if status["env_var"]:
            print(f"{'':<10} env: {status['env_var']}")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. `pip install fastapi uvicorn`", file=sys.stderr)
        return 2
    from .api import create_app

    uvicorn.run(create_app(_retriever(args)), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragforge",
        description="Retrieval and cited answers over a local corpus. No API key required.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", help="JSON corpus or directory of documents")
    common.add_argument("--size", type=int, default=120, help="chunk size in tokens (default 120)")
    common.add_argument("--overlap", type=int, default=30, help="chunk overlap in tokens (default 30)")
    common.add_argument("--strategy", default="heading", choices=["heading", "fixed"])
    common.add_argument("--json", action="store_true", help="machine-readable output")

    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", parents=[common], help="index and corpus statistics")
    stats.set_defaults(func=cmd_stats)

    ask = sub.add_parser("ask", parents=[common], help="ask a question, print the cited answer")
    ask.add_argument("question", nargs="+")
    ask.add_argument("-k", type=int, default=4, help="passages to retrieve (default 4)")
    ask.add_argument("--provider", choices=["mock", "openai", "anthropic", "ollama"])
    ask.set_defaults(func=cmd_ask)

    evaluate = sub.add_parser("eval", parents=[common], help="run the labelled evaluation set")
    evaluate.add_argument("--labels", help="JSON file of labelled questions")
    evaluate.add_argument("--cutoffs", type=int, nargs="+", default=[1, 3, 5, 10])
    evaluate.set_defaults(func=cmd_eval)

    providers = sub.add_parser("providers", parents=[common], help="show adapter configuration")
    providers.set_defaults(func=cmd_providers)

    serve = sub.add_parser("serve", parents=[common], help="run the FastAPI app (needs fastapi)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
