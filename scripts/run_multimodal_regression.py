"""HTTP smoke test for local multimodal regression modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

import requests


DEFAULT_QUESTIONS = {
    "text": "请总结这份文档的主要内容。",
    "image": "这份文档中的图片主要展示了什么？",
    "equation": "请解释文档中的主要公式和适用条件。",
    "table": "请总结表格中的关键信息。",
    "full": "请结合文字、图片、表格和公式总结这份文档。",
}

GENERIC_SUMMARY_QUESTION = "请给出这个 PDF 的详细总结"

EMPTY_CONTEXT_ANSWER_MARKERS = (
    "Document Chunks 为空",
    "Context 中不包含任何实际",
    "Context 不包含文档内容",
    "Reference Document List 也为空",
    "无法获取该 PDF 的总体文本",
    "no document chunks",
    "empty context",
)

EXPECTED_ENV_HINTS = {
    "text": "ENABLE_MULTIMODAL=false",
    "image": (
        "ENABLE_MULTIMODAL=true, ENABLE_IMAGE_PROCESSING=true, "
        "ENABLE_TABLE_PROCESSING=false, ENABLE_EQUATION_PROCESSING=false"
    ),
    "equation": (
        "ENABLE_MULTIMODAL=true, ENABLE_EQUATION_PROCESSING=true, "
        "ENABLE_IMAGE_PROCESSING=false, ENABLE_TABLE_PROCESSING=false"
    ),
    "table": (
        "ENABLE_MULTIMODAL=true, ENABLE_TABLE_PROCESSING=true, "
        "ENABLE_IMAGE_PROCESSING=false, ENABLE_EQUATION_PROCESSING=false"
    ),
    "full": (
        "ENABLE_MULTIMODAL=true, ENABLE_IMAGE_PROCESSING=true, "
        "ENABLE_TABLE_PROCESSING=true, ENABLE_EQUATION_PROCESSING=true"
    ),
}

PROCESS_FIELDS = [
    "text_indexed",
    "multimodal_enabled",
    "multimodal_status",
    "image_status",
    "table_status",
    "equation_status",
    "formula_status",
    "skipped_multimodal_items_count",
    "skipped_by_reason",
    "multimodal_warnings_count",
    "warnings_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local HTTP regression smoke test against FastAPI."
    )
    parser.add_argument("--pdf", required=True, help="Path to the local PDF to upload.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["text", "image", "equation", "table", "full"],
        help="Expected backend mode. The script does not modify .env.",
    )
    parser.add_argument("--question", help="Question to send to /api/chat.")
    parser.add_argument(
        "--question-generic-summary",
        action="store_true",
        help='Use the generic summary question: "请给出这个 PDF 的详细总结".',
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI base URL.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip upload and use the PDF filename as document_id.",
    )
    parser.add_argument("--output", help="Optional path to save a JSON report.")
    return parser.parse_args()


def print_step(name: str, response: requests.Response) -> None:
    print(f"[{name}] HTTP {response.status_code}")


def parse_json(response: requests.Response, step: str) -> object | None:
    try:
        return response.json()
    except ValueError:
        print(f"[{step}] Response is not JSON:")
        print(response.text[:1000])
        return None


def fail_hint(step: str, response: requests.Response) -> None:
    print(f"[FAIL] {step} failed with HTTP {response.status_code}", file=sys.stderr)
    print(response.text[:2000], file=sys.stderr)
    if response.status_code >= 500:
        print("Hint: check the uvicorn backend log.", file=sys.stderr)
    elif response.status_code == 409:
        print(
            "Hint: knowledge base is not ready; process may not have indexed text.",
            file=sys.stderr,
        )
    elif response.status_code == 404:
        print(
            "Hint: document_id may be wrong or URL encoding failed.",
            file=sys.stderr,
        )


def save_report(output_path: str | None, report: dict) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved report:", path)


def answer_has_empty_context_marker(answer: str) -> bool:
    normalized = answer.lower()
    return any(marker.lower() in normalized for marker in EMPTY_CONTEXT_ANSWER_MARKERS)


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[FAIL] PDF does not exist: {pdf_path}", file=sys.stderr)
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        print(f"[FAIL] Only PDF files are supported: {pdf_path}", file=sys.stderr)
        return 2

    question = (
        GENERIC_SUMMARY_QUESTION
        if args.question_generic_summary
        else args.question or DEFAULT_QUESTIONS[args.mode]
    )
    document_id = pdf_path.name
    encoded_id = quote(document_id, safe="")

    print("Mode:", args.mode)
    print("Expected .env:", EXPECTED_ENV_HINTS[args.mode])
    print("Reminder: after changing .env, restart uvicorn before running this script.")
    print("PDF:", pdf_path)
    print("Document ID:", document_id)
    print("Base URL:", args.base_url)

    report: dict = {
        "mode": args.mode,
        "pdf": str(pdf_path),
        "document_id": document_id,
        "base_url": args.base_url,
        "question": question,
        "steps": {},
    }

    session = requests.Session()

    try:
        health = session.get(f"{args.base_url}/api/health", timeout=30)
    except requests.RequestException as exc:
        print(f"[FAIL] health request failed: {exc}", file=sys.stderr)
        return 1
    print_step("health", health)
    health_json = parse_json(health, "health")
    report["steps"]["health"] = {
        "status_code": health.status_code,
        "json": health_json,
    }
    if health.status_code != 200:
        fail_hint("health", health)
        save_report(args.output, report)
        return 1

    if not args.no_upload:
        try:
            with pdf_path.open("rb") as file:
                upload = session.post(
                    f"{args.base_url}/api/upload",
                    files={"file": (pdf_path.name, file, "application/pdf")},
                    timeout=120,
                )
        except OSError as exc:
            print(f"[FAIL] failed to read PDF: {exc}", file=sys.stderr)
            return 2
        except requests.RequestException as exc:
            print(f"[FAIL] upload request failed: {exc}", file=sys.stderr)
            return 1

        print_step("upload", upload)
        upload_json = parse_json(upload, "upload")
        report["steps"]["upload"] = {
            "status_code": upload.status_code,
            "json": upload_json,
        }
        if upload.status_code != 200:
            fail_hint("upload", upload)
            save_report(args.output, report)
            return 1
        if isinstance(upload_json, dict):
            uploaded_id = upload_json.get("document", {}).get("id")
            if uploaded_id:
                document_id = uploaded_id
                encoded_id = quote(document_id, safe="")
                report["document_id"] = document_id

    try:
        process = session.post(
            f"{args.base_url}/api/documents/{encoded_id}/process",
            timeout=1800,
        )
    except requests.RequestException as exc:
        print(f"[FAIL] process request failed: {exc}", file=sys.stderr)
        save_report(args.output, report)
        return 1

    print_step("process", process)
    process_json = parse_json(process, "process")
    report["steps"]["process"] = {
        "status_code": process.status_code,
        "json": process_json,
    }
    if process.status_code != 200:
        fail_hint("process", process)
        save_report(args.output, report)
        return 1
    if not isinstance(process_json, dict):
        print("[FAIL] process response JSON is not an object.", file=sys.stderr)
        save_report(args.output, report)
        return 1

    print("\nProcess summary:")
    for field in PROCESS_FIELDS:
        print(f"- {field}: {process_json.get(field)}")

    if process_json.get("text_indexed") is not True:
        print(
            "[FAIL] text_indexed is not true; /api/chat is expected to be unreliable.",
            file=sys.stderr,
        )
        save_report(args.output, report)
        return 1

    chat_body = {
        "question": question,
        "document_id": document_id,
        "level": "undergraduate",
        "mode": "hybrid",
    }
    try:
        chat = session.post(
            f"{args.base_url}/api/chat",
            json=chat_body,
            timeout=300,
        )
    except requests.RequestException as exc:
        print(f"[FAIL] chat request failed: {exc}", file=sys.stderr)
        save_report(args.output, report)
        return 1

    print_step("chat", chat)
    chat_json = parse_json(chat, "chat")
    report["steps"]["chat"] = {
        "status_code": chat.status_code,
        "json": chat_json,
    }
    if chat.status_code != 200:
        fail_hint("chat", chat)
        save_report(args.output, report)
        return 1
    if not isinstance(chat_json, dict):
        print("[FAIL] chat response JSON is not an object.", file=sys.stderr)
        save_report(args.output, report)
        return 1

    answer = str(chat_json.get("answer") or "")
    print("Chat answer present:", bool(answer.strip()))
    if not answer.strip():
        print("[FAIL] chat response has no answer.", file=sys.stderr)
        save_report(args.output, report)
        return 1
    if answer_has_empty_context_marker(answer):
        print(
            "[FAIL] chat answer indicates empty document context.",
            file=sys.stderr,
        )
        save_report(args.output, report)
        return 1

    save_report(args.output, report)
    print("[PASS] Regression smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
