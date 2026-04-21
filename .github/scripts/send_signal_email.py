#!/usr/bin/env python3
import json
import os
import urllib.error
import urllib.request


def main() -> None:
    api_key = os.environ["RESEND_API_KEY"]
    to_email = os.environ["EMAIL_TO"]
    from_email = os.environ["EMAIL_FROM"]
    repository = os.environ["REPOSITORY"]
    run_url = os.environ["RUN_URL"]
    excerpt = os.environ.get("EXCERPT", "")

    subject = f"Option signal fired: {repository}"
    body = (
        "One or more option entry signals fired.\n\n"
        f"Repository: {repository}\n"
        f"Run URL: {run_url}\n\n"
        "Output excerpt:\n"
        "----------------------------------------\n"
        f"{excerpt}\n"
    )

    payload = {
        "from": from_email,
        "to": [to_email],
        "subject": subject,
        "text": body,
    }

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "option-signal-notifier/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            print(f"Resend accepted email request (HTTP {response.status}).")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Resend API failed (HTTP {exc.code}): {error_body}") from exc


if __name__ == "__main__":
    main()
