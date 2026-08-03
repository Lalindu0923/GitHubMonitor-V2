from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


def _normalize_text(value: str) -> str:
    return value.strip() if value else ""


def _build_summary(commit: dict) -> tuple[str, list[dict[str, Any]], int, int]:
    message = commit.get("commit", {}).get("message", "")
    files = commit.get("files", [])

    additions = sum(file.get("additions", 0) for file in files)
    deletions = sum(file.get("deletions", 0) for file in files)

    return message, files, additions, deletions


def _local_score(message: str, files: list[dict[str, Any]], additions: int) -> tuple[int, str]:
    score = 0

    lowered_message = message.lower()

    if any(keyword in lowered_message for keyword in ("fix", "bug", "patch")):
        score += 15

    if any(keyword in lowered_message for keyword in ("feature", "add", "implement")):
        score += 10

    if additions > 20:
        score += 10

    if len(files) > 3:
        score += 5

    if score >= 20:
        result = "good"
    elif score >= 10:
        result = "needs-review"
    else:
        result = "needs-improvement"

    return score, result


def _build_prompt(commit: dict, local_score: int, local_result: str) -> str:
    message = _normalize_text(commit.get("commit", {}).get("message", ""))
    files = commit.get("files", [])

    file_lines = []
    for file in files:
        filename = file.get("filename") or file.get("path") or "unknown-file"
        additions = file.get("additions", 0)
        deletions = file.get("deletions", 0)
        patch = file.get("patch") or ""
        snippet = patch[:1200]
        file_lines.append(
            f"- {filename} (+{additions} / -{deletions})\n"
            f"  Patch: {snippet}"
        )

    file_block = "\n".join(file_lines) if file_lines else "- No file details were provided."

    return (
        "You are a senior AI code reviewer for commit analysis.\n"
        "Analyze the commit message and the actual code changes (patch details) in the changed files.\n"
        "Evaluate the code quality of the added or modified lines and produce a concise JSON response with these keys:\n"
        "score (0-100 integer), verdict, summary, strengths, risks, suggestions.\n\n"
        "CRITICAL SCORING CRITERIA FOR THE SCORE (0-100):\n"
        "- The score MUST reflect the actual quality and correctness of the code lines, rather than just the size of the commit or keyword matches.\n"
        "- DO NOT anchor your score to the 'Local heuristic score' (which only ranges from 0-40 based on simple keyword matches and additions count). Treat it as secondary metadata.\n"
        "- Small, clean, and bug-free changes (such as simple fixes, documentation additions, or modular refactoring) are considered HIGH quality and should receive a high score (85-100).\n"
        "- SCORING SCALE:\n"
        "  * 85 - 100 (Verdict: \"good\"): The code changes are clean, functional, follow best practices, have good readability, proper error handling, and have no noticeable bugs or security issues.\n"
        "  * 60 - 84 (Verdict: \"needs-review\"): The code is functional but could be improved (e.g., missing comments, redundant logic, slightly suboptimal algorithm, or lack of proper error boundaries).\n"
        "  * 0 - 59 (Verdict: \"needs-improvement\"): The code has critical issues, logic bugs, syntax errors, potential security vulnerabilities (like hardcoded keys), or lacks any meaningful structure.\n\n"
        "CRITICAL ENFORCED VERDICT VALUES:\n"
        "The 'verdict' key in your response MUST be exactly one of the following lowercase strings:\n"
        "- \"good\" (score 85-100)\n"
        "- \"needs-review\" (score 60-84)\n"
        "- \"needs-improvement\" (score 0-59)\n\n"
        f"Commit message: {message or 'No commit message provided.'}\n"
        f"Local heuristic score: {local_score} (Note: This is a basic 0-40 structural heuristic, NOT a code quality score. Do not anchor on it.)\n"
        f"Local heuristic result: {local_result}\n\n"
        f"Changed files:\n{file_block}\n\n"
        "Return JSON only."
    )


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

    return None


def _call_llm(prompt: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if (provider == "gemini") or (not provider and gemini_api_key):
        provider = "gemini"
        model_name = os.getenv("MODEL") or "gemini-2.5-flash"
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            content = response.text
            llm_json = _extract_json_from_text(content)
            if llm_json is not None:
                return llm_json, provider, model_name
            return {"summary": content}, provider, model_name
        except Exception as e:
            error_msg = f"Gemini API Error: {str(e)}"
            print(error_msg)
            return {"summary": error_msg}, provider, model_name

    # Otherwise fallback to openai/llama
    if not provider:
        provider = "openai"

    if provider == "llama":
        model = os.getenv("LLM_MODEL") or "qwen3:8b"
    else:
        model = os.getenv("LLM_MODEL") or os.getenv("MODEL") or "gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""

    if provider == "llama":
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    else:
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    if not api_key and provider != "llama":
        return None, provider, model

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert code review assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        # Note: Set timeout to 120 seconds for local runs as model loading might take time.
        timeout = 120 if provider == "llama" else 30

        candidate_requests: list[tuple[str, dict[str, Any], str]] = []
        if provider == "llama":
            normalized_base_url = base_url.rstrip("/")
            if normalized_base_url.endswith("/v1"):
                candidate_requests.append(
                    (
                        f"{normalized_base_url}/chat/completions",
                        payload,
                        "openai-compatible",
                    )
                )
                candidate_requests.append(
                    (
                        f"{normalized_base_url[:-3]}/api/chat",
                        {
                            "model": model,
                            "messages": payload["messages"],
                            "temperature": payload["temperature"],
                            "stream": False,
                        },
                        "ollama-native",
                    )
                )
            else:
                candidate_requests.append(
                    (
                        f"{normalized_base_url}/api/chat",
                        {
                            "model": model,
                            "messages": payload["messages"],
                            "temperature": payload["temperature"],
                            "stream": False,
                        },
                        "ollama-native",
                    )
                )
                candidate_requests.append(
                    (
                        f"{normalized_base_url}/chat/completions",
                        payload,
                        "openai-compatible",
                    )
                )
        else:
            candidate_requests.append(
                (f"{base_url.rstrip('/')}/chat/completions", payload, "openai-compatible")
            )

        last_error: Exception | None = None
        for request_url, request_payload, request_mode in candidate_requests:
            request = urllib.request.Request(
                request_url,
                data=json.dumps(request_payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw_data = response.read().decode("utf-8")
                parsed = json.loads(raw_data)

                if request_mode == "ollama-native":
                    content = parsed.get("message", {}).get("content", "")
                else:
                    content = parsed["choices"][0]["message"]["content"]

                llm_json = _extract_json_from_text(content)
                if llm_json is not None:
                    return llm_json, provider, model
                return {"summary": content}, provider, model
            except urllib.error.HTTPError as e:
                last_error = e
                if provider == "llama" and e.code == 404:
                    continue
                raise

        if last_error is not None:
            raise last_error
    except Exception as e:
        print(f"Error calling LLM (provider={provider}): {e}")
        return None, provider, model


def analyze_commit_logic(commit: dict):
    message, files, additions, deletions = _build_summary(commit)
    local_score, local_result = _local_score(message, files, additions)
    prompt = _build_prompt(commit, local_score, local_result)

    llm_review, provider, model = _call_llm(prompt)

    ai_score = local_score
    ai_result = local_result
    ai_review = ""

    if llm_review:
        ai_review = str(
            llm_review.get("summary")
            or llm_review.get("review")
            or llm_review.get("verdict")
            or "AI review generated successfully."
        )

        parsed_score = llm_review.get("score")
        if isinstance(parsed_score, int):
            ai_score = max(0, min(100, parsed_score))

        parsed_result = llm_review.get("verdict")
        if isinstance(parsed_result, str) and parsed_result.strip():
            ai_result = parsed_result.strip()
    else:
        ai_review = (
            "LLM review unavailable, so the response was generated using local heuristics only. "
            "Configure OPENAI_API_KEY with LLM_BASE_URL or set LLM_PROVIDER=llama for a local OpenAI-compatible server."
        )

    return {
        "score": ai_score,
        "result": ai_result,
        "files_changed": len(files),
        "additions": additions,
        "deletions": deletions,
        "ai_review": ai_review,
        "prompt": prompt,
        "provider": provider,
        "model": model,
        "raw_llm_output": llm_review,
    }