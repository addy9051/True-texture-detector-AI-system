"""Verify AWS credentials + Bedrock model access with one tiny (~$0.0002) call.

    uv run python scripts/check_bedrock.py

Prints the account, region, model id, the model's reply, and the exact cost.
On failure it explains the most likely fix.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from src.concierge.bedrock_client import BedrockChat


def main():
    # 1. Credentials. Two supported paths:
    #    a) IAM access keys in ~/.aws/credentials (verifiable via STS)
    #    b) Bedrock API key in AWS_BEARER_TOKEN_BEDROCK — works ONLY for
    #       Bedrock endpoints, so the STS identity check must be skipped.
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("[ok] Using Bedrock API key from AWS_BEARER_TOKEN_BEDROCK "
              "(bedrock-only credential; skipping STS identity check)")
    else:
        try:
            ident = boto3.client("sts").get_caller_identity()
            print(f"[ok] Credentials found — account {ident['Account']}")
            print(f"     principal: {ident['Arn']}")
        except NoCredentialsError:
            sys.exit("[fail] No AWS credentials. Either create "
                     "%USERPROFILE%\\.aws\\credentials via `aws configure` "
                     "(see README 'AWS setup'), or set a Bedrock API key: "
                     '$env:AWS_BEARER_TOKEN_BEDROCK = "<key>".')

    # 2. Bedrock model ping
    chat = BedrockChat()
    print(f"[..] Pinging {chat.model_id} in {chat.region} ...")
    try:
        resp = chat.converse(
            system="Reply with exactly: OK",
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            max_tokens=10, temperature=0.0)
        text = resp["output"]["message"]["content"][0].get("text", "").strip()
        print(f"[ok] Model replied: {text!r}")

        # 3. Native tool-use probe — the concierge prefers forced tool calls but
        #    falls back to a JSON protocol if the account/model rejects toolConfig.
        try:
            chat.converse(
                system="Call the ping tool.",
                messages=[{"role": "user", "content": [{"text": "ping"}]}],
                tool_config={"tools": [{"toolSpec": {
                    "name": "ping", "description": "connectivity test",
                    "inputSchema": {"json": {"type": "object", "properties": {}}}}}],
                    "toolChoice": {"any": {}}},
                max_tokens=50)
            print("[ok] Native tool use supported — concierge will force tool calls")
        except ClientError as e:
            err = e.response.get("Error", {})
            print(f"[warn] Native tool use rejected ({err.get('Code')}: "
                  f"{err.get('Message')})")
            print("       Concierge will automatically use its JSON-protocol fallback.")

        print(f"\nTotal cost of this check: {chat.meter.summary()}")
        print("All good — run: uv run python scripts/run_concierge.py")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        print(f"[fail] {code}: {e.response.get('Error', {}).get('Message', '')}")
        if code == "AccessDeniedException":
            print("  Likely fixes:")
            print("  - Your IAM user/key lacks bedrock:InvokeModel permission")
            print("  - If you switched to an Anthropic model id: Claude is AWS-"
                  "Marketplace-billed and NOT covered by promotional credits — "
                  "stick with amazon.nova-* models")
        elif "Operation not allowed" in e.response.get("Error", {}).get("Message", ""):
            print("  ACCOUNT-LEVEL restriction — not a code/region/IAM problem:")
            print("  - New accounts on the AWS Free Tier *free plan* cannot invoke "
                  "Bedrock models. Fix: Billing console -> Free Tier -> upgrade to "
                  "the Paid plan (your promotional credits carry over and are spent "
                  "before any real charges).")
            print("  - If already on the Paid plan: open a free support case under "
                  "'Account and billing' to lift the restriction, and try Nova once "
                  "in the Bedrock console Playground.")
        elif code in ("ResourceNotFoundException", "ValidationException"):
            print("  The model/inference profile may not exist in this region.")
            print('  - Mistral Large 24.07 launched in Oregon: try '
                  '$env:BEDROCK_REGION = "us-west-2"')
            print('  - Or switch model: $env:BEDROCK_MODEL_ID = "us.amazon.nova-pro-v1:0" '
                  'with $env:BEDROCK_REGION = "us-east-1"')
        sys.exit(1)


if __name__ == "__main__":
    main()
