"""One-time sign-in for the mail and calendar accounts ORACLE reads.

    python connect_accounts.py            # Gmail and Outlook
    python connect_accounts.py gmail      # just one

Each opens your browser for the provider's own sign-in page; the tokens are
cached under %LOCALAPPDATA%\\ORACLE and refreshed silently afterwards.
"""
import sys

import core


def connect_gmail():
    core._get_gmail_token()
    r = core._gmail_request("GET", "/users/me/profile")
    r.raise_for_status()
    return r.json().get("emailAddress", "?")


def connect_outlook():
    token = core._get_graph_token()
    r = core.requests.get(f"{core.GRAPH_BASE}/me", headers={"Authorization": f"Bearer {token}"}, timeout=15)
    r.raise_for_status()
    me = r.json()
    return me.get("mail") or me.get("userPrincipalName", "?")


def main():
    wanted = [a.lower() for a in sys.argv[1:]] or ["gmail", "outlook"]
    for name, connect in (("gmail", connect_gmail), ("outlook", connect_outlook)):
        if name not in wanted:
            continue
        print(f"{name}: signing in (check your browser)...", flush=True)
        try:
            print(f"{name}: connected as {connect()}", flush=True)
        except Exception as e:
            print(f"{name}: not connected - {e}", flush=True)


if __name__ == "__main__":
    main()
