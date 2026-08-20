"""
Visits a Streamlit Cloud app with a real (headless) browser so it counts as genuine
traffic -- a plain HTTP request often does NOT keep a Streamlit app awake, since the
app needs an actual browser connection (WebSocket), not just an HTTP response.

If the app is asleep, this clicks the "Yes, get this app back up!" button for you.
"""
import os
import sys
from playwright.sync_api import sync_playwright

APP_URL = os.environ["STREAMLIT_APP_URL"]  # set as a GitHub secret, see setup steps


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"Visiting {APP_URL} ...")
        page.goto(APP_URL, timeout=60000)

        # If the app is asleep, Streamlit shows a wake-up button. If it's already
        # awake, this button never appears, and we just move on.
        try:
            wake_button = page.get_by_role("button", name="Yes, get this app back up!")
            wake_button.wait_for(timeout=10000)
            print("App was asleep -- clicking the wake-up button...")
            wake_button.click()
            page.wait_for_timeout(15000)  # give it a moment to actually start booting
            print("Wake-up triggered.")
        except Exception:
            print("App was already awake -- nothing to do.")

        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Keepalive run failed: {e}")
        sys.exit(1)
