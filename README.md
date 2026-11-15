# AetherShare (Demo 1 prototype)

Splits a document into visual shares that look like random noise. Any k shares
stacked on top of each other show the document; fewer than k show nothing.

## What it does
1. Takes the first page of a PDF (or a PNG/JPG) and turns it into a black and white bitmap.
2. Splits the bitmap into n shares with Naor-Shamir visual secret sharing (`vss.py`).
3. Saves each share in a different folder that stands in for a cloud bucket (S3, Google Cloud, Azure).
4. In the viewer, the browser fetches the chosen shares and stacks them on a canvas.
5. Lists stored documents and lets you open or delete them.

The original file and the combined image are never saved. The database holds metadata only.

## Run (Windows)
Double-click `run.bat`, or:

    py -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python app.py

Open http://127.0.0.1:5000 to see the login screen. Authentication is a UI
preview only; choose **Continue to the demo** or open http://127.0.0.1:5000/app
to use the document locker.

On startup, documents from the earlier one-page format are migrated automatically;
their existing share files are preserved.

## Check the security claim
    python verify.py

Prints, for every supported scheme, whether stacking 1, 2, ... n shares reveals the secret.

## Supported schemes
2 of 2, 2 of 3, 2 of 4, 3 of 3. General k of n is not implemented yet.

## Not done yet
- The "cloud" nodes are local folders. No AWS, Google Cloud or Azure account is used.
- The login screen is a UI preview; no accounts or authentication are configured.
- Black and white only, first page only, low contrast after stacking.
- The server does the splitting, so it briefly sees the document in memory.

## Files
- `app.py` Flask routes, database, image handling
- `vss.py` the visual secret sharing algorithm
- `verify.py` checks that k shares rebuild and fewer reveal nothing
- `index.html` the whole frontend
- `login.html` the login screen
- `data/` created on first run: `aethershare.db` and `clouds/<node>/` share files
