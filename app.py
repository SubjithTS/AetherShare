"""AetherShare demo backend.

Upload a PDF or image -> every page is rendered -> each page is turned into a
binary bitmap (black and white, or one halftone bitmap per color channel) ->
each bitmap is split into n visual shares (k-of-n) -> share i of every page is
saved in one simulated cloud folder. Only metadata is stored in the database;
the original file and the combined image are never written to disk.
"""
import io
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

import vss

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DB_PATH = DATA / "aethershare.db"

# Folders that stand in for the real cloud buckets.
NODES = {
    "s3": "Amazon S3 (simulated)",
    "gcs": "Google Cloud Storage (simulated)",
    "azure": "Azure Blob Storage (simulated)",
}
NODE_IDS = list(NODES)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD = 25 * 1024 * 1024
MAX_PAGES = 30
MODES = ("color", "bw")
# Largest size (width, height) in pixels that one page is rendered to.
QUALITY = {"standard": (1000, 1400), "high": (1600, 2300)}
THRESHOLD = 160          # black and white mode: gray values below this become black

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD


# ---------- database ----------

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DATA.mkdir(exist_ok=True)
    for node in NODE_IDS:
        (DATA / "clouds" / node).mkdir(parents=True, exist_ok=True)
    with closing(connect()) as db, db:
        columns = [row["name"] for row in db.execute("PRAGMA table_info(documents)")]
        legacy_schema = bool(columns and "pages" not in columns)
        legacy_documents = []
        legacy_shares = []
        if legacy_schema:
            legacy_documents = db.execute("SELECT * FROM documents").fetchall()
            legacy_shares = db.execute("SELECT * FROM shares").fetchall()
            db.execute("ALTER TABLE shares RENAME TO shares_legacy")
            db.execute("ALTER TABLE documents RENAME TO documents_legacy")
        db.execute("""CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT NOT NULL,
            k INTEGER NOT NULL, n INTEGER NOT NULL, m INTEGER NOT NULL,
            mode TEXT NOT NULL, pages INTEGER NOT NULL,
            white_level REAL NOT NULL, black_level REAL NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS pages (
            document_id TEXT NOT NULL REFERENCES documents(id),
            page INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
            PRIMARY KEY (document_id, page))""")
        db.execute("""CREATE TABLE IF NOT EXISTS shares (
            document_id TEXT NOT NULL REFERENCES documents(id),
            page INTEGER NOT NULL, share_index INTEGER NOT NULL,
            node TEXT NOT NULL, path TEXT NOT NULL,
            PRIMARY KEY (document_id, page, share_index))""")
        db.execute("""CREATE TABLE IF NOT EXISTS nodes (
            node TEXT PRIMARY KEY, online INTEGER NOT NULL DEFAULT 1)""")
        for node in NODE_IDS:
            db.execute("INSERT OR IGNORE INTO nodes (node) VALUES (?)", (node,))
        for doc in legacy_documents:
            db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (doc["id"], doc["filename"], doc["created_at"], doc["k"], doc["n"],
                        doc["m"], "bw", 1, doc["white_level"], doc["black_level"]))
            db.execute("INSERT INTO pages VALUES (?,?,?,?)",
                       (doc["id"], 1, doc["width"], doc["height"]))
        for share in legacy_shares:
            db.execute("INSERT INTO shares VALUES (?,?,?,?,?)",
                       (share["document_id"], 1, share["share_index"], share["node"], share["path"]))
        if legacy_schema:
            db.execute("DROP TABLE shares_legacy")
            db.execute("DROP TABLE documents_legacy")


# ---------- image handling ----------

class UploadError(Exception):
    """The upload cannot be used; the message is safe to show to the user."""


def fit_image(img, width, max_height):
    scale = min(width / img.width, max_height / img.height)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.LANCZOS)


def render_pages(data, filename, width, max_height):
    """Yield every page of the upload as an RGB image that fits in width x max_height."""
    if Path(filename).suffix.lower() == ".pdf":
        import fitz  # PyMuPDF
        try:
            pdf = fitz.open(stream=data, filetype="pdf")
        except Exception:
            raise UploadError("Could not read that file as a PDF.")
        try:
            if pdf.page_count == 0:
                raise UploadError("The PDF has no pages.")
            if pdf.page_count > MAX_PAGES:
                raise UploadError(f"This demo supports PDFs of up to {MAX_PAGES} pages.")
            for number in range(pdf.page_count):
                page = pdf[number]
                zoom = min(width / page.rect.width, max_height / page.rect.height)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
                yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            pdf.close()
    else:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception:
            raise UploadError("Could not read that file as an image.")
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            img = Image.alpha_composite(Image.new("RGBA", img.size, "white"), img)
        yield fit_image(img.convert("RGB"), width, max_height)


def secret_channels(img, mode):
    """Turn a page into binary bitmaps (1 = black).

    bw: one bitmap made with a fixed threshold, which keeps text sharp.
    color: three bitmaps, one per red, green and blue channel. Each channel is
    halftoned with Floyd-Steinberg error diffusion so that shades survive as dot
    density, and later stacked and averaged back into shades.
    """
    if mode == "bw":
        return [(np.asarray(img.convert("L")) < THRESHOLD).astype(np.uint8)]
    channels = []
    for band in img.split():
        halftone = np.asarray(band.convert("1"))      # True = white
        channels.append((~halftone).astype(np.uint8))
    return channels


def share_to_png(channels):
    """Encode one share. `channels` holds one array (bw) or three arrays (R, G, B), 1 = black."""
    buffer = io.BytesIO()
    if len(channels) == 1:
        pixels = np.where(channels[0] == 1, 0, 255).astype(np.uint8)
        Image.fromarray(pixels).convert("1", dither=Image.Dither.NONE).save(buffer, format="PNG")
    else:
        # Every subpixel is one of 8 colors (each channel fully on or off), so a
        # palette image keeps the files small.
        red, green, blue = channels
        index = ((red << 2) | (green << 1) | blue).astype(np.uint8)
        palette = []
        for i in range(8):
            palette += [0 if i & 4 else 255, 0 if i & 2 else 255, 0 if i & 1 else 255]
        image = Image.fromarray(index)
        image.putpalette(palette)
        image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


# ---------- helpers ----------

def error(message, status=400):
    return jsonify(error=message), status


def parse_scheme(text):
    try:
        k, n = (int(part) for part in text.split("-"))
    except (ValueError, AttributeError):
        return None
    return (k, n) if (k, n) in vss.SUPPORTED else None


def document_json(row):
    return {
        "id": row["id"], "filename": row["filename"], "created_at": row["created_at"],
        "k": row["k"], "n": row["n"], "m": row["m"],
        "mode": row["mode"], "pages": row["pages"],
        "white_level": row["white_level"], "black_level": row["black_level"],
    }


# ---------- routes ----------

@app.get("/")
@app.get("/login")
def login():
    return send_file(BASE / "login.html")


@app.get("/app")
def index():
    return send_file(BASE / "index.html")


@app.post("/api/documents")
def create_document():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return error("Choose a file first.")
    if Path(upload.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return error("Only PDF, PNG and JPG files are supported.")
    scheme = parse_scheme(request.form.get("scheme", "2-3"))
    if scheme is None:
        return error("Unsupported k-of-n choice.")
    k, n = scheme
    mode = request.form.get("mode", "color")
    if mode not in MODES:
        return error("Unsupported mode.")
    quality = request.form.get("quality", "standard")
    if quality not in QUALITY:
        return error("Unsupported quality.")
    max_width, max_height = QUALITY[quality]

    data = upload.read()
    if not data:
        return error("The file is empty.")

    m = vss.expansion(k, n)
    white_level, black_level = vss.stacked_lightness(k, n)
    doc_id = str(uuid.uuid4())
    written = []
    try:
        with closing(connect()) as db, db:
            page_count = 0
            for page_number, img in enumerate(render_pages(data, upload.filename, max_width, max_height), start=1):
                channels = secret_channels(img, mode)
                height, width = channels[0].shape
                # split_channels[c][i] is share i of channel c
                split_channels = [vss.split(channel, k, n) for channel in channels]
                if page_number == 1:
                    db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)",
                               (doc_id, Path(upload.filename).name,
                                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                k, n, m, mode, 0, white_level, black_level))
                db.execute("INSERT INTO pages VALUES (?,?,?,?)", (doc_id, page_number, width, height))
                for i in range(1, n + 1):
                    node = NODE_IDS[(i - 1) % len(NODE_IDS)]
                    path = DATA / "clouds" / node / f"{doc_id}_p{page_number}_share{i}.png"
                    path.write_bytes(share_to_png([split[i - 1] for split in split_channels]))
                    written.append(path)
                    db.execute("INSERT INTO shares VALUES (?,?,?,?,?)",
                               (doc_id, page_number, i, node, str(path.relative_to(BASE))))
                page_count = page_number
            db.execute("UPDATE documents SET pages=? WHERE id=?", (page_count, doc_id))
    except UploadError as exc:
        remove_files(written)
        return error(str(exc))
    except Exception:
        remove_files(written)
        raise
    return jsonify(id=doc_id), 201


def remove_files(paths):
    for path in paths:
        try:
            path.unlink()
        except OSError:
            pass


@app.get("/api/documents")
def list_documents():
    with closing(connect()) as db:
        rows = db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    return jsonify([document_json(r) for r in rows])


@app.delete("/api/documents/<doc_id>")
def delete_document(doc_id):
    with closing(connect()) as db, db:
        doc = db.execute("SELECT id FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not doc:
            return error("Document not found.", 404)

        rows = db.execute("SELECT path FROM shares WHERE document_id=?", (doc_id,)).fetchall()
        cloud_root = (DATA / "clouds").resolve()
        share_paths = []
        for row in rows:
            share_path = (BASE / row["path"]).resolve()
            try:
                relative_path = share_path.relative_to(cloud_root)
            except ValueError:
                return error("Stored share path is invalid.", 500)
            if not relative_path.parts:
                return error("Stored share path is invalid.", 500)
            share_paths.append(share_path)

        db.execute("DELETE FROM shares WHERE document_id=?", (doc_id,))
        db.execute("DELETE FROM pages WHERE document_id=?", (doc_id,))
        db.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    cleanup_failed = False
    for share_path in share_paths:
        try:
            share_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            cleanup_failed = True
    if cleanup_failed:
        return error("Document was deleted, but one or more share files could not be removed.", 500)
    return jsonify(ok=True)


@app.get("/api/documents/<doc_id>")
def get_document(doc_id):
    with closing(connect()) as db:
        doc = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not doc:
            return error("Document not found.", 404)
        rows = db.execute("""SELECT DISTINCT s.share_index, s.node, n.online FROM shares s
                             JOIN nodes n ON n.node = s.node
                             WHERE s.document_id=? ORDER BY s.share_index""", (doc_id,)).fetchall()
        pages = db.execute("SELECT page, width, height FROM pages WHERE document_id=? ORDER BY page",
                           (doc_id,)).fetchall()
    result = document_json(doc)
    result["shares"] = [{"index": r["share_index"], "node": r["node"], "node_label": NODES[r["node"]],
                         "online": bool(r["online"])} for r in rows]
    result["page_sizes"] = [dict(p) for p in pages]
    return jsonify(result)


@app.get("/api/documents/<doc_id>/pages/<int:page>/shares/<int:index>")
def get_share(doc_id, page, index):
    with closing(connect()) as db:
        row = db.execute("""SELECT s.path, n.online FROM shares s JOIN nodes n ON n.node = s.node
                            WHERE s.document_id=? AND s.page=? AND s.share_index=?""",
                         (doc_id, page, index)).fetchone()
    if not row:
        return error("Share not found.", 404)
    if not row["online"]:
        return error("This cloud node is offline.", 503)
    return send_file(BASE / row["path"], mimetype="image/png")


@app.get("/api/nodes")
def list_nodes():
    with closing(connect()) as db:
        rows = db.execute("""SELECT n.node, n.online, COUNT(s.node) AS share_count FROM nodes n
                             LEFT JOIN shares s ON s.node = n.node GROUP BY n.node""").fetchall()
    by_id = {r["node"]: r for r in rows}
    return jsonify([{"node": node, "label": NODES[node], "online": bool(by_id[node]["online"]),
                     "share_count": by_id[node]["share_count"]} for node in NODE_IDS])


@app.post("/api/nodes/<node>/toggle")
def toggle_node(node):
    if node not in NODES:
        return error("Unknown node.", 404)
    with closing(connect()) as db, db:
        db.execute("UPDATE nodes SET online = 1 - online WHERE node=?", (node,))
    return jsonify(ok=True)


@app.errorhandler(413)
def too_large(_):
    return error("Maximum upload size is 10 MB.", 413)


init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
