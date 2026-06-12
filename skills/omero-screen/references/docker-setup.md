# OMERO Server Setup

Docs: https://hocheggerlab.github.io/omero-screen/
OMERO Docker: https://github.com/ome/docker-stacks

---

## Production OMERO Server

The production server runs via Docker Compose (`docker-compose.yml` in project root). It is managed separately from the test server.

```bash
# Start production server
docker-compose up -d

# Stop production server
docker-compose down

# Check status
docker-compose ps
```

Production credentials live in `.env.production`. Never run the test server while the production server is up — they share port 4064.

---

## Test Server (for development and e2e testing)

A separate OMERO instance on **127.0.0.2:4064** (loopback alias) so it never conflicts with the production server at 127.0.0.1.

### Loopback alias setup (macOS — one-time)
```bash
sudo ifconfig lo0 alias 127.0.0.2
```
This does not persist across reboots. Add to a login script or macOS launchd plist if needed.

### Test server management
```bash
# Start test server (stops production server first if running)
./scripts/manage_test_server.sh start

# Stop test server
./scripts/manage_test_server.sh stop

# Check if running
./scripts/manage_test_server.sh status

# Restart
./scripts/manage_test_server.sh restart
```

**Test credentials:** `root` / `omero` at `127.0.0.2:4064`

The test server uses `docker-compose.test.yml`. It waits up to 60 seconds for OMERO to be ready before returning.

---

## Loading Test Data

Test plates must be uploaded to the test OMERO server before e2e tests can run.

```bash
# Load all plates from a directory
./scripts/load_plates.sh -d /path/to/test/plates -x

# The -x flag enables extended/Excel metadata loading
# The script authenticates interactively if needed
```

The `scripts/download_sample_data.py` can fetch sample plates if you don't have local test data:
```bash
python scripts/download_sample_data.py --output /path/to/plates
```

---

## E2E Test Environment

Tests use `.env.e2etest` which points to the test server:
```
HOST=127.0.0.2
PORT=4064
USERNAME=root
PASSWORD=omero
TEST_DATABASE=true
```

Running e2e tests:
```bash
# Requires test server running and test data loaded
omero-integration-test e2e_connection      # connectivity check
omero-integration-test e2e_excel           # Excel metadata parsing
omero-integration-test e2e_pixelsize       # pixel size extraction
omero-integration-test e2e_flatfield_corr  # flatfield correction pipeline
omero-integration-test e2e_omero_screen    # full pipeline (takes several minutes)
```

---

## Connecting to OMERO Programmatically

All omero-screen code uses the `@omero_connect` decorator from `omero-utils`:

```python
from omero_utils.omero_connect import omero_connect
from omero.gateway import BlitzGateway

@omero_connect
def my_function(plate_id: int, conn: BlitzGateway | None = None) -> None:
    assert conn is not None
    plate = conn.getObject("Plate", plate_id)
    # conn is automatically closed on function exit
```

The decorator reads connection parameters from the active `.env` file.

### Direct connection (without decorator)
```python
import omero
from omero.gateway import BlitzGateway

conn = BlitzGateway("root", "omero", host="127.0.0.2", port=4064)
conn.connect()
try:
    plate = conn.getObject("Plate", 1234)
finally:
    conn.close()
```

---

## OMERO CLI (omero command)

```bash
# Login to test server
omero login root@127.0.0.2 -p 4064

# List plates
omero list plates

# Upload an image
omero import --target Dataset:name:MyDataset /path/to/image.tiff

# Attach a file to a plate
omero attach --target Plate:1234 /path/to/file.csv

# Check sessions
omero sessions list

# Logout
omero logout
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Connection refused 127.0.0.2:4064` | Test server not running — `./scripts/manage_test_server.sh start` |
| `Address already in use` | Production server occupying 4064 — stop it first with `docker-compose down` |
| Loopback alias missing | Run `sudo ifconfig lo0 alias 127.0.0.2` (macOS) |
| Test server takes too long to start | Wait up to 60 seconds; check `docker-compose.test.yml` logs with `docker-compose -f docker-compose.test.yml logs` |
| Test plates not found | Run `./scripts/load_plates.sh` to import test data |
