# Anima — Azure (Ubuntu VM + Docker) Deployment & Power Apps Integration Guide

How to take the Anima backend from "runs locally in Docker" to "reachable by your
Power Apps frontend in the cloud." The hosting model is deliberately simple and
matches what already works locally:

**One Ubuntu VM on Azure running your existing `docker-compose` stack** — the SQL
Server container, the one-shot seed, and the FastAPI container, unchanged — with a
small **Caddy** reverse-proxy container added in front to give it HTTPS (which Power
Apps custom connectors require).

Two halves:

1. **Stand up the VM and run the stack** (Parts 1–5).
2. **Connect Power Apps** with a custom connector and bind the analysis + XAI
   output (Parts 6–9).

> Two facts that shape the Power Apps side (verified current):
> - **Power Apps custom connectors accept OpenAPI 2.0 (Swagger) only — not 3.0.**
>   FastAPI emits 3.x, so there's a one-time conversion step (Part 7). The spec must
>   also be **under 1 MB**.
> - **Custom connectors require an HTTPS endpoint** — hence the Caddy step. Caddy
>   gets a free Let's Encrypt certificate automatically for your VM's Azure DNS name.

---

## Target architecture

```
Power Apps (canvas app)
      │  Custom Connector  (HTTPS)
      ▼
┌──────────────────── one Ubuntu VM ────────────────────┐
│  Caddy  ──(reverse_proxy)──►  api (FastAPI :8000)      │
│  :80/:443  auto-HTTPS                     │            │
│                              db (SQL Server 2022 :1433)│
│                              seed (one-shot, then exits)│
│  volumes: mssql_data, mri_images, caddy_data (on disk) │
└────────────────────────────────────────────────────────┘
```

Nothing about your app changes. `db`, `seed`, and `api` are your current compose
services; Caddy is the only addition, and it's the only thing exposed to the
internet (ports 80/443). SQL Server (1433) and the API (8000) stay on the VM's
private Docker network.

---

## Part 0 — Prerequisites

- An Azure subscription — **Azure for Students** gives free credit, no card:
  https://azure.microsoft.com/free/students/
- **Azure CLI** installed and logged in locally: `az login`.
- An SSH client (built into Windows/PowerShell).
- Your repo, plus the git-ignored **data** (the CSVs, and optionally the MRI
  folder) to copy up, and the values for your `.env`.

Pick names once:

```powershell
$RG   = "anima-rg"
$LOC  = "uksouth"
$VM   = "anima-vm"
$DNS  = "anima-$(Get-Random -Max 99999)"   # becomes <DNS>.uksouth.cloudapp.azure.com
az group create -n $RG -l $LOC
```

---

## Part 1 — Create the Ubuntu VM

SQL Server + PyTorch/BERT + the API want memory, so use a **B2ms (2 vCPU, 8 GB)**;
B2s (4 GB) is the bare minimum and can be tight.

```powershell
az vm create -g $RG -n $VM `
  --image Ubuntu2204 --size Standard_B2ms `
  --admin-username azureuser --generate-ssh-keys `
  --public-ip-address-dns-name $DNS
```

Open only the ports Caddy needs (plus SSH). Do **not** open 1433 or 8000 — they
stay private.

```powershell
az vm open-port -g $RG -n $VM --port 80  --priority 1001
az vm open-port -g $RG -n $VM --port 443 --priority 1002
```

Your public hostname is now `${DNS}.${LOC}.cloudapp.azure.com` — note it; Caddy
will get a TLS cert for exactly this name.

---

## Part 2 — Install Docker on the VM

```powershell
ssh azureuser@${DNS}.${LOC}.cloudapp.azure.com
```

Then on the VM:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER      # so you can run docker without sudo
exit                                # log out and back in for the group to apply
```

Reconnect, then confirm: `docker compose version` (the Compose plugin ships with
Docker now).

---

## Part 3 — Get the code, data, and `.env` onto the VM

From the VM, clone the repo (or `scp` your whole folder up instead):

```bash
git clone <your-repo-url> anima && cd anima
```

The CSVs and MRI scans are git-ignored, so copy them up from your **local** machine
(new terminal). The seed only needs the two CSVs; MRI images are optional and can
wait:

```powershell
$HOST = "${DNS}.${LOC}.cloudapp.azure.com"
scp .\data\*.csv azureuser@${HOST}:~/anima/data/
# optional, larger: scp -r .\data\mri azureuser@${HOST}:~/anima/data/
```

Create `.env` in `~/anima` on the VM (same variables your compose reads):

```bash
cat > .env <<'EOF'
MSSQL_SA_PASSWORD=<Strong_SA_Passw0rd!>
DB_NAME=anima
DB_USER=sa
DB_PASSWORD=<Strong_SA_Passw0rd!>
FERNET_KEY=<the SAME key you seeded with locally>
EOF
```

`DB_USER=sa` and `DB_PASSWORD` must equal `MSSQL_SA_PASSWORD` (the API connects as
`sa`). Use the **same `FERNET_KEY`** as your local seed so encrypted patient names
decrypt correctly. SQL Server rejects weak SA passwords — use 12+ chars with mixed
case, digits, and a symbol.

---

## Part 4 — Add HTTPS with Caddy

Create two files in `~/anima`. First a `Caddyfile` (swap in your real hostname):

```
anima-XXXXX.uksouth.cloudapp.azure.com {
    reverse_proxy api:8000
}
```

Then a compose overlay `docker-compose.prod.yml` that adds Caddy (and stops the API
from publishing 8000 to the host, so only Caddy is public):

```yaml
services:
  api:
    ports: []          # don't expose 8000 publicly; reach it only via Caddy
  caddy:
    image: caddy:2
    container_name: anima_caddy
    depends_on: [api]
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
      - caddy_config:/config
    networks:
      - anima_net
volumes:
  caddy_data:
  caddy_config:
```

Caddy automatically obtains and renews a Let's Encrypt certificate for that
hostname over ports 80/443 — no manual certs.

---

## Part 5 — Bring the stack up

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Order is handled for you: `db` starts and becomes healthy → `seed` runs
`init_db.sql` + `seed_dsm5.py` and exits → `api` starts → `caddy` fronts it. Watch
progress with `docker compose logs -f seed` (should finish and exit 0) and
`docker compose logs -f api`.

**Verify** from any browser:

- `https://<your-host>/health` → `{"status":"healthy","database":"anima"}`
- `https://<your-host>/docs` → the interactive API docs.

Notes:
- The **first** DSM-5 / combined call is slow while Bio_ClinicalBERT loads; it's
  fast after that. Hit `/api/analysis/dsm5/0010001` once to warm it.
- **MRI images** (optional): on the VM, `docker compose exec api python seed_mri.py
  --mri-dir /data/mri/<folder>` writes JPEGs into the `mri_images` volume. Until
  then MRI reports `pending` and the combined score runs on the DSM-5 channel.
- **Redeploy** after a code change: `git pull` then re-run the `up -d --build`
  command. The seed is idempotent (skips existing patients).

---

## Part 6 — (Recommended) Add a simple API key

The analysis endpoints trust a `requester_user_id` query parameter — fine for the
RBAC logic, but nothing stops a stranger who finds the URL. Before a public demo,
add a shared key the connector sends as a header. Minimal FastAPI dependency in
`main.py`:

```python
import os
from fastapi import Header, HTTPException, Depends
API_KEY = os.getenv("ANIMA_API_KEY")
def require_api_key(x_api_key: str = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")
# add to the protected routers, e.g.:
# app.include_router(combined_analysis_router, dependencies=[Depends(require_api_key)])
```

Add `ANIMA_API_KEY=<random-secret>` to `.env`, pass it through in the `api` service
`environment:` block, and the connector sends it as `X-API-Key` (Part 8).

---

## Part 7 — Get an OpenAPI 2.0 (Swagger) spec

Power Apps needs Swagger 2.0; FastAPI serves 3.x at `/openapi.json`. Convert it on
your local machine:

```powershell
npm install -g api-spec-converter
api-spec-converter --from=openapi_3 --to=swagger_2 --syntax=json `
  "https://<your-host>/openapi.json" > anima-swagger2.json
```

Open the file and check/trim:
- it starts with `"swagger": "2.0"`;
- add if missing: `"host": "<your-host>"`, `"basePath": "/"`, `"schemes": ["https"]`;
- delete endpoints you won't call so it stays **under 1 MB** (you mainly need
  `/api/auth/*`, the analysis endpoints, and maybe `/api/patients`).

> Alternative if the converted file is messy: in the connector wizard pick **Create
> from blank** and add the ~5 actions by hand (method, path, params, a sample JSON
> response). Often faster and more reliable than converting.

---

## Part 8 — Create the Power Apps custom connector

1. https://make.powerapps.com → **Data** → **Custom connectors** → **New custom
   connector** → **Import an OpenAPI file** → choose `anima-swagger2.json`.
2. **General:** Host = `<your-host>`, Base URL = `/`, scheme **HTTPS**.
3. **Security:** with the API key (Part 6) choose **API Key**, label `API Key`,
   parameter name `X-API-Key`, location **Header**. Otherwise **No authentication**.
4. **Definition:** confirm each action — for the analysis POSTs, `patient_id` is a
   **path** param and `requester_user_id` a **query** param (both required).
5. **Create connector** → **Test** tab → make a connection (paste the key) → test
   `POST /api/analysis/combined/0010001` with `requester_user_id=A000001`. You should
   get the JSON back including the `explanation` block.

---

## Part 9 — Bind it in the canvas app

1. **Data** → **Add data** → your custom connector.
2. **Auth flow:** call `login` then `verify-otp`; store the returned `user_ID`:
   `Set(gUser, AnimaAPI.login({email:..., password:...}).user_ID)`.
3. **Run analysis** on a button:
   `Set(gResult, AnimaAPI.AnalyzeCombined("0010001", {requester_user_id: gUser}))`
4. **Bind outputs:**
   - Risk labels → `gResult.final_combined_score`, `gResult.predicted_diagnosis`,
     `gResult.predicted_subtype`.
   - **Feature-importance bar chart** → `Items =
     gResult.explanation.text_model.feature_importance` (X `feature`, Y `impact_percent`).
   - **Influential-words gallery** → `Items =
     gResult.explanation.text_model.influential_words` (label `word`, value `push`).
   - **Brain heatmap** → an **Image** control with
     `Image = gResult.explanation.mri_model.heatmap_image` (base64 PNG data URI —
     Power Apps renders it directly). Guard with
     `If(gResult.explanation.mri_model.available, ...)`.

---

## Part 10 — Gotchas & housekeeping

- **HTTPS is mandatory** for connectors — that's why Caddy is there. If cert
  issuance fails, check that NSG ports 80 **and** 443 are open and you used the
  exact `cloudapp.azure.com` hostname in the `Caddyfile`.
- **OpenAPI 2.0 only, ≤ 1 MB, no OAuth client-credentials** for connectors.
- **MRI ZIP upload from Power Apps** is awkward (canvas apps don't do multipart
  cleanly) — prefer server-side seeding or a Blob-upload flow; the JSON endpoints
  bind nicely.
- **Cold start:** first DSM-5 call loads BERT (a few seconds) — warm it post-deploy.
- **Cost control:** a running VM burns credit. Between demos, **deallocate** it
  (`az vm deallocate -g $RG -n $VM`) and `az vm start` before you need it — deallocated
  VMs don't bill for compute. Tear everything down with `az group delete -n $RG`.
  The data survives stop/start on the VM's disk (the named volumes are on disk).
- **Secrets:** keep `.env` (SA password, `FERNET_KEY`, API key) off git — it's
  already git-ignored. It lives only on the VM.

---

## Sources

- Power Apps custom connector from an OpenAPI definition (2.0-only, 1 MB limit):
  https://learn.microsoft.com/en-us/connectors/custom-connectors/define-openapi-definition
- Create a custom connector from scratch (build-from-blank):
  https://learn.microsoft.com/en-us/connectors/custom-connectors/define-blank
- Create a Linux VM with the Azure CLI:
  https://learn.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-cli
- Install Docker Engine on Ubuntu: https://docs.docker.com/engine/install/ubuntu/
- Caddy automatic HTTPS: https://caddyserver.com/docs/automatic-https
