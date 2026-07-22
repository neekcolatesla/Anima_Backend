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
$LOC  = "swedencentral"
$VM   = "anima-vm"
$DNS  = "anima-$(Get-Random -Max 99999)"   # becomes <DNS>.swedencentral.cloudapp.azure.com
az group create -n $RG -l $LOC
```

> **Azure-for-Students region lock (learned the hard way on the first run).** This
> subscription enforces a policy that only permits a fixed set of regions.
> `uksouth`, `ukwest`, and `westeurope` are all **rejected at deploy time** with
> `RequestDisallowedByAzure` — the resource group is created but the VM deployment
> fails. Check your own allow-list *before* choosing `$LOC`:
>
> ```powershell
> az policy assignment list --query "[].{Name:displayName, Parameters:parameters}" -o json
> ```
>
> On this subscription the allowed regions were **germanywestcentral, swedencentral,
> italynorth, switzerlandnorth, norwayeast**; we used `swedencentral`. The DNS suffix
> follows the region you pick (`<DNS>.<region>.cloudapp.azure.com`). If you change the
> resource group's region you must `az group delete -n $RG --yes` first — a group
> can't be re-created in a new location while it still exists.

---

## Part 1 — Create the Ubuntu VM

SQL Server + PyTorch/BERT + the API want memory, so aim for **2 vCPU / 8 GB**. We
deployed on a **Standard_D2s_v3** (2 vCPU, 8 GB) and also enlarged the OS disk to
**64 GB** up front — see the two callouts below for why both of those matter.

```powershell
az vm create -g $RG -n $VM `
  --image Ubuntu2204 --size Standard_D2s_v3 `
  --os-disk-size-gb 64 `
  --admin-username azureuser --generate-ssh-keys `
  --public-ip-address-dns-name $DNS
```

> **VM size can be capacity-blocked (`SkuNotAvailable`), which is *not* the same as
> the region policy above.** On the first run the cheap B-series was unavailable in
> every allowed region we tried — `Standard_B2ms`, `Standard_B1s`, `Standard_B2s`,
> and `Standard_A2_v2` all returned `SkuNotAvailable` (capacity restrictions), region
> by region. `Standard_D2s_v3` in `swedencentral` was the first that provisioned. If
> you get `SkuNotAvailable`, either try another allowed region or step up to a
> D-series; you can list what's actually available with
> `az vm list-skus -l $LOC --size Standard_D --all -o table`. D2s_v3 is a good fit
> anyway — more headroom for the torch + BERT image than a B-series.

> **The default 30 GB OS disk is too small — it caused the deploy to fail.** The API
> image bundles PyTorch and the pre-downloaded Bio_ClinicalBERT weights, so extracting
> it needs well over 30 GB. On the first attempt the build finished but the image
> *export* died with `no space left on device` (on the Bio_ClinicalBERT blob), even
> after `docker system prune`. Creating the VM with `--os-disk-size-gb 64` avoids this.
> If you already created a 30 GB VM, resize it after the fact — see Appendix A.

Open only the ports Caddy needs (plus SSH). Do **not** open 1433 or 8000 — they
stay private.

```powershell
az vm open-port -g $RG -n $VM --port 80  --priority 1001
az vm open-port -g $RG -n $VM --port 443 --priority 1002
```

> **If `az vm open-port` fails with `ResourceNotFound` for `<vm>NSG`/`<vm>VMNic`,**
> the auto-created network security group wasn't named the way the shortcut expects
> (this happened to us). Find the real NSG name and add the rules directly — pick
> priorities that don't collide with any existing rule:
>
> ```powershell
> $REAL_NSG = az network nsg list -g $RG --query "[0].name" -o tsv
> az network nsg rule create -g $RG --nsg-name $REAL_NSG --name AllowHTTP  --priority 1100 --destination-port-ranges 80  --protocol Tcp --access Allow
> az network nsg rule create -g $RG --nsg-name $REAL_NSG --name AllowHTTPS --priority 1110 --destination-port-ranges 443 --protocol Tcp --access Allow
> ```

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
# NB: don't use $HOST — it's a read-only built-in in PowerShell and the assignment
# silently fails (scp then tries to resolve a bogus hostname). Use $FQDN.
$FQDN = "${DNS}.${LOC}.cloudapp.azure.com"
scp .\data\*.csv azureuser@${FQDN}:~/anima/data/
# optional, larger: scp -r .\data\mri azureuser@${FQDN}:~/anima/data/
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

> **No spaces around the `=` in `.env`.** Write `MSSQL_SA_PASSWORD=Th!s1s...`, not
> `MSSQL_SA_PASSWORD= Th!s1s...`. Compose does not strip a leading space, so the space
> becomes part of the password. It still *works* as long as `MSSQL_SA_PASSWORD` and
> `DB_PASSWORD` carry the identical space (they stay consistent), but it will bite you
> the moment you type the password by hand — e.g. connecting with `sqlcmd`. Cleanest
> to avoid it entirely.

---

## Part 4 — Add HTTPS with Caddy

Create two files in `~/anima`. First a `Caddyfile` (swap in your real hostname):

```
anima-XXXXX.swedencentral.cloudapp.azure.com {
    reverse_proxy api:8000
}
```

> Use your **exact** VM hostname here (region included) — Caddy requests the TLS cert
> for this literal name, so a mismatch means no certificate. And put it in the
> `Caddyfile`; typing it at the shell prompt (easy slip) just gives "command not found".

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

## Part 5.5 — Verify the API is up and running

Work from the outside in: containers → API process → database → public HTTPS. There
are **two** endpoints and they check different things:

- `GET /`  → `{"service":"Anima API","status":"online",...}` — the process is alive
  (a *liveness* check; does **not** touch SQL).
- `GET /health` → `{"status":"healthy","database":"anima"}` — the API can run a live
  `SELECT 1` against SQL Server (a *readiness* check; returns **503
  `Database unavailable`** if the DB is down). This is the one that proves the whole
  stack is wired together.

**On the VM** (over SSH, in `~/anima`):

```bash
# 1. All four services in the expected state: db healthy, seed exited 0, api + caddy up
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps -a

# 2. The seed finished cleanly (this is what populated the DB)
docker compose logs seed | tail -20            # look for a success line, then "exited with code 0"

# 3. The API booted and connected to SQL
docker compose logs api | tail -30             # expect uvicorn "Application startup complete"

# 4. Hit the API directly, bypassing Caddy/TLS. 8000 isn't published to the host
#    (prod overlay sets ports: []), so curl from *inside* the api container:
docker compose exec api curl -s http://localhost:8000/          # -> {"status":"online",...}
docker compose exec api curl -s http://localhost:8000/health    # -> {"status":"healthy","database":"anima"}
```

If step 4's `/health` returns `online` on `/` but 503 on `/health`, the API is up but
can't reach SQL — check the `.env` password (including the no-spaces gotcha in Part 3)
and that `db` is healthy (`docker compose logs db | tail`).

**Through Caddy + HTTPS** (from your laptop, or anywhere) — this is the path Power
Apps will actually use:

```bash
# -k tolerates a not-yet-issued cert; drop it once TLS is live
curl -sk https://<your-host>/health         # -> {"status":"healthy","database":"anima"}
curl -sI https://<your-host>/docs           # -> HTTP/2 200
```

```powershell
# PowerShell equivalents — NB: in PowerShell `curl` is an ALIAS for Invoke-WebRequest,
# not real curl, so Unix flags like -s fail ("A drive with the name 'https' does not
# exist"). Use the native cmdlet, or call real curl explicitly as `curl.exe`.
Invoke-RestMethod  https://<your-host>/health
(Invoke-WebRequest https://<your-host>/docs -UseBasicParsing).StatusCode   # 200
curl.exe -s https://<your-host>/health                                     # real curl on Win10/11
```

**Confirm the TLS certificate actually issued** (Caddy needs 80 **and** 443 open and
the exact `cloudapp.azure.com` hostname in the `Caddyfile`):

```bash
docker compose logs caddy | grep -iE "certificate obtained|serving initial|error"
# from your laptop, inspect the live cert:
curl -vI https://<your-host>/health 2>&1 | grep -iE "SSL connection|subject|issuer"
```

A green `curl -s https://<your-host>/health` (no `-k` needed) returning
`"database":"anima"` means: process up, TLS valid, and SQL reachable — the stack is
fully live. Finally, **warm the model** so the first real Power Apps call isn't slow:

```bash
curl -s -X POST "https://<your-host>/api/analysis/dsm5/0010001"   # first call loads BERT (~seconds)
```

---

## Part 6 — API key on the feature routers (implemented)

The analysis endpoints trust a `requester_user_id` query parameter — fine for the
RBAC logic, but nothing stops a stranger who finds the URL. `main.py` therefore
puts **every feature router** behind an optional shared key that the Power Apps
connector sends as an `X-API-Key` header. It's already wired in:

```python
# main.py
API_KEY = os.getenv("ANIMA_API_KEY")

def require_api_key(x_api_key: str = Header(default=None, alias="X-API-Key")) -> None:
    """Reject callers missing the shared key - but only when a key is configured."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

_protected = [Depends(require_api_key)]
app.include_router(auth_router,     dependencies=_protected)
app.include_router(patients_router, dependencies=_protected)
# ... through combined_analysis_router — all feature routers gated
```

Three properties worth knowing:

- **Off by default.** With `ANIMA_API_KEY` unset the dependency is a no-op, so local
  dev, the seed, and your current deployment keep working unchanged — turning it on
  is opt-in.
- **`/`, `/health`, `/docs`, and `/openapi.json` stay open**, so the uptime checks
  (Part 5.5) and connector-spec generation (Part 7) keep working without the key.
- The header is advertised in the OpenAPI spec, so the connector wizard detects it.

`docker-compose.yml` already forwards the variable to the api container:

```yaml
    environment:
      ...
      ANIMA_API_KEY: ${ANIMA_API_KEY:-}   # blank = gate disabled
```

**To turn it on**, add a strong random value to `.env` on the VM and recreate the api:

```bash
# on the VM, in ~/anima
echo "ANIMA_API_KEY=$(openssl rand -hex 24)" >> .env       # or paste your own secret
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --force-recreate api
```

After that, unauthenticated calls to the feature endpoints return **401** while
`/health` still answers `healthy`. Quick check:

```bash
curl.exe -s -o /dev/null -w "%{http_code}\n" https://<your-host>/api/analysis/combined/0010001   # 401
curl.exe -s -H "X-API-Key: <your-secret>" https://<your-host>/api/analysis/combined/0010001?requester_user_id=A000001
```

The connector sends this value as `X-API-Key` (Part 8).

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

## Appendix A — First real deployment run (what actually happened & the fixes)

The main guide above has been corrected to reflect what we learned; this appendix is
the chronological account so the reasoning is preserved. Target: one Ubuntu 22.04 VM,
Docker stack, Sweden Central. Final result: **stack live** — `anima_sqlserver`
healthy, `anima_seed` exited 0, `anima_api` and `anima_caddy` running.

**1. Region rejections (`RequestDisallowedByAzure`).** `uksouth`, then `ukwest`, then
`westeurope` all failed at VM-deploy time — the resource group created fine but every
child resource (VNET/NSG/PublicIP/NIC/VM) was disallowed. Root cause: an
Azure-for-Students policy, "Allowed resource deployment regions." Diagnosed with
`az policy assignment list ...`, which returned the allow-list: germanywestcentral,
swedencentral, italynorth, switzerlandnorth, norwayeast. → **Fix:** deploy only into
an allowed region (we chose swedencentral). *Also note:* a resource group can't be
recreated in a new region while it exists, so each region change needed
`az group delete -n anima-rg --yes` first.

**2. VM size not available (`SkuNotAvailable`).** A separate wall. Within the allowed
regions, `Standard_B2ms`, `Standard_B1s`, `Standard_B2s`, and `Standard_A2_v2` were
all capacity-restricted (tried across Sweden Central, Germany West Central, Italy
North, Norway East, Switzerland North). → **Fix:** `Standard_D2s_v3` in swedencentral
provisioned first try. FQDN `anima-2526.swedencentral.cloudapp.azure.com`, public IP
`4.223.102.132`.

**3. `az vm open-port` failed (`ResourceNotFound`).** The shortcut looked for
`anima-vmNSG`/`anima-vmVMNic` and didn't find them under those names. → **Fix:**
resolved the real NSG via `az network nsg list -g $RG --query "[0].name" -o tsv`, then
created `AllowHTTP` (priority 1100) and `AllowHTTPS` (1110) directly. (Priority 1001
collided with a pre-existing `open-port-80` rule, hence 1100/1110.)

**4. Docker install + code.** `curl -fsSL https://get.docker.com | sudo sh`,
`usermod -aG docker $USER`, re-login. Docker 29.6.2, Compose v5.3.1. Cloned the repo,
created `.env`, uploaded data by `scp` from the laptop (all CSVs + the full MRI set).

**5. Build failed: `no space left on device`.** The `docker compose up --build`
built the images but died while *exporting/unpacking* the API image — specifically on
the Bio_ClinicalBERT blob — because the default **30 GB** OS disk isn't enough for the
PyTorch + BERT image. `docker system prune -a --volumes` reclaimed ~13 GB but the
retry hit the same wall. → **Fix:** grew the OS disk to **64 GB**:

```powershell
az vm deallocate -g $RG -n $VM
$DISK = az vm show -g $RG -n $VM --query "storageProfile.osDisk.name" -o tsv
az disk update -g $RG -n $DISK --size-gb 64
az vm start -g $RG -n $VM
```

(For a fresh VM, create it with `--os-disk-size-gb 64` from the start — now baked into
Part 1 — to skip this entirely.) After the resize, `docker compose ... up -d --build`
completed and all containers came up.

**Two harmless detours** worth noting so they aren't mistaken for problems: `pytest`
"command not found" on the VM (the test deps aren't installed in the deployment image
— tests run in your dev env, not on the server), and `az: command not found` inside
the SSH session (the Azure CLI runs on your laptop, not the VM). Also, in PowerShell,
`$HOST` is read-only — use a different variable name (Part 3).

**Net changes folded into this guide:** default region → an allowed one; VM size →
`Standard_D2s_v3`; OS disk → `--os-disk-size-gb 64`; NSG-rule fallback for open-port;
`.env` no-spaces note; `$FQDN` instead of `$HOST`; and the new Part 5.5 health checks.

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
