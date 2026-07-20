# Anima — Azure Deployment & Power Apps Integration Guide

How to take the Anima backend from "runs locally in Docker" to "reachable by your
Power Apps frontend in the cloud." It has two halves:

1. **Deploy the backend to Azure** so it has a public HTTPS URL.
2. **Connect Power Apps to it** with a custom connector, and bind the analysis +
   explanation (XAI) output to controls.

> Two facts that shape everything below (verified current):
> - **Power Apps custom connectors accept OpenAPI 2.0 (Swagger) only — not
>   OpenAPI 3.0.** FastAPI produces 3.x, so there is a one-time conversion step
>   (Part 3). The spec file must also be **under 1 MB**.
> - **Azure SQL Database has a free offer**, so you don't need to run SQL Server
>   in a container in the cloud — your `init_db.sql` runs on it as-is.

---

## Target architecture

```
Power Apps (canvas app)
      │  (Custom Connector, server-side HTTPS calls)
      ▼
Azure Web App for Containers  ──►  Azure SQL Database   (Patient/DSM5/MRI/Analysis_Result)
  (the FastAPI image)         ──►  Azure Files share    (mounted at /app/static/mri_images)
      ▲
Azure Container Registry (holds the anima-api image)
```

Local `docker-compose` maps to Azure like this: the **db** service → Azure SQL
Database; the **api** service → Web App for Containers; the MRI image volume →
an Azure Files share; the **seed** step → run once against the cloud DB.

---

## Part 0 — Prerequisites

- An Azure subscription. **Azure for Students** gives free credit with no card:
  https://azure.microsoft.com/free/students/
- **Azure CLI** installed and logged in: `az login`.
- The Anima repo on your machine (with the `Dockerfile`).
- A **Power Apps** environment on the same Microsoft 365 tenant
  (https://make.powerapps.com).
- Docker Desktop is *optional* — we build the image in the cloud with `az acr build`.

Pick names and a region once and reuse them:

```powershell
$RG      = "anima-rg"
$LOC     = "uksouth"
$ACR     = "animaacr$(Get-Random -Max 9999)"   # must be globally unique, lowercase
$SA      = "animastore$(Get-Random -Max 9999)" # storage account, globally unique
$SQLSRV  = "anima-sql-$(Get-Random -Max 9999)"
$APP     = "anima-api-$(Get-Random -Max 9999)"
az group create -n $RG -l $LOC
```

---

## Part 1 — Azure SQL Database (the DB)

1. **Create the server and a free-offer database:**

   ```powershell
   az sql server create -g $RG -n $SQLSRV -l $LOC `
     --admin-user animaadmin --admin-password "<StrongPassw0rd!>"

   az sql db create -g $RG -s $SQLSRV -n anima `
     --edition GeneralPurpose --compute-model Serverless `
     --family Gen5 --capacity 2 --use-free-limit true `
     --free-limit-exhaustion-behavior AutoPause
   ```

   (`--use-free-limit` is the free monthly allowance — see the free-offer doc in
   Sources. It auto-pauses when the free vCore-seconds run out, so it won't bill.)

2. **Open the firewall** to Azure services and to your machine (for seeding):

   ```powershell
   az sql server firewall-rule create -g $RG -s $SQLSRV -n AllowAzure `
     --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0
   az sql server firewall-rule create -g $RG -s $SQLSRV -n MyMachine `
     --start-ip-address <your.public.ip> --end-ip-address <your.public.ip>
   ```

3. **Create the schema.** Point `sqlcmd` (or Azure Data Studio) at the server and
   run your existing script — it is standard T-SQL and runs on Azure SQL unchanged:

   ```powershell
   sqlcmd -S "$SQLSRV.database.windows.net" -d anima -U animaadmin -P "<StrongPassw0rd!>" `
     -G -i ..\db\init_db.sql
   ```

4. **Seed the data** by running your existing seeder locally but pointed at the
   cloud DB. In your `.env` (or shell env) set:

   ```ini
   DB_HOST=<SQLSRV>.database.windows.net
   DB_PORT=1433
   DB_NAME=anima
   DB_USER=animaadmin
   DB_PASSWORD=<StrongPassw0rd!>
   ODBC_DRIVER=ODBC Driver 18 for SQL Server
   FERNET_KEY=<the SAME key you use locally>   # so encrypted names line up
   ```

   ```powershell
   python seed_dsm5.py      # patients + DSM-5 rows into Azure SQL
   ```

   (MRI image seeding is handled after deploy — see Part 4 note. The text &
   combined analysis work fine without it, thanks to the graceful MRI fallback.)

---

## Part 2 — Build & push the API image

1. **Create the container registry and build the image in the cloud** (no local
   Docker needed — `az acr build` uploads the context and builds server-side):

   ```powershell
   az acr create -g $RG -n $ACR --sku Basic --admin-enabled true
   az acr build -r $ACR -t anima-api:latest .    # run from the folder with the Dockerfile
   ```

   Your `Dockerfile` already bakes Bio_ClinicalBERT into the image and sets the
   offline flags, so the container starts without needing Hugging Face at runtime.

---

## Part 3 — Storage for MRI images

Processed MRI JPEGs must live on persistent storage, not the container's ephemeral
disk. Create an Azure Files share to mount at `/app/static/mri_images`:

```powershell
az storage account create -g $RG -n $SA -l $LOC --sku Standard_LRS
$SAKEY = az storage account keys list -g $RG -n $SA --query "[0].value" -o tsv
az storage share-rm create --storage-account $SA -n mri-images --quota 5
```

---

## Part 4 — Deploy the Web App for Containers

1. **Create an App Service plan (Linux) and the web app from your image:**

   ```powershell
   az appservice plan create -g $RG -n anima-plan --is-linux --sku B1
   az webapp create -g $RG -p anima-plan -n $APP `
     --deployment-container-image-name "$ACR.azurecr.io/anima-api:latest"
   ```

2. **Give the web app the registry credentials and the container port:**

   ```powershell
   $ACRPWD = az acr credential show -n $ACR --query "passwords[0].value" -o tsv
   az webapp config container set -g $RG -n $APP `
     --container-image-name "$ACR.azurecr.io/anima-api:latest" `
     --container-registry-url "https://$ACR.azurecr.io" `
     --container-registry-user $ACR --container-registry-password $ACRPWD
   az webapp config appsettings set -g $RG -n $APP --settings WEBSITES_PORT=8000
   ```

3. **Set the app's environment variables** (same names your `database.py` /
   `security.py` read):

   ```powershell
   az webapp config appsettings set -g $RG -n $APP --settings `
     DB_HOST="$SQLSRV.database.windows.net" DB_PORT=1433 DB_NAME=anima `
     DB_USER=animaadmin DB_PASSWORD="<StrongPassw0rd!>" `
     "ODBC_DRIVER=ODBC Driver 18 for SQL Server" `
     FERNET_KEY="<same key as seeding>" `
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 `
     ALLOWED_ORIGINS="*"
   ```

4. **Mount the Azure Files share** at the MRI images path (one clean command on
   App Service):

   ```powershell
   az webapp config storage-account add -g $RG -n $APP `
     --custom-id mri --storage-type AzureFiles `
     --account-name $SA --share-name mri-images --access-key $SAKEY `
     --mount-path /app/static/mri_images
   ```

5. **Verify.** Your URL is `https://$APP.azurewebsites.net`:

   - `https://<app>.azurewebsites.net/health` → `{"status":"healthy",...}`
   - `https://<app>.azurewebsites.net/docs` → the Swagger UI.

   > First call is slow — the clinical language model loads on the first DSM-5
   > request. Hitting `/api/analysis/dsm5/<id>` once after deploy "warms" it.

   **MRI image seeding (optional, after the app is up):** open an SSH session to
   the container (App Service → *SSH*) and run `python seed_mri.py --mri-dir <path>`
   there so the JPEGs land on the mounted share; or ingest per-patient through
   `POST /api/ingest/mri`. Until then, MRI simply reports `pending` and the combined
   score runs on the DSM-5 channel.

---

## Part 5 — Prepare the API for a custom connector

### 5.1 (Recommended) Add a simple API key

Right now the analysis endpoints trust a `requester_user_id` query parameter — fine
for the RBAC logic, but there's no secret stopping a stranger who finds the URL from
calling it. For a cloud deployment, add a shared API key that the connector sends as
a header. A minimal FastAPI dependency (in `main.py`) does it:

```python
import os
from fastapi import Header, HTTPException
API_KEY = os.getenv("ANIMA_API_KEY")
def require_api_key(x_api_key: str = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")
# then: app.include_router(<router>, dependencies=[Depends(require_api_key)])
```

Set `ANIMA_API_KEY` as an app setting (Part 4 step 3). The connector will send it as
`X-API-Key` (Part 6). This is optional but strongly recommended before a public demo.

### 5.2 Get an OpenAPI **2.0** spec (the conversion step)

Power Apps needs Swagger 2.0; FastAPI serves 3.x at `/openapi.json`. Convert it:

```powershell
npm install -g api-spec-converter
api-spec-converter --from=openapi_3 --to=swagger_2 --syntax=json `
  "https://$APP.azurewebsites.net/openapi.json" > anima-swagger2.json
```

Then open `anima-swagger2.json` and check/trim:

- it starts with `"swagger": "2.0"`;
- add if missing: `"host": "<app>.azurewebsites.net"`, `"basePath": "/"`,
  `"schemes": ["https"]`;
- delete endpoints the app doesn't call, to stay **under 1 MB** and keep the
  connector tidy (you mainly need: `/api/auth/*`, the analysis endpoints, and
  maybe `/api/patients`).

> Alternative if the converted file is messy: in the connector wizard choose
> **"Create from blank"** and add just the handful of actions by hand (method, path,
> query params, and a sample JSON response so Power Apps learns the fields). For ~5
> endpoints this is often faster and more reliable than converting.

---

## Part 6 — Create the Power Apps custom connector

1. Go to https://make.powerapps.com → **Data** → **Custom connectors** →
   **New custom connector** → **Import an OpenAPI file** → pick `anima-swagger2.json`.
2. **General:** confirm Host = `<app>.azurewebsites.net`, Base URL = `/`, scheme HTTPS.
3. **Security:** if you added the API key (5.1), choose **API Key**, Parameter label
   `API Key`, Parameter name `X-API-Key`, Location **Header**. Otherwise **No
   authentication**.
4. **Definition:** review the actions (each analysis endpoint). For the POST
   analysis calls, `patient_id` is a **path** parameter and `requester_user_id` a
   **query** parameter — make sure they're marked required.
5. **Create connector**, then open the **Test** tab, create a connection (paste the
   API key if used), and test e.g. `POST /api/analysis/combined/0010001` with
   `requester_user_id=A000001`. You should get the JSON back including the
   `explanation` block.

---

## Part 7 — Bind it in the canvas app

1. In your app: **Data** → **Add data** → your custom connector.
2. **Auth flow:** call `login` then `verify-otp` (your API returns the OTP for the
   client to email); store the returned `user_ID` in a variable, e.g.
   `Set(gUser, AnimaAPI.login({email:..., password:...}).user_ID)`.
3. **Run an analysis** on a button:

   ```
   Set(gResult, AnimaAPI.AnalyzeCombined("0010001", {requester_user_id: gUser}))
   ```

4. **Bind the outputs:**
   - **Risk gauge / label** → `gResult.final_combined_score`,
     `gResult.predicted_diagnosis`, `gResult.predicted_subtype`.
   - **Feature-importance bar chart** → set the chart/gallery `Items` to
     `gResult.explanation.text_model.feature_importance`
     (X = `feature`, Y = `impact_percent`).
   - **Influential words gallery** → `Items =
     gResult.explanation.text_model.influential_words` (label `word`, value `push`).
   - **Brain heatmap** → an **Image** control with
     `Image = gResult.explanation.mri_model.heatmap_image`
     (it's a base64 PNG data URI — Power Apps renders it directly).
   - Guard for the pending case: show the heatmap only
     `If(gResult.explanation.mri_model.available, ...)`.

---

## Part 8 — Known limitations & gotchas

- **OpenAPI 2.0 only, ≤ 1 MB, no OAuth client-credentials** for custom connectors
  (Sources). Convert or build-from-blank; trim the spec.
- **MRI ZIP upload from Power Apps** is awkward — canvas apps don't do multipart
  file POST cleanly. For clinician uploads, prefer uploading the ZIP to Azure Blob
  Storage and triggering ingestion, or do bulk seeding server-side; the JSON
  analysis endpoints are what bind nicely to the UI.
- **Cold start:** the first DSM-5 call loads Bio_ClinicalBERT (a few seconds). Warm
  it after each deploy, or add a startup warmup call.
- **CORS** isn't the blocker for connectors — connector calls are made server-side
  by the Power Platform, not from the browser — but leaving `ALLOWED_ORIGINS=*` is
  fine for the demo. Tighten it if you also call the API from a browser page.
- **Cost control:** the SQL free offer auto-pauses; the B1 App Service plan is not
  free. To avoid burning student credit between demos, `az webapp stop -g $RG -n
  $APP` (and start it before a demo), or delete the resource group when done:
  `az group delete -n $RG`.
- **Secrets:** don't commit `FERNET_KEY`, DB password, or the API key. They live as
  app settings in Azure and in your local `.env` (which is git-ignored).

---

## Sources

- Power Apps custom connector from an OpenAPI definition (2.0-only requirement, 1 MB
  limit): https://learn.microsoft.com/en-us/connectors/custom-connectors/define-openapi-definition
- Create a custom connector from scratch (build-from-blank):
  https://learn.microsoft.com/en-us/connectors/custom-connectors/define-blank
- Run a custom container on Azure App Service:
  https://learn.microsoft.com/en-us/azure/app-service/quickstart-custom-container
- Azure SQL Database free offer:
  https://learn.microsoft.com/en-us/azure/azure-sql/database/free-offer
- Azure Container Apps (alternative host) quickstart:
  https://learn.microsoft.com/en-us/azure/container-apps/quickstart-code-to-cloud
