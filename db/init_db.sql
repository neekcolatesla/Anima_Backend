/* ===========================================================================
   Anima - foundational database schema (Microsoft SQL Server / T-SQL)

   Builds the Anima relational schema and seeds the initial system
   administrator. Idempotent: creates each object only if absent and inserts
   the seed admin only if missing, so it is safe to re-run.

   Run with sqlcmd:
       sqlcmd -S localhost,1433 -U sa -P "<SA_PASSWORD>" -C -i init_db.sql
   or paste into SSMS / Azure Data Studio and execute.

   NOTE: the authentication supertype table is kept plural as Users, because
   the singular "User" is a reserved keyword in T-SQL.

   ---------------------------------------------------------------------------
   Supertype/Subtype: Users is the sole authentication supertype. The role
   tables (Admin, Clinician, Guardian, Patient) each carry user_ID as a
   FOREIGN KEY back to Users - role IDs are never stored inside Users.

   ID convention (enforced by CHECK constraints):
       user_ID = <role letter> + <numeric role-table ID>
       Admin      user_ID = 'A' + admin_ID       ('000001'  -> 'A000001')
       Clinician  user_ID = 'C' + clinician_ID
       Guardian   user_ID = 'G' + guardian_ID
       Patient    user_ID = 'P' + patient_ID     ('0010001' -> 'P0010001')

   Relationships (from the project ERD):
       Users 1 -- 0..1 Admin / Clinician / Guardian / Patient / User_OTP
       Guardian 0..1 -- 0..* Patient            (guardian_ID FK on Patient)
       Patient  1    -- 0..* DSM5_Assessment
       Patient  1    -- 0..* MRI
       Clinician 1   -- 0..* Clinician_Patient_Assignment
       Patient   1   -- 0..* Clinician_Patient_Assignment   (many-to-many)
       Admin 1 -- 0..* RBAC_Controls
=========================================================================== */

/* ---------------------------------------------------------------------------
   1. Create the database if it does not already exist
   (CREATE DATABASE cannot run inside a transaction, so it stands alone.)
--------------------------------------------------------------------------- */
IF DB_ID('anima') IS NULL
    CREATE DATABASE anima;
GO

USE anima;
GO

/* ---------------------------------------------------------------------------
   2. Users - authentication SUPERTYPE  (kept plural; "User" is reserved in T-SQL)
      CK_Users_id_role_prefix: first letter of user_ID must match the role.
--------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.Users', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.Users (
        user_ID        VARCHAR(10)   NOT NULL PRIMARY KEY,
        email          VARCHAR(500)  NOT NULL UNIQUE,
        password_hash  VARCHAR(255)  NOT NULL,
        role           VARCHAR(20)   NOT NULL
            CONSTRAINT CK_Users_role
            CHECK (role IN ('Admin', 'Clinician', 'Guardian', 'Patient')),
        CONSTRAINT CK_Users_id_role_prefix CHECK (
               (role = 'Admin'     AND user_ID LIKE 'A%')
            OR (role = 'Clinician' AND user_ID LIKE 'C%')
            OR (role = 'Guardian'  AND user_ID LIKE 'G%')
            OR (role = 'Patient'   AND user_ID LIKE 'P%')
        )
    );
END
GO

/* ---------------------------------------------------------------------------
   3. Admin - SUBTYPE   (user_ID = 'A' + admin_ID)
--------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.Admin', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.Admin (
        admin_ID  VARCHAR(7)    NOT NULL PRIMARY KEY,
        user_ID   VARCHAR(10)   NOT NULL UNIQUE,
        name      VARCHAR(500)  NOT NULL,
        CONSTRAINT FK_Admin_Users
            FOREIGN KEY (user_ID) REFERENCES dbo.Users(user_ID),
        CONSTRAINT CK_Admin_user_id_format
            CHECK (user_ID = 'A' + admin_ID)
    );
END
GO

/* ---------------------------------------------------------------------------
   4. Clinician - SUBTYPE   (user_ID = 'C' + clinician_ID)
--------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.Clinician', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.Clinician (
        clinician_ID  VARCHAR(7)    NOT NULL PRIMARY KEY,
        user_ID       VARCHAR(10)   NOT NULL UNIQUE,
        name          VARCHAR(500)  NOT NULL,
        is_verified   BIT           NOT NULL DEFAULT 0,
        CONSTRAINT FK_Clinician_Users
            FOREIGN KEY (user_ID) REFERENCES dbo.Users(user_ID),
        CONSTRAINT CK_Clinician_user_id_format
            CHECK (user_ID = 'C' + clinician_ID)
    );
END
GO

/* ---------------------------------------------------------------------------
   5. Guardian - SUBTYPE   (user_ID = 'G' + guardian_ID)
      Created before Patient (Patient.guardian_ID references this table).
--------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.Guardian', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.Guardian (
        guardian_ID  VARCHAR(7)    NOT NULL PRIMARY KEY,
        user_ID      VARCHAR(10)   NOT NULL UNIQUE,
        name         VARCHAR(500)  NOT NULL,
        CONSTRAINT FK_Guardian_Users
            FOREIGN KEY (user_ID) REFERENCES dbo.Users(user_ID),
        CONSTRAINT CK_Guardian_user_id_format
            CHECK (user_ID = 'G' + guardian_ID)
    );
END
GO

/* ---------------------------------------------------------------------------
   6. Patient - SUBTYPE
      user_ID is NULLABLE: child patients (is_child = 1) cannot log in.
      guardian_ID (NULLABLE FK) implements "a guardian manages 0..* patients".
--------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.Patient', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.Patient (
        patient_ID      VARCHAR(15)   NOT NULL PRIMARY KEY,
        user_ID         VARCHAR(10)   NULL UNIQUE,
        guardian_ID     VARCHAR(7)    NULL,
        name            VARCHAR(500)  NULL,
        biological_sex  INT           NULL,
        is_child        BIT           NOT NULL DEFAULT 0,
        age             DECIMAL(4,2)  NULL,
        CONSTRAINT FK_Patient_Users
            FOREIGN KEY (user_ID) REFERENCES dbo.Users(user_ID),
        CONSTRAINT FK_Patient_Guardian
            FOREIGN KEY (guardian_ID) REFERENCES dbo.Guardian(guardian_ID),
        CONSTRAINT CK_Patient_user_id_format
            CHECK (user_ID IS NULL OR user_ID = 'P' + patient_ID)
    );
END
GO

/* ---------------------------------------------------------------------------
   7. User_OTP - one-time passcodes
      (surrogate otp_ID PK added; the ERD lists columns but no key)
--------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.User_OTP', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.User_OTP (
        otp_ID      INT           IDENTITY(1,1) NOT NULL PRIMARY KEY,
        user_ID     VARCHAR(10)   NOT NULL,
        otp_code    VARCHAR(10)   NOT NULL,
        expires_at  DATETIME2     NOT NULL,
        CONSTRAINT FK_UserOTP_Users
            FOREIGN KEY (user_ID) REFERENCES dbo.Users(user_ID)
    );
END
GO

/* ---------------------------------------------------------------------------
   8. DSM5_Assessment - questionnaire, phenotypic + scoring per patient
--------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.DSM5_Assessment', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.DSM5_Assessment (
        assessment_ID        INT           IDENTITY(1,1) NOT NULL PRIMARY KEY,
        patient_ID           VARCHAR(15)   NOT NULL,
        ground_truth_dx      INT           NULL,
        adhd_index           INT           NULL,
        inattentive_score    INT           NULL,
        hyperactive_score    INT           NULL,
        iq_measure           INT           NULL,
        med_status           INT           NULL,
        nlp_risk_score       DECIMAL(5,2)  NULL,
        final_combined_score DECIMAL(5,2)  NULL,
        raw_answers          NVARCHAR(MAX) NULL,   -- questionnaire JSON payload
        clinician_notes      NVARCHAR(MAX) NULL,   -- free-text NLP narrative (self/guardian/clinician)
        CONSTRAINT FK_DSM5_Patient
            FOREIGN KEY (patient_ID) REFERENCES dbo.Patient(patient_ID),
        CONSTRAINT CK_DSM5_raw_answers_json
            CHECK (raw_answers IS NULL OR ISJSON(raw_answers) = 1)
    );
END
GO

/* ---------------------------------------------------------------------------
   9. MRI - processed 2D slice paths + QC + image risk score per patient
--------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.MRI', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.MRI (
        MRI_ID          INT           IDENTITY(1,1) NOT NULL PRIMARY KEY,
        patient_ID      VARCHAR(15)   NOT NULL,
        scan_type       VARCHAR(10)   NOT NULL
            CONSTRAINT CK_MRI_scan_type CHECK (scan_type IN ('anat', 'anat_gm')),
        file_path       VARCHAR(500)  NOT NULL,   -- directory of the slice stack
        slice_count     INT           NULL,       -- number of JPEG slices in folder
        qc_anatomical_1 VARCHAR(10)   NULL,
        qc_anatomical_2 VARCHAR(10)   NULL,
        mri_risk_score  DECIMAL(5,2)  NULL,
        CONSTRAINT FK_MRI_Patient
            FOREIGN KEY (patient_ID) REFERENCES dbo.Patient(patient_ID),
        CONSTRAINT UQ_MRI_patient_scan UNIQUE (patient_ID, scan_type)
    );
END
GO

/* ---------------------------------------------------------------------------
   10. Clinician_Patient_Assignment - many-to-many junction
       Composite PK (clinician_ID, patient_ID).
--------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.Clinician_Patient_Assignment', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.Clinician_Patient_Assignment (
        clinician_ID  VARCHAR(7)   NOT NULL,
        patient_ID    VARCHAR(15)  NOT NULL,
        assigned_at   DATETIME2    NOT NULL DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_CPA PRIMARY KEY (clinician_ID, patient_ID),
        CONSTRAINT FK_CPA_Clinician
            FOREIGN KEY (clinician_ID) REFERENCES dbo.Clinician(clinician_ID),
        CONSTRAINT FK_CPA_Patient
            FOREIGN KEY (patient_ID) REFERENCES dbo.Patient(patient_ID)
    );
END
GO

/* ---------------------------------------------------------------------------
   11. RBAC_Controls - JSON access rules, keyed by role_name,
       tracked by updated_by_admin (FK -> Admin).
--------------------------------------------------------------------------- */
IF OBJECT_ID('dbo.RBAC_Controls', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.RBAC_Controls (
        role_name           VARCHAR(20)   NOT NULL PRIMARY KEY
            CONSTRAINT CK_RBAC_role
            CHECK (role_name IN ('Admin', 'Clinician', 'Guardian', 'Patient')),
        permissions_config  NVARCHAR(MAX) NOT NULL,   -- JSON payload
        updated_by_admin    VARCHAR(7)    NULL,
        CONSTRAINT FK_RBAC_Admin
            FOREIGN KEY (updated_by_admin) REFERENCES dbo.Admin(admin_ID),
        CONSTRAINT CK_RBAC_config_json
            CHECK (ISJSON(permissions_config) = 1)
    );
END
GO

/* ===========================================================================
   12. Seed the initial system administrator
       user_ID = 'A' + admin_ID  =>  'A000001'
       password = 'password'  (bcrypt hash below; NEVER store plaintext)

   NOTE: the hash is a pre-computed bcrypt digest of the string 'password',
   produced with the same algorithm as app/security.py (bcrypt, cost 12).
   bcrypt cannot be computed in T-SQL, so it is embedded as a literal. Rotate
   this immediately after first login.
=========================================================================== */
IF NOT EXISTS (SELECT 1 FROM dbo.Admin WHERE admin_ID = '000001')
BEGIN
    INSERT INTO dbo.Users (user_ID, email, password_hash, role)
    VALUES (
        'A000001',
        'thapaneharikawork@gmail.com',
        '$2b$12$E7Xg9OWZjZY0CFpVSA60TuCJ5SK0zrdLFcID5WJ3hwxNAabn0DL6C',
        'Admin'
    );

    INSERT INTO dbo.Admin (admin_ID, user_ID, name)
    VALUES ('000001', 'A000001', 'Neeharika Thapa');
END
GO