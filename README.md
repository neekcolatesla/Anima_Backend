Project Overview

Building the backend of the Anima website hosted on MS PowerApps with Claude Code. The Anima project is designed to give clinicians a platform for assessing Attention-Deficit/Hyperactivity Disorder (ADHD). By combining patient demographics with complex medical imaging, the goal is to automatically calculate an ADHD risk score and classify specific ADHD subtypes, ultimately helping medical professionals make faster and more accurate diagnoses.

To make this happen, the system architecture is split between an accessible frontend and a heavy-duty backend. For the frontend, the UI is designed in Figma and brought to life using Microsoft PowerApps, which includes an Outlook-MS PowerApps extension to fit naturally into a clinician's daily workflow. The backend handles the heavy lifting using Python and FastAPI for routing and data processing, Microsoft SQL Server for efficient relational database management, and Docker to containerise the entire environment.

The data ingestion pipeline takes in two distinct types of files. First, it processes CSV files containing NYU Athena Pre-processed Phenotypic data (from the ADHD200 consortium study), which includes essential demographic and medical details like age, medication status, and IQ. Second, it ingests .zip 2D MRI scans for each patient, specifically targeting the enclosed anat.nii.gz and anat_gm.nii.gz files. During this processing phase, the CSV data is parsed and mapped directly to the database schemas. Meanwhile, the MRI files are extracted, and the medical imaging data is normalized to a 0-255 scale and converted into readable JPEG formats.

All of this processed data is securely stored across two locations: the converted 2D MRI image slices are saved to a local container in the Docker environment, while the structured patient demographics and image file paths are housed in the Microsoft SQL Server database.

Once the data is stored, clinicians can utilize it to run deep analyses through the FastAPI backend, which powers two distinct machine learning models:
1. A text and demographic model that evaluates the patient's DSM-5 questionnaire answers, NLP data, and phenotypic information. 
2. An image classification model that analyses the visual data extracted from the anat_gm.nii and anat.nii MRI scans.

FastAPI processes the outputs from both of these models in tandem to calculate a comprehensive ADHD risk score and pinpoint the exact ADHD subtype for the clinician to review. 

To keep this sensitive medical data secure, the platform uses strict Role-Based Access Control (RBAC). The system logically separates adult and child patients, ensuring that child profiles cannot log in on their own. Instead, child records are strictly filtered to be visible only to authorized clinicians or legally assigned guardians. Guardians can easily create an account in the Anima application by selecting a specific registration toggle that links their profile to their child's unique patient ID. Once registered and logged in, the guardian can complete and submit the DSM-5 questionnaire on their child's behalf, ensuring the clinician has all the necessary behavioural context before the evaluation.

Technology Stack

Frontend UI: Figma (Prototyping), Microsoft PowerApps (Application), Outlook-MS PowerApps Extension (Clinician workflow integration).

Backend API: Python, FastAPI.

Database: Microsoft SQL Server.

Infrastructure: Docker (Containerized API and SQL Server environments).
