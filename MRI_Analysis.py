"""
Anima - MRI image analysis engine.  (PLACEHOLDER)

Will own the image classification model: reads a patient's processed 2D slice
stacks (anat, anat_gm) from the MRI folders and produces an image-based ADHD
probability contribution. Endpoints will be added under the /api/analysis router
below when the analysis engine is built.

(SysArchitecture: "Machine Learning Engine -> Image Classification Model".)
"""

import logging

from fastapi import APIRouter

logger = logging.getLogger("anima.analysis.mri")

# Endpoints to come, e.g. POST /api/analysis/mri/{patient_id}
router = APIRouter(prefix="/api/analysis", tags=["Analysis - MRI"])