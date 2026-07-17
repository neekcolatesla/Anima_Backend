"""
Anima - DSM-5 / text & demographic analysis engine.  (PLACEHOLDER)

Will own the text & demographic model: DSM-5 questionnaire answers, NLP data and
phenotypic demographics -> an ADHD probability contribution. Endpoints will be
added under the /api/analysis router below when the analysis engine is built.

(SysArchitecture: "Machine Learning Engine -> Text & Demographic Model".)
"""

import logging

from fastapi import APIRouter

logger = logging.getLogger("anima.analysis.dsm5")

# Endpoints to come, e.g. POST /api/analysis/dsm5/{patient_id}
router = APIRouter(prefix="/api/analysis", tags=["Analysis - DSM5"])