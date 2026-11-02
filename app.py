
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hallucination_guard.hallucination_guard import HallucinationGuard
from hallucination_guard.document_loader import DocumentLoader


CORPUS_PATH = PROJECT_ROOT / "data" / "sample" / "microsoft.json"

st.set_page_config(
    page_title="HallucinationGuard",
    page_icon="🛡️",
    layout="wide",
)


def show_result(result):
    verification = result["verification"]
    correction = result["correction"]
    metrics = verification["metrics"]

    st.subheader("Verification result")

    status = verification["status"]

    if status == "SUPPORTED":
        st.success("SUPPORTED — Evidence supports the claim.")
    elif status == "CONTRADICTED":
        st.error("CONTRADICTED — Evidence contradicts the claim.")
    elif status == "CONFLICTING":
        st.warning("CONFLICTING — Retrieved evidence disagrees.")
    else:
        st.info("INSUFFICIENT — There is not enough evidence.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Passages retrieved", metrics["passages_retrieved"])
    col2.metric("NLI evaluations", metrics["nli_evaluations"])
    col3.metric("Unique sources", metrics["unique_sources"])

    st.subheader("Correction decision")

    action = correction["action"]

    if action == "PRESERVE":
        st.success("Original claim preserved.")
    elif action == "REPLACE_WITH_EVIDENCE":
        st.success("A correction was validated.")
    else:
        st.warning("The system abstained from making a correction.")

    if action in ("PRESERVE", "REPLACE_WITH_EVIDENCE"):
        if correction.get("final_claim"):
            st.markdown("**Final claim**")
            st.write(correction["final_claim"])
    else:
        st.caption("No final claim was produced.")

    st.caption(correction["reason"])

    evidence_tab, nli_tab, audit_tab, metrics_tab = st.tabs([
        "Retrieved evidence",
        "NLI assessments",
        "Correction audit",
        "Technical metrics",
    ])

    with evidence_tab:
        for document in verification["evidence"]:
            with st.expander(
                f"{document['id']} · {document['source']}"
            ):
                st.write(document["text"])

    with nli_tab:
        for assessment in verification["nli_assessments"]:
            with st.expander(
                f"{assessment['id']} · {assessment['label']}"
            ):
                st.json(assessment["scores"])

    with audit_tab:
        attempts = correction.get("candidate_attempts", [])
        if attempts:
            st.json(attempts)
        else:
            st.info("No correction candidates were evaluated.")

    with metrics_tab:
        st.json(metrics)


st.title("🛡️ HallucinationGuard")

st.caption(
    "Evidence-based claim verification and hallucination detection"
)

st.write(
    "Upload your knowledge documents and verify factual claims "
    "against their contents."
)

st.divider()

st.subheader("Knowledge base")

uploaded_files = st.file_uploader(
    "Upload knowledge documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    loader = DocumentLoader()

    extracted_documents = []

    for uploaded_file in uploaded_files:
        try:
            extracted_text = loader.load(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )

            extracted_documents.append({
                "name": uploaded_file.name,
                "text": extracted_text,
            })

        except Exception as error:
            st.error(f"{uploaded_file.name}: {error}")

    if extracted_documents:
        st.success(
            f"{len(extracted_documents)} document(s) extracted successfully."
        )

        for document in extracted_documents:
            with st.expander(document["name"]):
                st.write(document["text"][:5000])

        st.session_state["uploaded_documents"] = extracted_documents

st.divider()

st.subheader("Claim verification")

with st.form("claim_verification_form"):
    claim = st.text_area(
        "Enter a factual claim",
        placeholder="Example: Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        height=120,
    )

    submitted = st.form_submit_button(
        "Verify Claim",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not claim.strip():
        st.warning("Please enter a factual claim.")

    elif not CORPUS_PATH.exists():
        st.error("The sample knowledge base could not be found.")

    else:
        st.session_state.pop("verification_result", None)

        try:
            with st.spinner("Verifying claim..."):
                guard = HallucinationGuard(
                    corpus_path=str(CORPUS_PATH),
                    max_passages=3,
                )

                result = guard.analyze(claim.strip())

            st.session_state["verification_result"] = result
            st.session_state["verified_claim"] = claim.strip()

        except Exception as error:
            st.error(f"Verification failed: {error}")

if "verification_result" in st.session_state:
    st.markdown("**Submitted claim**")
    st.write(st.session_state["verified_claim"])
    show_result(st.session_state["verification_result"])

st.divider()

st.caption("HallucinationGuard · Experimental prototype")
