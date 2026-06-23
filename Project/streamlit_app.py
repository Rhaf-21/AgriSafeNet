import streamlit as st
import requests

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="Agriculture Safety AI",
    page_icon="🌱",
    layout="centered"
)

# ======================
# Label Map
# ======================
pretty_label_map = {
    0: "Safe (In Domain)",
    1: "Unsafe - Pesticide Use",
    2: "Self Harm Risk",
    3: "Adversarial Input",
    4: "Indirect Unsafe Content",
    5: "Medical/Legal Advice",
    6: "Out of Scope",
    7: "Unknown / Ambiguous",
    8: "Policy Sensitive"
}

# ======================
# Title
# ======================
st.title("🌱 Agriculture Safety AI")
st.markdown("Enter a text and get classification + AI explanation")

# ======================
# Input
# ======================
text = st.text_area("✍️ Enter your text here:")

# ======================
# Predict Button
# ======================
if st.button("🚀 Predict"):

    if text.strip() == "":
        st.warning("Please enter text first")

    else:

        # ======================
        # Call FastAPI
        # ======================
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text}
            )

            result = response.json()

        except Exception as e:
            st.error(f"Error connecting to API: {e}")
            st.stop()

        # ======================
        # Raw Output (Debug)
        # ======================
        with st.expander("🔍 Raw Output"):
            st.json(result)

        # ======================
        # Prediction Section
        # ======================
        st.subheader("📊 Prediction Result")

        prediction = result.get("prediction")

        if prediction is None:
            label = result.get("label", "N/A")
            st.success(f"Label: {label}")

        else:

            # إذا الرقم
            if isinstance(prediction, int):
                label = pretty_label_map.get(prediction, "Unknown")

                st.success(f"Prediction Code: {prediction}")
                st.success(f"Label: {label}")

            # إذا نص (مثل out_of_scope)
            else:
                label = prediction
                st.success(f"Label: {label}")

        # ======================
        # Confidence
        # ======================
        st.metric("Confidence", result.get("confidence", 0))

        # ======================
        # OpenAI Explanation
        # ======================
        st.subheader("🧠 OpenAI Explanation")

        explanation = result.get("explanation")

        if explanation:
            if isinstance(explanation, str) and "error" not in explanation.lower():
                st.info(explanation)
            else:
                st.warning("⚠️ Explanation returned error or invalid response")
                st.code(explanation)
        else:
            st.info("ℹ️ No explanation returned from API")
