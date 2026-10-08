import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="AI-Based Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F8F5F5;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .hero {
        background: linear-gradient(135deg, #6E1025 0%, #8E1733 55%, #A51D3D 100%);
        padding: 42px 48px;
        border-radius: 0 0 24px 24px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 30px rgba(110,16,37,0.18);
    }

    .hero h1 {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 15px;
        opacity: 0.92;
        margin: 0;
    }

    .section-title {
        color: #6E1025;
        font-size: 23px;
        font-weight: 750;
        margin-top: 18px;
        margin-bottom: 18px;
    }

    .card {
        background: white;
        border: 1px solid #E8DDE0;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 5px 18px rgba(60,20,30,0.06);
        margin-bottom: 20px;
    }

    .card-title {
        color: #6E1025;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .result-card {
        background: white;
        border: 1px solid #E8DDE0;
        border-radius: 22px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 8px 28px rgba(60,20,30,0.08);
        margin-top: 15px;
    }

    .risk-label {
        font-size: 27px;
        font-weight: 800;
        color: #6E1025;
        margin-top: -10px;
    }

    .probability {
        font-size: 17px;
        color: #5C5154;
        font-weight: 600;
    }

    .info-box {
        background: #FBF1F3;
        border-left: 4px solid #8E1733;
        padding: 16px 18px;
        border-radius: 10px;
        color: #4A343A;
        margin-top: 15px;
    }

    .suggestion {
        background: #FFFFFF;
        border: 1px solid #E8DDE0;
        border-radius: 13px;
        padding: 14px 16px;
        margin-bottom: 10px;
        color: #43363A;
    }

    .chat-user {
        background: #F3E5E8;
        padding: 12px 16px;
        border-radius: 14px 14px 4px 14px;
        margin: 8px 0 8px 20%;
        color: #4A252F;
    }

    .chat-bot {
        background: #FFFFFF;
        border: 1px solid #E8DDE0;
        padding: 12px 16px;
        border-radius: 14px 14px 14px 4px;
        margin: 8px 20% 8px 0;
        color: #40363A;
    }

    div.stButton > button {
        background: #8E1733;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 11px 25px;
        font-weight: 700;
        width: 100%;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        background: #6E1025;
        color: white;
        border: none;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 9px;


[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stNumberInput label,
.stSelectbox label,
.stTextInput label {
    color: #6E1025 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
    }
    /* Make all Streamlit input labels clearly visible */
[data-testid="stWidgetLabel"] {
    color: #6E1025 !important;
    opacity: 1 !important;
}

[data-testid="stWidgetLabel"] p {
    color: #6E1025 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

[data-testid="stWidgetLabel"] label {
    color: #6E1025 !important;
    opacity: 1 !important;
}New Zealand default

    .footer {
        text-align: center;
        color: #76696D;
        font-size: 12px;
        padding: 25px 0 10px 0;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>AI-Based Credit Risk Assessment</h1>
        <p>Smart, data-driven credit risk assessment based on applicant financial and personal information.</p>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv("credit_risk_dataset.csv")

data = load_data()

X = data.drop("default", axis=1)
y = data["default"]

numeric_features = [
    "age",
    "income",
    "loan_amount",
    "loan_term_months",
    "credit_history",
    "employment_years",
    "existing_debt",
    "dependents"
]

categorical_features = [
    "employment_status",
    "residence"
]

numeric_processing = Pipeline(
    steps=[
        ("missing_values", SimpleImputer(strategy="median")),
        ("scaling", StandardScaler())
    ]
)

categorical_processing = Pipeline(
    steps=[
        ("missing_values", SimpleImputer(strategy="most_frequent")),
        ("encoding", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numeric_processing, numeric_features),
        ("categorical", categorical_processing, categorical_features)
    ]
)

@st.cache_resource
def train_model(data):

    X = data.drop("default", axis=1)
    y = data["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [10, 15, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    confusion = confusion_matrix(
        y_test,
        predictions
    )

    return best_model, accuracy, confusion

machine_learning_pipeline, accuracy, confusion = train_model(data)

st.markdown(
    '<div class="section-title">Applicant Information</div>',
    unsafe_allow_html=True
)



st.markdown(
    '<div class="card-title">Personal Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30
    )

with col2:
    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=1
    )

with col3:
    residence = st.selectbox(
        "Residence",
        ["Owned", "Rented", "Other"]
    )

st.markdown(
    '<div class="card-title">Financial Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input(
        "Annual Income (₹)",
        min_value=10000,
        max_value=10000000,
        value=500000,
        step=10000
    )

with col2:
    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=5000,
        max_value=10000000,
        value=300000,
        step=10000
    )

with col3:
    loan_term = st.selectbox(
        "Loan Term (Months)",
        [12, 24, 36, 48, 60]
    )

st.markdown(
    '<div class="card-title">Employment & Credit Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    credit_history_text = st.selectbox(
        "Credit History",
        ["Good", "Poor"]
    )

with col2:
    employment_years = st.number_input(
        "Years of Employment",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )

with col3:
    employment_status = st.selectbox(
        "Employment Status",
        ["Employed", "Self-employed", "Unemployed"]
    )

col1, col2 = st.columns(2)

with col1:
    existing_debt = st.number_input(
        "Existing Debt (₹)",
        min_value=0,
        max_value=10000000,
        value=50000,
        step=5000
    )

with col2:
    st.write("")
    st.write("")
    assess_button = st.button(
        "Assess My Credit Risk",
        type="primary"
    )



if "assessment_done" not in st.session_state:
    st.session_state.assessment_done = False

if "risk" not in st.session_state:
    st.session_state.risk = None

if "probability" not in st.session_state:
    st.session_state.probability = None

if "applicant_data" not in st.session_state:
    st.session_state.applicant_data = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if assess_button:

    credit_history = 1 if credit_history_text == "Good" else 0

    applicant_data = pd.DataFrame(
        {
            "age": [age],
            "income": [income],
            "loan_amount": [loan_amount],
            "loan_term_months": [loan_term],
            "credit_history": [credit_history],
            "employment_years": [employment_years],
            "employment_status": [employment_status],
            "existing_debt": [existing_debt],
            "dependents": [dependents],
            "residence": [residence]
        }
    )

    probability = machine_learning_pipeline.predict_proba(
        applicant_data
    )[0][1]

    if probability <= 0.30:
        risk = "LOW RISK"
    elif probability <= 0.60:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK"

    st.session_state.assessment_done = True
    st.session_state.risk = risk
    st.session_state.probability = probability
    st.session_state.applicant_data = applicant_data
    st.session_state.chat_history = []

if st.session_state.assessment_done:

    probability = st.session_state.probability
    risk = st.session_state.risk
    applicant_data = st.session_state.applicant_data

    st.markdown(
        '<div class="section-title">Your Credit Risk Assessment</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 4.3),
        facecolor="white"
    )

    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(
        Wedge(
            (0, 0),
            1,
            0,
            180,
            width=0.25,
            facecolor="#E8E1E3",
            edgecolor="white"
        )
    )

    low_end = 54
    medium_end = 108

    ax.add_patch(
        Wedge(
            (0, 0),
            1,
            0,
            low_end,
            width=0.25,
            facecolor="#B7C9B1",
            edgecolor="white"
        )
    )

    ax.add_patch(
        Wedge(
            (0, 0),
            1,
            low_end,
            medium_end,
            width=0.25,
            facecolor="#D8C28A",
            edgecolor="white"
        )
    )

    ax.add_patch(
        Wedge(
            (0, 0),
            1,
            medium_end,
            180,
            width=0.25,
            facecolor="#B86A7A",
            edgecolor="white"
        )
    )

    angle = 180 - (probability * 180)

    x = 0.77 * np.cos(np.radians(angle))
    y_pos = 0.77 * np.sin(np.radians(angle))

    ax.plot(
        [0, x],
        [0, y_pos],
        linewidth=3,
        color="#3A252B"
    )

    ax.scatter(
        [x],
        [y_pos],
        s=100,
        color="#8E1733",
        zorder=5
    )

    ax.text(
        0,
        0.18,
        f"{probability:.0%}",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="#6E1025"
    )

    ax.text(
        -0.98,
        -0.08,
        "LOW",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#5B6E55"
    )

    ax.text(
        0,
        -0.08,
        "MEDIUM",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#80652F"
    )

    ax.text(
        0.98,
        -0.08,
        "HIGH",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#8E1733"
    )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.18, 1.08)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="risk-label">{risk}</div>
            <div class="probability">
                Estimated Default Probability: {probability:.2%}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if risk == "LOW RISK":
        explanation = (
            "Based on the information provided, the model estimates "
            "a relatively lower probability of default."
        )

        suggestions = [
            "Continue maintaining a good credit history.",
            "Keep existing debt at a manageable level.",
            "Maintain stable income and employment where possible."
        ]

    elif risk == "MEDIUM RISK":
        explanation = (
            "Based on the information provided, the model estimates "
            "a moderate probability of default. Some aspects of the "
            "applicant profile may require attention."
        )

        suggestions = [
            "Work toward reducing existing debt.",
            "Maintain or improve your credit history.",
            "Maintain stable income and employment where possible."
        ]

    else:
        explanation = (
            "Based on the information provided, the model estimates "
            "a higher probability of default."
        )

        suggestions = [
            "Consider reducing existing debt.",
            "Work on maintaining or improving your credit history.",
            "Maintain stable and consistent income where possible."
        ]

    st.markdown(
        f"""
        <div class="info-box">
            <strong>What does this result mean?</strong><br><br>
            {explanation}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">How Can You Improve Your Profile?</div>',
        unsafe_allow_html=True
    )

    for suggestion in suggestions:
        st.markdown(
            f'<div class="suggestion">• {suggestion}</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Assessment Information</div>',
        unsafe_allow_html=True
    )

    display_data = pd.DataFrame(
        {
            "Information": [
                "Credit History",
                "Employment Status",
                "Annual Income",
                "Loan Amount",
                "Existing Debt"
            ],
            "Provided Value": [
                credit_history_text,
                employment_status,
                f"₹{income:,.0f}",
                f"₹{loan_amount:,.0f}",
                f"₹{existing_debt:,.0f}"
            ]
        }
    )

    st.dataframe(
        display_data,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">💬 Credit Risk Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
        <div class="card-title">Have a question about your assessment?</div>
        Ask about your risk level, default probability, or ways to improve your profile.
        </div>
        """,
        unsafe_allow_html=True
    )

    quick_col1, quick_col2, quick_col3 = st.columns(3)

    with quick_col1:
        why_button = st.button("Why is my risk this level?")

    with quick_col2:
        improve_button = st.button("How can I improve?")

    with quick_col3:
        probability_button = st.button("What does my probability mean?")

    if why_button:
        st.session_state.chat_history.append(
            (
                "Why is my risk this level?",
                f"Your current assessment is {risk.lower()} with an estimated default probability of {probability:.0%}. The model considers the applicant information provided, including income, loan amount, existing debt, credit history and employment information."
            )
        )

    if improve_button:
        st.session_state.chat_history.append(
            (
                "How can I improve?",
                "You can focus on maintaining a good credit history, keeping existing debt manageable and maintaining stable income or employment. These are general suggestions and do not guarantee a different model prediction."
            )
        )

    if probability_button:
        st.session_state.chat_history.append(
            (
                "What does my probability mean?",
                f"Your estimated default probability is {probability:.0%}. This is the probability produced by the machine-learning model for the applicant information you entered. It is an estimate, not a guarantee of future repayment."
            )
        )

    for question, answer in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-user"><strong>You:</strong> {question}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="chat-bot"><strong>Assistant:</strong> {answer}</div>',
            unsafe_allow_html=True
        )

    user_question = st.chat_input(
        "Ask about your credit risk assessment..."
    )

    if user_question:

        question = user_question.lower()

        if any(word in question for word in ["why", "reason"]):

            answer = (
                f"Your current assessment is {risk.lower()} with an "
                f"estimated default probability of {probability:.0%}. "
                "The model considers the applicant information provided, "
                "including income, loan amount, existing debt, credit "
                "history and employment information."
            )

        elif any(
            word in question
            for word in ["improve", "better", "increase", "change"]
        ):

            answer = (
                "You can focus on maintaining a good credit history, "
                "reducing existing debt and maintaining stable income "
                "or employment. These are general suggestions and do "
                "not guarantee a different model prediction."
            )

        elif any(
            word in question
            for word in [
                "probability",
                "percentage",
                "percent",
                "risk score"
            ]
        ):

            answer = (
                f"Your estimated default probability is {probability:.0%}. "
                "It is the probability produced by the machine-learning "
                "model for the information you entered. It is an estimate, "
                "not a guarantee."
            )

        elif any(
            word in question
            for word in ["low", "medium", "high", "risk"]
        ):

            answer = (
                f"Your current assessment is {risk.lower()}. "
                f"The estimated default probability is {probability:.0%}. "
                "The risk level is determined from the model's estimated "
                "probability using the application's Low, Medium and High "
                "risk ranges."
            )

        elif any(
            word in question
            for word in [
                "loan",
                "amount",
                "debt",
                "income",
                "credit",
                "employment"
            ]
        ):

            answer = (
                "The assessment uses the applicant information entered "
                "in the form, including income, loan amount, existing debt, "
                "credit history, employment information, age and dependents."
            )

        else:

            answer = (
                "I can help explain your risk level, estimated default "
                "probability, or general ways to improve your financial profile. "
                "Try asking: 'Why is my risk high?' or 'How can I improve?'"
            )

        st.session_state.chat_history.append(
            (user_question, answer)
        )

        st.rerun()


"""
    <div class="footer">
        This system is a decision-support prototype for academic purposes.
        It does not independently approve or reject a real loan.
    </div>
    """,
unsafe_allow_html=True
