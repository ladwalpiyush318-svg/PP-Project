
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "personality_prediction_dataset_12000.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "personality_type"

# Remove ID from input features
X = df.drop(columns=["person_id", TARGET])
y = df[TARGET]

categorical_columns = X.select_dtypes(include="object").columns.tolist()
numerical_columns = X.select_dtypes(include="number").columns.tolist()

# ==========================================
# 2. TRAIN GAUSSIAN NAIVE BAYES MODEL
# ==========================================

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer([
    ("cat", categorical_transformer, categorical_columns),
    ("num", numerical_transformer, numerical_columns)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "my_model.pkl")

# ==========================================
# 3. MODEL PERFORMANCE
# ==========================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

auc = roc_auc_score(
    y_test,
    y_prob,
    labels=model.classes_,
    multi_class="ovr",
    average="weighted"
)

# ==========================================
# 4. DASHBOARD GRAPHS
# ==========================================

# Personality distribution
personality_counts = df[TARGET].value_counts().reset_index()
personality_counts.columns = ["Personality", "Count"]

fig_distribution = px.pie(
    personality_counts,
    names="Personality",
    values="Count",
    hole=0.45,
    title="Personality Type Distribution"
)

# Average personality scores
score_columns = numerical_columns

average_scores = df.groupby(TARGET)[score_columns].mean().reset_index()

average_scores = average_scores.melt(
    id_vars=TARGET,
    var_name="Personality Score",
    value_name="Average Score"
)

fig_scores = px.bar(
    average_scores,
    x="Personality Score",
    y="Average Score",
    color=TARGET,
    barmode="group",
    title="Average Personality Scores"
)

# Model performance chart
metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy", "Precision", "Recall",
        "F1 Score", "Kappa", "ROC-AUC", "MCC"
    ],
    "Score": [
        accuracy, precision, recall,
        f1, kappa, auc, mcc
    ]
})

fig_metrics = px.bar(
    metrics_df,
    x="Metric",
    y="Score",
    title="Gaussian Naive Bayes Performance",
    text_auto=".3f"
)

# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def predict_personality(
    social_preference,
    planning_habit,
    novelty_seeking,
    teamwork_style,
    stress_response,
    extraversion_score,
    conscientiousness_score,
    openness_score,
    agreeableness_score,
    neuroticism_score
):

    input_data = pd.DataFrame([{
        "social_preference": social_preference,
        "planning_habit": planning_habit,
        "novelty_seeking": novelty_seeking,
        "teamwork_style": teamwork_style,
        "stress_response": stress_response,
        "extraversion_score": extraversion_score,
        "conscientiousness_score": conscientiousness_score,
        "openness_score": openness_score,
        "agreeableness_score": agreeableness_score,
        "neuroticism_score": neuroticism_score
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    result = (
        f"Predicted Personality: {prediction}\n"
        f"Prediction Confidence: {confidence:.2f}%"
    )

    confidence_df = pd.DataFrame({
        "Personality": model.classes_,
        "Probability": probabilities * 100
    })

    fig_confidence = px.bar(
        confidence_df,
        x="Personality",
        y="Probability",
        color="Personality",
        title="Personality Prediction Probabilities",
        text_auto=".2f"
    )

    return result, fig_confidence

# ==========================================
# 6. DESIGN GRADIO DASHBOARD
# ==========================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🧠 PERSONALITY PREDICTION DASHBOARD

    ### Machine Learning Project | Gaussian Naive Bayes

    Analyze personality patterns and predict personality types
    using behavioral characteristics and personality scores.
    """)

    # Dataset overview
    with gr.Tab("📊 Dashboard Overview"):

        gr.Markdown("## Dataset Summary")

        with gr.Row():
            gr.Number(
                value=len(df),
                label="Total Records",
                interactive=False
            )
            gr.Number(
                value=len(df.columns),
                label="Total Columns",
                interactive=False
            )
            gr.Number(
                value=df[TARGET].nunique(),
                label="Personality Categories",
                interactive=False
            )
            gr.Number(
                value=df.isnull().sum().sum(),
                label="Missing Values",
                interactive=False
            )

        gr.Markdown("## Personality Distribution")

        gr.Plot(value=fig_distribution)

        gr.Markdown("## Personality Score Analysis")

        gr.Plot(value=fig_scores)

        gr.Markdown("## Dataset Preview")

        gr.Dataframe(
            value=df.head(10),
            label="First 10 Dataset Records"
        )

    # Model performance
    with gr.Tab("📈 Model Performance"):

        gr.Markdown("## Gaussian Naive Bayes Evaluation")

        with gr.Row():
            gr.Number(
                value=round(accuracy * 100, 2),
                label="Accuracy (%)",
                interactive=False
            )
            gr.Number(
                value=round(precision, 4),
                label="Precision",
                interactive=False
            )
            gr.Number(
                value=round(recall, 4),
                label="Recall",
                interactive=False
            )
            gr.Number(
                value=round(f1, 4),
                label="F1 Score",
                interactive=False
            )

        gr.Plot(value=fig_metrics)

        gr.Markdown("### Detailed Performance Metrics")

        gr.Dataframe(value=metrics_df)

    # Personality prediction
    with gr.Tab("🔮 Predict Personality"):

        gr.Markdown("""
        ## Personality Prediction System

        Enter the person's behavioral preferences and personality
        scores to predict their personality category.
        """)

        with gr.Row():

            with gr.Column():

                gr.Markdown("### Behavioral Information")

                social = gr.Dropdown(
                    choices=sorted(df["social_preference"].dropna().unique().tolist()),
                    label="Social Preference",
                    value=df["social_preference"].mode()[0]
                )

                planning = gr.Dropdown(
                    choices=sorted(df["planning_habit"].dropna().unique().tolist()),
                    label="Planning Habit",
                    value=df["planning_habit"].mode()[0]
                )

                novelty = gr.Dropdown(
                    choices=sorted(df["novelty_seeking"].dropna().unique().tolist()),
                    label="Novelty Seeking",
                    value=df["novelty_seeking"].mode()[0]
                )

                teamwork = gr.Dropdown(
                    choices=sorted(df["teamwork_style"].dropna().unique().tolist()),
                    label="Teamwork Style",
                    value=df["teamwork_style"].mode()[0]
                )

                stress = gr.Dropdown(
                    choices=sorted(df["stress_response"].dropna().unique().tolist()),
                    label="Stress Response",
                    value=df["stress_response"].mode()[0]
                )

            with gr.Column():

                gr.Markdown("### Personality Scores (1–99)")

                ext = gr.Slider(
                    1, 99, value=50, step=1,
                    label="Extraversion Score"
                )

                con = gr.Slider(
                    1, 99, value=50, step=1,
                    label="Conscientiousness Score"
                )

                ope = gr.Slider(
                    1, 99, value=50, step=1,
                    label="Openness Score"
                )

                agr = gr.Slider(
                    1, 99, value=50, step=1,
                    label="Agreeableness Score"
                )

                neu = gr.Slider(
                    1, 99, value=50, step=1,
                    label="Neuroticism Score"
                )

        predict_button = gr.Button(
            "🔮 Predict Personality",
            variant="primary"
        )

        prediction_output = gr.Textbox(
            label="Prediction Result",
            lines=3
        )

        confidence_output = gr.Plot(
            label="Prediction Confidence"
        )

        predict_button.click(
            fn=predict_personality,
            inputs=[
                social, planning, novelty,
                teamwork, stress,
                ext, con, ope, agr, neu
            ],
            outputs=[
                prediction_output,
                confidence_output
            ]
        )

    gr.Markdown("""
    ---
    **Project:** Personality Prediction Using Machine Learning

    **Algorithm:** Gaussian Naive Bayes

    **Dataset:** Personality Prediction Dataset (12,000 Records)
    """)

# ==========================================
# 7. LAUNCH APPLICATION
# ==========================================



if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )
