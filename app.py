"""
Smart Data Analysis Agent
An agentic AI assistant that analyses any CSV file automatically.
Agent loop: PLAN -> ACT (call a tool) -> OBSERVE -> REPORT
Run:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder


# ----------------------------- TOOLS -----------------------------------
def tool_profile(df):
    """Tool 1: understand the dataset."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes("number").columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
    }


def tool_clean(df):
    """Tool 2: remove duplicates and fill missing values."""
    log = []
    before = len(df)
    df = df.drop_duplicates().copy()
    log.append(f"Removed {before - len(df)} duplicate rows")
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            log.append(f"Filled {missing} missing values in '{col}' with median")
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            log.append(f"Filled {missing} missing values in '{col}' with mode")
    return df, log


def tool_visualize(df):
    """Tool 3: draw histograms and a correlation summary."""
    figs = []
    num = df.select_dtypes("number")
    if not num.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        num.hist(ax=ax if num.shape[1] == 1 else None, bins=20)
        figs.append(plt.gcf())
    return figs, (num.corr().round(2) if num.shape[1] > 1 else None)


def tool_model(df, target):
    """Tool 4: train a Random Forest model on the chosen target column."""
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    y = df[target]
    is_class = (not pd.api.types.is_numeric_dtype(y)) or y.nunique() <= 10
    if is_class:
        y = LabelEncoder().fit_transform(y.astype(str))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    if is_class:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)
        score, metric, kind = accuracy_score(y_te, model.predict(X_te)), "Accuracy", "Classification"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)
        score, metric, kind = r2_score(y_te, model.predict(X_te)), "R2 score", "Regression"
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
    return {"task": kind, "metric": metric, "score": round(float(score), 3), "top_features": imp}


# ----------------------------- AGENT -----------------------------------
class DataAgent:
    """A simple rule-based agent that plans which tools to run and keeps a memory log."""

    def __init__(self):
        self.memory = []  # Thought / Action / Observation log

    def note(self, kind, text):
        self.memory.append((kind, text))

    def plan(self, df, target):
        steps = ["profile"]
        if df.isna().sum().sum() > 0 or df.duplicated().sum() > 0:
            steps.append("clean")
        steps.append("visualize")
        if target and target != "(none)":
            steps.append("model")
        self.note("Thought", "Plan created: " + " -> ".join(steps))
        return steps

    def run(self, df, target):
        results = {}
        for step in self.plan(df, target):
            self.note("Action", f"Running tool: {step}")
            if step == "profile":
                results["profile"] = tool_profile(df)
                self.note("Observation", f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns")
            elif step == "clean":
                df, log = tool_clean(df)
                results["clean_log"] = log
                self.note("Observation", "; ".join(log))
            elif step == "visualize":
                results["figs"], results["corr"] = tool_visualize(df)
                self.note("Observation", "Charts generated")
            elif step == "model":
                results["model"] = tool_model(df, target)
                m = results["model"]
                self.note("Observation", f"{m['task']} model trained, {m['metric']} = {m['score']}")
        results["clean_df"] = df
        return results


# ------------------------------ UI -------------------------------------
st.set_page_config(page_title="Smart Data Analysis Agent", layout="wide")
st.title("Smart Data Analysis Agent")
st.caption("Upload a CSV file and let the AI agent profile, clean, visualise and model it.")

file = st.file_uploader("Upload CSV file", type=["csv"])
if file:
    data = pd.read_csv(file)
    st.subheader("Data preview")
    st.dataframe(data.head())
    target = st.selectbox("Select target column to predict (optional)", ["(none)"] + data.columns.tolist())

    if st.button("Run Agent"):
        agent = DataAgent()
        res = agent.run(data, target)

        st.subheader("1. Dataset profile")
        st.json(res["profile"])

        if "clean_log" in res:
            st.subheader("2. Cleaning report")
            for line in res["clean_log"]:
                st.write("-", line)

        st.subheader("3. Visualisation")
        for fig in res["figs"]:
            st.pyplot(fig)
        if res["corr"] is not None:
            st.write("Correlation matrix")
            st.dataframe(res["corr"])

        if "model" in res:
            m = res["model"]
            st.subheader("4. Model result")
            st.write(f"Task: **{m['task']}** | {m['metric']}: **{m['score']}**")
            st.write("Top 5 important features")
            st.bar_chart(m["top_features"])

        st.subheader("Agent memory log")
        for kind, text in agent.memory:
            st.write(f"**{kind}:** {text}")

        st.download_button("Download cleaned CSV", res["clean_df"].to_csv(index=False), "cleaned_data.csv")
