import builtins

import pandas as pd
import plotly.express as px
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.functions import round as spark_round

from analytics import (
    register_student_view,
    sql_class_summary,
    sql_top_students,
    sql_at_risk_students,
    rdd_grade_distribution,
    rdd_pass_fail_distribution,
)
from data.data_manager import add_student, load_students, student_exists
from ml_model import predict_student_support, get_model_evaluation


st.set_page_config(
    page_title="Student Assistance | Academic Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- THEME ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {
        --bg: #F4F7FC; --primary: #356FE5; --primary-dark: #2457C5;
        --primary-soft: #EAF1FF; --text: #1E293B; --muted: #64748B;
        --border: #E2E8F0;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--bg); color: var(--text); }
    .block-container { max-width: 1450px; padding-top: 1.8rem; padding-bottom: 3rem; }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] * { color: var(--text); }
    [data-testid="stSidebar"] [aria-checked="true"] { background: var(--primary-soft) !important; border-radius: 9px; }
    h1, h2, h3, h4, p, label { color: var(--text); }
    h1, h2, h3 { font-weight: 700; letter-spacing: -0.025em; }
    .hero { background: linear-gradient(120deg, #FFFFFF 0%, #EEF4FF 100%); border: 1px solid #DCE7FB; border-radius: 18px; padding: 26px 28px; margin-bottom: 20px; }
    .hero-kicker { color: #356FE5; font-size: .78rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; margin-bottom: 8px; }
    .hero-title { color: #17356F; font-size: 2rem; line-height: 1.2; font-weight: 800; margin: 0 0 8px; }
    .hero-subtitle { color: #64748B; font-size: .98rem; margin: 0; }
    [data-testid="stMetric"] { background: #FFF; border: 1px solid var(--border); border-radius: 14px; padding: 17px 18px; box-shadow: 0 4px 16px rgba(30,64,175,.035); }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .84rem; font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--primary) !important; font-size: 1.75rem; font-weight: 800; }
    .stButton > button, [data-testid="stFormSubmitButton"] > button { background: var(--primary); color: white !important; border: 1px solid var(--primary); border-radius: 9px; padding: .58rem 1rem; font-weight: 650; }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background: var(--primary-dark); border-color: var(--primary-dark); }
    .stTextInput input, .stNumberInput input, .stTextArea textarea { background: #FFF !important; color: var(--text) !important; border: 1px solid #CBD5E1 !important; border-radius: 9px !important; }
    [data-testid="stSelectbox"] > div > div { background: #FFF !important; border-radius: 9px; }
    [data-testid="stForm"] { background: #FFF; border: 1px solid var(--border); border-radius: 14px; padding: 22px; }
    .stTabs [data-baseweb="tab-list"] { background: #EAF1FF; border-radius: 10px; padding: 5px; gap: 5px; }
    .stTabs [data-baseweb="tab"] { color: #475569; border-radius: 8px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: #FFF !important; color: var(--primary) !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
    [data-testid="stAlert"] { border-radius: 10px; }
    hr { border-color: var(--border); margin: 1.4rem 0; }
    .page-title { color: #1D4ED8; font-size: 2.05rem; font-weight: 800; letter-spacing: -.04em; margin-bottom: .25rem; }
    .page-subtitle { color: #64748B; font-size: .98rem; margin-bottom: 1.25rem; }
    @media (max-width: 768px) { .block-container { padding-top: 1rem; } .hero-title { font-size: 1.55rem; } .page-title { font-size: 1.65rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_COLORS = ["#356FE5", "#60A5FA", "#93C5FD", "#818CF8", "#34D399", "#FBBF24", "#F87171"]
PASS_COLOR = "#16A34A"
FAIL_COLOR = "#DC2626"
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#475569", family="Inter, sans-serif"),
    title_font=dict(color="#1E293B", size=16, family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#E9EEF7", color="#64748B", zerolinecolor="#CBD5E1"),
    yaxis=dict(gridcolor="#E9EEF7", color="#64748B", zerolinecolor="#CBD5E1"),
    legend=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", borderwidth=1),
    margin=dict(l=20, r=20, t=55, b=20),
)


# ------------------------- SPARK SESSION -------------------------
@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("StudentAssistanceAnalytics")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


spark = get_spark()


def create_dataframe():
    """Read the student CSV with Spark; return None if there are no records."""
    students = load_students()
    if not students:
        return None
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv("data/students.csv")
    )


def calculate_performance(df):
    """Add average, grade, and pass/fail status to the Spark DataFrame."""
    df = df.withColumn(
        "Average",
        spark_round((col("Math") + col("Science") + col("English")) / 3, 2),
    )
    df = df.withColumn(
        "Grade",
        when(col("Average") >= 90, "A")
        .when(col("Average") >= 80, "B")
        .when(col("Average") >= 70, "C")
        .when(col("Average") >= 60, "D")
        .otherwise("F"),
    )
    return df.withColumn(
        "Status", when(col("Average") >= 40, "PASS").otherwise("FAIL")
    )


def show_title(title, subtitle=""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def to_pandas(df):
    return df.toPandas()


# ------------------------- SIDEBAR -------------------------
st.sidebar.markdown(
    """
    <div style="padding:10px 4px 18px;border-bottom:1px solid #E2E8F0;margin-bottom:15px">
      <div style="font-size:30px">🎓</div>
      <div style="color:#1D4ED8;font-size:21px;font-weight:800">Student Assistance</div>
      <div style="color:#64748B;font-size:12px;margin-top:5px">Academic Performance System</div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 Dashboard", "➕ Add Student", "🔍 Find Student", "👥 All Students",
        "📈 Analytics", "🤖 Performance Predictor", "⚡ Spark SQL & RDD",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, PySpark and Plotly")
st.sidebar.caption("Student records are stored in data/students.csv")


# ============================= DASHBOARD =============================
if page == "📊 Dashboard":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">ACADEMIC OVERVIEW</div>
          <div class="hero-title">Student Performance Dashboard</div>
          <p class="hero-subtitle">Understand class performance, attendance and subject trends at a glance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    df = create_dataframe()
    if df is None:
        st.info("No student records found. Use Add Student to create the first record.")
    else:
        pdf = to_pandas(calculate_performance(df))
        total = len(pdf)
        pass_pct = pdf["Status"].eq("PASS").sum() / total * 100 if total else 0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Students", total)
        c2.metric("Class Average", f'{pdf["Average"].mean():.1f}')
        c3.metric("Avg Attendance", f'{pdf["Attendance"].mean():.1f}%')
        c4.metric("Study Hours / Day", f'{pdf["Study_Hours"].mean():.1f}')
        c5.metric("Pass Rate", f"{pass_pct:.1f}%")

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            st.subheader("Grade Distribution")
            counts = pdf["Grade"].value_counts().reindex(["A", "B", "C", "D", "F"], fill_value=0)
            chart_df = counts.rename_axis("Grade").reset_index(name="Students")
            fig = px.bar(chart_df, x="Grade", y="Students", color="Grade",
                         color_discrete_sequence=PLOTLY_COLORS, title="Students by grade")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Pass / Fail")
            status = pdf["Status"].value_counts().reindex(["PASS", "FAIL"], fill_value=0)
            status_df = status.rename_axis("Status").reset_index(name="Students")
            fig = px.pie(status_df, names="Status", values="Students", hole=.55,
                         color="Status", color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                         title="Class result distribution")
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_traces(marker=dict(line=dict(color="#FFFFFF", width=2)))
            st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Average Marks by Subject")
            subject_df = pd.DataFrame({
                "Subject": ["Math", "Science", "English"],
                "Average": [pdf["Math"].mean(), pdf["Science"].mean(), pdf["English"].mean()],
            })
            fig = px.bar(subject_df, x="Subject", y="Average", color="Subject",
                         color_discrete_sequence=PLOTLY_COLORS, title="Subject averages")
            fig.update_layout(**PLOTLY_LAYOUT, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Attendance Distribution")
            fig = px.histogram(pdf, x="Attendance", nbins=10,
                               color_discrete_sequence=[PLOTLY_COLORS[0]],
                               title="How attendance is distributed")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Students Needing Attention")
        at_risk = pdf[(pdf["Attendance"] < 75) | (pdf["Average"] < 50)].copy()
        if at_risk.empty:
            st.success("No students meet the current attention thresholds.")
        else:
            st.dataframe(
                at_risk[["Student_ID", "Name", "Attendance", "Average", "Grade", "Status"]].sort_values("Average"),
                use_container_width=True, hide_index=True,
            )


# ============================= ADD STUDENT =============================
elif page == "➕ Add Student":
    show_title("Add a Student", "Enter student details and academic marks.")
    with st.form("add_student_form", clear_on_submit=False):
        left, right = st.columns(2)
        with left:
            student_id = st.number_input("Student ID", min_value=1, step=1)
            name = st.text_input("Full Name")
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
            study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=4.0)
        with right:
            math = st.number_input("Math Marks", min_value=0.0, max_value=100.0, value=50.0)
            science = st.number_input("Science Marks", min_value=0.0, max_value=100.0, value=50.0)
            english = st.number_input("English Marks", min_value=0.0, max_value=100.0, value=50.0)
            average = builtins.round((math + science + english) / 3, 2)
            st.metric("Calculated Average", f"{average:.2f} / 100")
        submitted = st.form_submit_button("Save Student", use_container_width=True)
        if submitted:
            if not name.strip():
                st.error("Name cannot be empty.")
            elif student_exists(int(student_id)):
                st.error(f"Student ID {int(student_id)} already exists.")
            else:
                student = {
                    "Student_ID": int(student_id), "Name": name.strip(), "Gender": gender,
                    "Attendance": attendance, "Study_Hours": study_hours,
                    "Math": math, "Science": science, "English": english,
                }
                if add_student(student):
                    st.success(f"Student {name.strip()} added successfully.")
                    st.balloons()
                else:
                    st.error("The student could not be added. Please check the ID.")


# ============================= FIND STUDENT =============================
elif page == "🔍 Find Student":
    show_title("Find a Student", "Search for a record by Student ID.")
    student_id = st.number_input("Student ID", min_value=1, step=1, key="find_student_id")
    if st.button("Search Student", use_container_width=True):
        df = create_dataframe()
        if df is None:
            st.info("No student records are available yet.")
        else:
            df = calculate_performance(df)
            rows = df.filter(col("Student_ID") == int(student_id)).limit(1).collect()
            if not rows:
                st.error("Student not found. Check the ID and try again.")
            else:
                s = rows[0]
                st.success("Student record found.")
                a, b = st.columns(2)
                with a:
                    st.subheader("Personal Information")
                    st.write(f"**Student ID:** {s['Student_ID']}")
                    st.write(f"**Name:** {s['Name']}")
                    st.write(f"**Gender:** {s['Gender']}")
                    st.write(f"**Attendance:** {s['Attendance']}%")
                    st.write(f"**Study hours:** {s['Study_Hours']} per day")
                with b:
                    st.subheader("Academic Performance")
                    st.write(f"**Math:** {s['Math']}")
                    st.write(f"**Science:** {s['Science']}")
                    st.write(f"**English:** {s['English']}")
                    st.metric("Average", f"{s['Average']:.2f} / 100")
                    st.write(f"**Grade:** {s['Grade']}")
                    st.success("Result: PASS") if s["Status"] == "PASS" else st.error("Result: FAIL")


# ============================= ALL STUDENTS =============================
elif page == "👥 All Students":
    show_title("All Students", "Browse and filter saved student records.")
    df = create_dataframe()
    if df is None:
        st.info("No student records are available yet.")
    else:
        pdf = to_pandas(calculate_performance(df))
        c1, c2 = st.columns([1, 2])
        c1.metric("Total Students", len(pdf))
        query = c2.text_input("Search by name or student ID")
        if query.strip():
            mask = (
                pdf["Name"].astype(str).str.contains(query, case=False, na=False)
                | pdf["Student_ID"].astype(str).str.contains(query, case=False, na=False)
            )
            pdf = pdf[mask]
        st.dataframe(
            pdf[["Student_ID", "Name", "Gender", "Attendance", "Study_Hours",
                 "Math", "Science", "English", "Average", "Grade", "Status"]].sort_values("Student_ID"),
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "Download displayed records as CSV",
            data=pdf.to_csv(index=False).encode("utf-8"),
            file_name="student_records.csv", mime="text/csv",
        )


# ============================= ANALYTICS =============================
elif page == "📈 Analytics":
    show_title("Detailed Analytics", "Explore student outcomes and possible support needs.")
    df = create_dataframe()
    if df is None:
        st.info("No student records are available yet.")
    else:
        pdf = to_pandas(calculate_performance(df))
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Students", "⚠️ At Risk", "📚 Subjects", "🔗 Correlations"])
        with tab1:
            st.subheader("Top 5 Students")
            top = pdf.sort_values("Average", ascending=False).head(5)
            st.dataframe(top[["Student_ID", "Name", "Average", "Grade", "Status"]],
                         use_container_width=True, hide_index=True)
            fig = px.bar(top, x="Name", y="Average", color="Grade",
                         color_discrete_sequence=PLOTLY_COLORS, title="Top students by average")
            fig.update_layout(**PLOTLY_LAYOUT, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.subheader("Students Requiring Attention")
            at_risk = pdf[(pdf["Attendance"] < 75) | (pdf["Average"] < 50)].copy()
            if at_risk.empty:
                st.success("No students meet the current attention thresholds.")
            else:
                def reason(row):
                    if row["Attendance"] < 75 and row["Average"] < 50:
                        return "Low attendance and low marks"
                    if row["Attendance"] < 75:
                        return "Attendance below 75%"
                    return "Average below 50"
                at_risk["Reason"] = at_risk.apply(reason, axis=1)
                st.dataframe(at_risk[["Student_ID", "Name", "Attendance", "Average", "Grade", "Reason"]].sort_values("Average"),
                             use_container_width=True, hide_index=True)
                st.caption("These are transparent threshold-based flags, not ML predictions.")
        with tab3:
            st.subheader("Subject Analysis")
            m1, m2, m3 = st.columns(3)
            m1.metric("Math Average", f'{pdf["Math"].mean():.2f}')
            m2.metric("Science Average", f'{pdf["Science"].mean():.2f}')
            m3.metric("English Average", f'{pdf["English"].mean():.2f}')
            fig = px.box(pdf, y=["Math", "Science", "English"], title="Subject score distribution",
                         color_discrete_sequence=PLOTLY_COLORS)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with tab4:
            st.subheader("Attendance vs Performance")
            fig = px.scatter(pdf, x="Attendance", y="Average", color="Grade", size="Study_Hours",
                             hover_data=["Name"], color_discrete_sequence=PLOTLY_COLORS,
                             title="Attendance vs average marks")
            fig.update_layout(**PLOTLY_LAYOUT, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.scatter(pdf, x="Study_Hours", y="Average", color="Grade",
                              hover_data=["Name"], color_discrete_sequence=PLOTLY_COLORS,
                              title="Study hours vs average marks")
            fig2.update_layout(**PLOTLY_LAYOUT, yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)


# ============================= ML PREDICTOR =============================
elif page == "🤖 Performance Predictor":
    show_title("Student Performance Predictor", "Explore a demonstration ML prediction and support suggestions.")
    st.warning(
        "Demonstration model: its training labels are generated from transparent example rules, "
        "not from verified historical student outcomes. Treat predictions as educational examples, "
        "not reliable forecasts about real students."
    )
    with st.form("predictor_form"):
        left, right = st.columns(2)
        with left:
            attendance = st.slider("Attendance (%)", 0, 100, 75)
            study_hours = st.slider("Study hours per day", 0.0, 12.0, 4.0, 0.5)
        with right:
            math = st.slider("Math marks", 0, 100, 50)
            science = st.slider("Science marks", 0, 100, 50)
            english = st.slider("English marks", 0, 100, 50)
        predict = st.form_submit_button("Predict Support Category", use_container_width=True)

    if predict:
        inputs = {
            "Attendance": float(attendance), "Study_Hours": float(study_hours),
            "Math": float(math), "Science": float(science), "English": float(english),
        }
        result = predict_student_support(inputs)
        st.markdown("---")
        st.subheader(f"Suggested category: {result['category']}")
        st.write(result["explanation"])
        st.subheader("Suggested next steps")
        for suggestion in result["suggestions"]:
            st.markdown(f"- {suggestion}")
        st.caption("The category is a demo model output, not a diagnosis or a guaranteed academic outcome.")

    # This block is intentionally inside the Performance Predictor branch.
    # It therefore appears ONLY on this page, not on the Dashboard or other pages.
    st.markdown("---")
    st.subheader("📊 Machine Learning Model Evaluation")
    metrics = get_model_evaluation()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    st.caption(f"Training samples: {metrics['training_samples']} | Testing samples: {metrics['test_samples']}")
    st.subheader("Confusion Matrix")
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["class_labels"],
        columns=metrics["class_labels"],
    )
    matrix.index.name = "Actual"
    matrix.columns.name = "Predicted"
    st.dataframe(matrix, use_container_width=True)
    st.info(
        "This demonstration model is trained on synthetic data. These scores measure "
        "how well it reproduces generated rules, not validated real-world student outcomes."
    )


# ============================= SPARK SQL & RDD =============================
elif page == "⚡ Spark SQL & RDD":
    show_title("Spark SQL & RDD Analytics", "Run distributed-data operations on the student dataset.")
    df = create_dataframe()
    if df is None:
        st.info("No student records are available yet.")
    else:
        df = calculate_performance(df)
        register_student_view(df)
        tab1, tab2 = st.tabs(["🔷 Spark SQL", "🔶 Spark RDD"])
        with tab1:
            st.subheader("Spark SQL Queries")
            st.caption("These buttons run queries through spark.sql().")
            if st.button("Run Class Summary", use_container_width=True):
                st.dataframe(sql_class_summary(spark).toPandas(), use_container_width=True, hide_index=True)
            if st.button("Run Top Students", use_container_width=True):
                st.dataframe(sql_top_students(spark).toPandas(), use_container_width=True, hide_index=True)
            if st.button("Run At-Risk Students", use_container_width=True):
                result = sql_at_risk_students(spark).toPandas()
                if result.empty:
                    st.success("No students meet the current attention thresholds.")
                else:
                    st.dataframe(result, use_container_width=True, hide_index=True)
        with tab2:
            st.subheader("Spark RDD Operations")
            st.caption("These use map(), reduceByKey() and collect().")
            if st.button("Run Grade Distribution", use_container_width=True):
                results = rdd_grade_distribution(df)
                rdd_df = pd.DataFrame(results, columns=["Grade", "Count"]).sort_values("Grade")
                st.dataframe(rdd_df, use_container_width=True, hide_index=True)
                fig = px.bar(rdd_df, x="Grade", y="Count", color="Grade",
                             color_discrete_sequence=PLOTLY_COLORS, title="Grade distribution via RDD")
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            if st.button("Run Pass / Fail Distribution", use_container_width=True):
                results = rdd_pass_fail_distribution(df)
                rdd_df = pd.DataFrame(results, columns=["Status", "Count"])
                st.dataframe(rdd_df, use_container_width=True, hide_index=True)
                fig = px.pie(rdd_df, names="Status", values="Count", hole=.55,
                             color="Status", color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                             title="Pass/fail distribution via RDD")
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_traces(marker=dict(line=dict(color="#FFFFFF", width=2)))
                st.plotly_chart(fig, use_container_width=True)
