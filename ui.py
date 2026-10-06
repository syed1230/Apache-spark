import streamlit as st
import plotly.express as px
import pandas as pd
import builtins

from data.data_manager import (
    add_student,
    load_students,
    student_exists
)

from analytics import (
    register_student_view,
    sql_class_summary,
    sql_top_students,
    sql_at_risk_students,
    rdd_grade_distribution,
    rdd_pass_fail_distribution
)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg
from pyspark.sql.functions import round as spark_round


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM COLOR THEME — TAUPE, UMBER, CACAO
# ============================================================
st.markdown("""
    <style>

    /* Main background */
    .stApp {
        background-color: #EFF0D1;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #4A5568 !important;
    }

    /* Main text */
    .stMarkdown, .stText, p, h1, h2, h3, h4, label {
        color: #2D3748 !important;
    }

    /* Cards / metric containers */
    [data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Metric label */
    [data-testid="metric-container"] label {
        color: #718096 !important;
    }

    /* Metric value */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #2D3748 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #7EB8D4;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background-color: #5A9DBF;
        color: #FFFFFF;
    }

    /* Input fields */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select {
        background-color: #FFFFFF;
        color: #2D3748;
        border: 1px solid #CBD5E0;
        border-radius: 6px;
    }

    /* Form */
    [data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #EDF2F7;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #718096;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #7EB8D4;
        color: #FFFFFF !important;
        border-radius: 8px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background-color: #FFFFFF;
    }

    /* Divider */
    hr {
        border-color: #E2E8F0;
    }

    /* Success */
    .stSuccess {
        background-color: #F0FFF4;
        border-color: #9AE6B4;
        color: #276749;
    }

    /* Error */
    .stError {
        background-color: #FFF5F5;
        border-color: #FEB2B2;
        color: #9B2C2C;
    }

    /* Warning */
    .stWarning {
        background-color: #FFFBEB;
        border-color: #F6E05E;
        color: #975A16;
    }

    /* Title styling */
    .main-title {
        color: #2D3748;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sub-title {
        color: #718096;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    </style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME — TAUPE UMBER CACAO
# ============================================================

PLOTLY_COLORS = [
    "#7EB8D4",   # soft sky blue
    "#85C1A3",   # mint green
    "#F4A97F",   # warm peach
    "#B8A9D9",   # lavender
    "#F7C96E",   # soft amber
    "#89C4C4",   # teal mist
    "#F29BB0"    # blush pink
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FDE0C5",   # near white background
    plot_bgcolor="#F0F4F8",    # very light blue grey
    font=dict(color="#4A5568"),
    title_font=dict(
        color="#2D3748",
        size=16
    ),
    xaxis=dict(
        gridcolor="#E2E8F0",   # very soft grid lines
        color="#718096"
    ),
    yaxis=dict(
        gridcolor="#E2E8F0",
        color="#718096"
    )
)


# ============================================================
# SPARK SETUP
# ============================================================

@st.cache_resource
def get_spark():
    spark = (
        SparkSession.builder
        .appName("Student Performance Analytics")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


spark = get_spark()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_dataframe():
    students = load_students()
    if not students:
        return None
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv("data/students.csv")
    )
    return df


def calculate_performance(df):
    df = df.withColumn(
        "Average",
        spark_round(
            (col("Math") + col("Science") + col("English")) / 3,
            2
        )
    )
    df = df.withColumn(
        "Grade",
        when(col("Average") >= 90, "A")
        .when(col("Average") >= 80, "B")
        .when(col("Average") >= 70, "C")
        .when(col("Average") >= 60, "D")
        .otherwise("F")
    )
    df = df.withColumn(
        "Status",
        when(col("Average") >= 40, "PASS")
        .otherwise("FAIL")
    )
    return df


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🎓 Student Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Dashboard",
        "➕ Add Student",
        "🔍 Find Student",
        "👥 All Students",
        "📈 Analytics",
        "⚡ Spark SQL & RDD"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with PySpark + Streamlit")


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.markdown(
        '<div class="main-title">📊 Student Performance Dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-title">Real-time analytics powered by Apache Spark</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    df = create_dataframe()

    if df is None:
        st.warning(
            "No student records found. Please add students first."
        )

    else:

        df = calculate_performance(df)
        pandas_df = df.toPandas()

        # --- METRIC CARDS ---
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Students",
                len(pandas_df)
            )

        with col2:
            st.metric(
                "Class Average",
                f"{pandas_df['Average'].mean():.2f}"
            )

        with col3:
            st.metric(
                "Avg Attendance",
                f"{pandas_df['Attendance'].mean():.2f}%"
            )

        with col4:
            st.metric(
                "Avg Study Hours",
                f"{pandas_df['Study_Hours'].mean():.2f}"
            )

        with col5:
            pass_pct = (
                len(pandas_df[pandas_df["Status"] == "PASS"])
                / len(pandas_df) * 100
            )
            st.metric(
                "Pass Percentage",
                f"{pass_pct:.1f}%"
            )

        st.markdown("---")

        # --- CHARTS ROW 1 ---
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Grade Distribution")
            grade_counts = (
                pandas_df["Grade"]
                .value_counts()
                .reset_index()
            )
            grade_counts.columns = ["Grade", "Count"]
            fig = px.bar(
                grade_counts,
                x="Grade",
                y="Count",
                color="Grade",
                title="Students per Grade",
                color_discrete_sequence=PLOTLY_COLORS
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Pass / Fail Distribution")
            status_counts = (
                pandas_df["Status"]
                .value_counts()
                .reset_index()
            )
            status_counts.columns = ["Status", "Count"]
            fig2 = px.pie(
                status_counts,
                names="Status",
                values="Count",
                color="Status",
                color_discrete_map={
                    "PASS": "#B5A49B",
                    "FAIL": "#635147"
                },
                title="Pass vs Fail"
            )
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        # --- CHARTS ROW 2 ---
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Subject Averages")
            subject_data = pd.DataFrame({
                "Subject": ["Math", "Science", "English"],
                "Average": [
                    pandas_df["Math"].mean(),
                    pandas_df["Science"].mean(),
                    pandas_df["English"].mean()
                ]
            })
            fig3 = px.bar(
                subject_data,
                x="Subject",
                y="Average",
                color="Subject",
                title="Average Marks by Subject",
                color_discrete_sequence=PLOTLY_COLORS
            )
            fig3.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            st.subheader("Attendance Distribution")
            fig4 = px.histogram(
                pandas_df,
                x="Attendance",
                nbins=10,
                title="Attendance Distribution",
                color_discrete_sequence=["#B5A49B"]
            )
            fig4.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig4, use_container_width=True)


# ============================================================
# PAGE 2 — ADD STUDENT
# ============================================================

elif page == "➕ Add Student":

    st.markdown(
        '<div class="main-title">➕ Add New Student</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    with st.form("add_student_form"):

        col1, col2 = st.columns(2)

        with col1:
            student_id = st.number_input(
                "Student ID",
                min_value=1,
                step=1
            )
            name = st.text_input("Full Name")
            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"]
            )
            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0
            )
            study_hours = st.number_input(
                "Study Hours per Day",
                min_value=0.0,
                max_value=24.0,
                value=4.0
            )

        with col2:
            math = st.number_input(
                "Math Marks",
                min_value=0.0,
                max_value=100.0,
                value=50.0
            )
            science = st.number_input(
                "Science Marks",
                min_value=0.0,
                max_value=100.0,
                value=50.0
            )
            english = st.number_input(
                "English Marks",
                min_value=0.0,
                max_value=100.0,
                value=50.0
            )

            # BUG FIX — use builtins.round
            # not PySpark round
            average = builtins.round(
                (math + science + english) / 3,
                2
            )
            st.metric("Calculated Average", average)

        submitted = st.form_submit_button(
            "➕ Add Student",
            use_container_width=True
        )

        if submitted:

            if not name.strip():
                st.error("Name cannot be empty.")

            elif student_exists(int(student_id)):
                st.error(
                    f"Student ID {int(student_id)} already exists!"
                )

            else:
                student = {
                    "Student_ID": int(student_id),
                    "Name": name.strip(),
                    "Gender": gender,
                    "Attendance": attendance,
                    "Study_Hours": study_hours,
                    "Math": math,
                    "Science": science,
                    "English": english
                }

                success = add_student(student)

                if success:
                    st.success(
                        f"✅ Student {name} added successfully!"
                    )
                    st.balloons()
                else:
                    st.error("Could not add student.")


# ============================================================
# PAGE 3 — FIND STUDENT
# ============================================================

elif page == "🔍 Find Student":

    st.markdown(
        '<div class="main-title">🔍 Find Student</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    student_id = st.number_input(
        "Enter Student ID",
        min_value=1,
        step=1
    )

    if st.button("🔍 Search", use_container_width=True):

        df = create_dataframe()

        if df is None:
            st.warning("No student records found.")

        else:
            df = calculate_performance(df)
            result = df.filter(
                col("Student_ID") == int(student_id)
            )

            if result.count() == 0:
                st.error("❌ Student not found.")

            else:
                student = result.collect()[0]

                st.success("✅ Student Found!")
                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Personal Information")
                    st.write(
                        f"**Student ID:** {student['Student_ID']}"
                    )
                    st.write(f"**Name:** {student['Name']}")
                    st.write(f"**Gender:** {student['Gender']}")
                    st.write(
                        f"**Attendance:** {student['Attendance']}%"
                    )
                    st.write(
                        f"**Study Hours:** "
                        f"{student['Study_Hours']} hrs/day"
                    )

                with col2:
                    st.subheader("Academic Performance")
                    st.write(f"**Math:** {student['Math']}")
                    st.write(f"**Science:** {student['Science']}")
                    st.write(f"**English:** {student['English']}")
                    st.write(f"**Average:** {student['Average']}")
                    st.write(f"**Grade:** {student['Grade']}")

                    if student["Status"] == "PASS":
                        st.success(
                            f"✅ Result: {student['Status']}"
                        )
                    else:
                        st.error(
                            f"❌ Result: {student['Status']}"
                        )


# ============================================================
# PAGE 4 — ALL STUDENTS
# ============================================================

elif page == "👥 All Students":

    st.markdown(
        '<div class="main-title">👥 All Students</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    df = create_dataframe()

    if df is None:
        st.warning("No student records found.")

    else:
        df = calculate_performance(df)
        pandas_df = df.toPandas()

        st.metric("Total Students", len(pandas_df))
        st.markdown("---")

        st.dataframe(
            pandas_df[[
                "Student_ID", "Name", "Gender",
                "Attendance", "Study_Hours",
                "Math", "Science", "English",
                "Average", "Grade", "Status"
            ]].sort_values("Student_ID"),
            use_container_width=True
        )


# ============================================================
# PAGE 5 — ANALYTICS
# ============================================================

elif page == "📈 Analytics":

    st.markdown(
        '<div class="main-title">📈 Detailed Analytics</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    df = create_dataframe()

    if df is None:
        st.warning("No student records found.")

    else:
        df = calculate_performance(df)
        pandas_df = df.toPandas()

        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Top Students",
            "⚠️ At Risk",
            "📚 Subjects",
            "🔗 Correlations"
        ])

        # --- TOP STUDENTS ---
        with tab1:
            st.subheader("🏆 Top 5 Students")
            top = (
                pandas_df
                .sort_values("Average", ascending=False)
                .head(5)
            )
            st.dataframe(
                top[[
                    "Student_ID", "Name",
                    "Average", "Grade", "Status"
                ]],
                use_container_width=True
            )
            fig = px.bar(
                top,
                x="Name",
                y="Average",
                color="Grade",
                title="Top 5 Students by Average",
                color_discrete_sequence=PLOTLY_COLORS
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # --- AT RISK ---
        with tab2:
            st.subheader("⚠️ At-Risk Students")
            at_risk = pandas_df[
                (pandas_df["Attendance"] < 75) |
                (pandas_df["Average"] < 50)
            ].copy()

            if len(at_risk) == 0:
                st.success("✅ No at-risk students found!")
            else:
                def get_reason(row):
                    if (
                        row["Attendance"] < 75
                        and row["Average"] < 50
                    ):
                        return "Low Attendance + Low Performance"
                    elif row["Attendance"] < 75:
                        return "Low Attendance"
                    else:
                        return "Low Academic Performance"

                at_risk["Reason"] = at_risk.apply(
                    get_reason, axis=1
                )
                st.dataframe(
                    at_risk[[
                        "Student_ID", "Name",
                        "Attendance", "Average",
                        "Grade", "Reason"
                    ]],
                    use_container_width=True
                )

        # --- SUBJECTS ---
        with tab3:
            st.subheader("📚 Subject Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Math Average",
                    f"{pandas_df['Math'].mean():.2f}"
                )
            with col2:
                st.metric(
                    "Science Average",
                    f"{pandas_df['Science'].mean():.2f}"
                )
            with col3:
                st.metric(
                    "English Average",
                    f"{pandas_df['English'].mean():.2f}"
                )

            fig = px.box(
                pandas_df,
                y=["Math", "Science", "English"],
                title="Subject Score Distribution",
                color_discrete_sequence=PLOTLY_COLORS
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # --- CORRELATIONS ---
        with tab4:
            st.subheader("🔗 Attendance vs Performance")
            fig = px.scatter(
                pandas_df,
                x="Attendance",
                y="Average",
                color="Grade",
                size="Study_Hours",
                hover_data=["Name"],
                title="Attendance vs Average Marks",
                color_discrete_sequence=PLOTLY_COLORS
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔗 Study Hours vs Performance")
            fig2 = px.scatter(
                pandas_df,
                x="Study_Hours",
                y="Average",
                color="Grade",
                hover_data=["Name"],
                title="Study Hours vs Average Marks",
                color_discrete_sequence=PLOTLY_COLORS
            )
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# PAGE 6 — SPARK SQL & RDD
# ============================================================

elif page == "⚡ Spark SQL & RDD":

    st.markdown(
        '<div class="main-title">⚡ Spark SQL & RDD Analytics</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    df = create_dataframe()

    if df is None:
        st.warning("No student records found.")

    else:
        df = calculate_performance(df)
        register_student_view(df)

        tab1, tab2 = st.tabs([
            "🔷 Spark SQL",
            "🔶 Spark RDD"
        ])

        # --- SPARK SQL ---
        with tab1:
            st.subheader("🔷 Spark SQL Queries")
            st.caption(
                "These queries run directly through spark.sql()"
            )

            if st.button(
                "▶ Run Class Summary",
                use_container_width=True
            ):
                result = sql_class_summary(spark).toPandas()
                st.dataframe(result, use_container_width=True)

            if st.button(
                "▶ Run Top Students",
                use_container_width=True
            ):
                result = sql_top_students(spark).toPandas()
                st.dataframe(result, use_container_width=True)

            if st.button(
                "▶ Run At-Risk Students",
                use_container_width=True
            ):
                result = sql_at_risk_students(spark).toPandas()
                if len(result) == 0:
                    st.success("✅ No at-risk students!")
                else:
                    st.dataframe(
                        result,
                        use_container_width=True
                    )

        # --- RDD ---
        with tab2:
            st.subheader("🔶 Spark RDD Operations")
            st.caption(
                "These use map(), reduceByKey(), collect()"
            )

            if st.button(
                "▶ Run Grade Distribution",
                use_container_width=True
            ):
                results = rdd_grade_distribution(df)
                rdd_df = pd.DataFrame(
                    results,
                    columns=["Grade", "Count"]
                ).sort_values("Grade")
                st.dataframe(
                    rdd_df,
                    use_container_width=True
                )
                fig = px.bar(
                    rdd_df,
                    x="Grade",
                    y="Count",
                    color="Grade",
                    title="Grade Distribution via RDD",
                    color_discrete_sequence=PLOTLY_COLORS
                )
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            if st.button(
                "▶ Run Pass/Fail Distribution",
                use_container_width=True
            ):
                results = rdd_pass_fail_distribution(df)
                rdd_df = pd.DataFrame(
                    results,
                    columns=["Status", "Count"]
                )
                st.dataframe(
                    rdd_df,
                    use_container_width=True
                )
                fig2 = px.pie(
                    rdd_df,
                    names="Status",
                    values="Count",
                    color="Status",
                    color_discrete_map={
                        "PASS": "#B5A49B",
                        "FAIL": "#635147"
                    },
                    title="Pass/Fail via RDD"
                )
                fig2.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)