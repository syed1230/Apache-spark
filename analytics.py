from pyspark.sql.functions import col, avg, round, count


def register_student_view(df):
    """Register the student DataFrame as a Spark SQL temporary view."""
    df.createOrReplaceTempView("students")


def sql_class_summary(spark):
    """Generate class summary using Spark SQL."""

    query = """
        SELECT
            COUNT(*) AS Total_Students,
            ROUND(AVG(Average), 2) AS Class_Average,
            ROUND(AVG(Attendance), 2) AS Average_Attendance,
            ROUND(AVG(Study_Hours), 2) AS Average_Study_Hours
        FROM students
    """

    return spark.sql(query)


def sql_top_students(spark):
    """Find the top 5 students using Spark SQL."""

    query = """
        SELECT
            Student_ID,
            Name,
            Average,
            Grade,
            Status
        FROM students
        ORDER BY Average DESC
        LIMIT 5
    """

    return spark.sql(query)


def sql_at_risk_students(spark):
    """Find students who may need academic attention."""

    query = """
        SELECT
            Student_ID,
            Name,
            Attendance,
            Average,
            Grade,
            Status
        FROM students
        WHERE Attendance < 75
           OR Average < 50
        ORDER BY Average ASC
    """

    return spark.sql(query)


def rdd_grade_distribution(df):
    """Calculate grade distribution using an RDD."""

    grade_rdd = df.select("Grade").rdd.map(
        lambda row: row["Grade"]
    )

    grade_counts = grade_rdd.map(
        lambda grade: (grade, 1)
    ).reduceByKey(
        lambda a, b: a + b
    )

    return grade_counts.collect()


def rdd_pass_fail_distribution(df):
    """Calculate pass/fail distribution using an RDD."""

    status_rdd = df.select("Status").rdd.map(
        lambda row: row["Status"]
    )

    status_counts = status_rdd.map(
        lambda status: (status, 1)
    ).reduceByKey(
        lambda a, b: a + b
    )

    return status_counts.collect()