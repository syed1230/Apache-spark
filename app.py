from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, when, avg, max, min

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


CSV_FILE = "data/students.csv"


# ============================================================
# SPARK SETUP
# ============================================================

spark = (
    SparkSession.builder
    .appName("Student Performance Analytics System")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# ============================================================
# CREATE DATAFRAME
# ============================================================

def create_dataframe():

    students = load_students()

    if not students:
        return None

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(CSV_FILE)
    )

    return df


# ============================================================
# CALCULATE PERFORMANCE
# ============================================================

def calculate_performance(df):

    df = df.withColumn(
        "Average",
        round(
            (
                col("Math")
                + col("Science")
                + col("English")
            ) / 3,
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
# 1. ADD STUDENT
# ============================================================

def add_student_menu():

    print("\n" + "=" * 60)
    print("                    ADD STUDENT")
    print("=" * 60)

    try:

        student_id = int(input("Enter Student ID: "))

        if student_exists(student_id):
            print("\nStudent ID already exists!")
            return

        name = input("Enter Name: ").strip()
        gender = input("Enter Gender: ").strip()

        attendance = float(input("Enter Attendance (%): "))
        study_hours = float(input("Enter Study Hours/Day: "))

        math = float(input("Enter Math Marks: "))
        science = float(input("Enter Science Marks: "))
        english = float(input("Enter English Marks: "))

        if not name:
            print("\nName cannot be empty.")
            return

        if not 0 <= attendance <= 100:
            print("\nAttendance must be between 0 and 100.")
            return

        if study_hours < 0:
            print("\nStudy hours cannot be negative.")
            return

        for mark in [math, science, english]:

            if not 0 <= mark <= 100:
                print("\nMarks must be between 0 and 100.")
                return

        student = {
            "Student_ID": student_id,
            "Name": name,
            "Gender": gender,
            "Attendance": attendance,
            "Study_Hours": study_hours,
            "Math": math,
            "Science": science,
            "English": english
        }

        success = add_student(student)

        if success:
            print("\nStudent added successfully!")

        else:
            print("\nCould not add student.")

    except ValueError:

        print("\nPlease enter valid numeric values.")


# ============================================================
# 2. FIND STUDENT
# ============================================================

def find_student():

    print("\n" + "=" * 60)
    print("                    FIND STUDENT")
    print("=" * 60)

    try:

        student_id = int(input("Enter Student ID: "))

    except ValueError:

        print("\nPlease enter a valid Student ID.")
        return

    df = create_dataframe()

    if df is None:

        print("\nNo student records found.")
        return

    df = calculate_performance(df)

    result = df.filter(
        col("Student_ID") == student_id
    )

    if result.count() == 0:

        print("\nStudent not found.")
        return

    student = result.collect()[0]

    print("\n" + "=" * 60)
    print("                  STUDENT INFORMATION")
    print("=" * 60)

    print(f"Student ID    : {student['Student_ID']}")
    print(f"Name          : {student['Name']}")
    print(f"Gender        : {student['Gender']}")
    print(f"Attendance    : {student['Attendance']}%")
    print(f"Study Hours   : {student['Study_Hours']} hours/day")

    print("-" * 60)

    print("MARKS")

    print("-" * 60)

    print(f"Math          : {student['Math']}")
    print(f"Science       : {student['Science']}")
    print(f"English       : {student['English']}")

    print("-" * 60)

    print(f"Average       : {student['Average']}")
    print(f"Grade         : {student['Grade']}")
    print(f"Result        : {student['Status']}")

    print("=" * 60)


# ============================================================
# 3. VIEW ALL STUDENTS
# ============================================================

def view_all_students():

    print("\n" + "=" * 70)
    print("                     ALL STUDENTS")
    print("=" * 70)

    df = create_dataframe()

    if df is None:

        print("\nNo student records found.")
        return

    df = calculate_performance(df)

    (
        df.select(
            "Student_ID",
            "Name",
            "Attendance",
            "Study_Hours",
            "Math",
            "Science",
            "English",
            "Average",
            "Grade",
            "Status"
        )
        .orderBy("Student_ID")
        .show(truncate=False)
    )


# ============================================================
# 4. CLASS STATISTICS
# ============================================================

def class_statistics():

    print("\n" + "=" * 60)
    print("                  CLASS STATISTICS")
    print("=" * 60)

    df = create_dataframe()

    if df is None:

        print("\nNo student records found.")
        return

    df = calculate_performance(df)

    total_students = df.count()

    class_average = (
        df.select(
            round(avg("Average"), 2).alias("Class_Average")
        )
        .collect()[0]["Class_Average"]
    )

    subject_averages = (
        df.select(
            round(avg("Math"), 2).alias("Math_Average"),
            round(avg("Science"), 2).alias("Science_Average"),
            round(avg("English"), 2).alias("English_Average")
        )
        .collect()[0]
    )

    pass_count = df.filter(
        col("Status") == "PASS"
    ).count()

    fail_count = df.filter(
        col("Status") == "FAIL"
    ).count()

    print(f"Total Students : {total_students}")
    print(f"Class Average  : {class_average}")

    print("\nSubject Averages")
    print("-" * 40)

    print(f"Math           : {subject_averages['Math_Average']}")
    print(f"Science        : {subject_averages['Science_Average']}")
    print(f"English        : {subject_averages['English_Average']}")

    print("\nResults")
    print("-" * 40)

    print(f"Passed         : {pass_count}")
    print(f"Failed         : {fail_count}")

    print("=" * 60)


# ============================================================
# 5. ATTENDANCE ANALYSIS
# ============================================================

def attendance_analysis():

    print("\n" + "=" * 60)
    print("                  ATTENDANCE ANALYSIS")
    print("=" * 60)

    df = create_dataframe()

    if df is None:

        print("\nNo student records found.")
        return

    average_attendance = (
        df.select(
            round(avg("Attendance"), 2)
            .alias("Average_Attendance")
        )
        .collect()[0]["Average_Attendance"]
    )

    high_attendance = df.filter(
        col("Attendance") >= 75
    ).count()

    low_attendance = df.filter(
        col("Attendance") < 75
    ).count()

    print(f"Average Attendance : {average_attendance}%")
    print(f"Attendance >= 75%  : {high_attendance}")
    print(f"Attendance < 75%   : {low_attendance}")

    print("\nStudents with Low Attendance")
    print("-" * 60)

    (
        df.filter(col("Attendance") < 75)
        .select(
            "Student_ID",
            "Name",
            "Attendance"
        )
        .show(truncate=False)
    )

    print("=" * 60)


# ============================================================
# 6. STUDY HOURS ANALYSIS
# ============================================================

def study_hours_analysis():

    print("\n" + "=" * 60)
    print("                  STUDY HOURS ANALYSIS")
    print("=" * 60)

    df = create_dataframe()

    if df is None:

        print("\nNo student records found.")
        return

    average_hours = (
        df.select(
            round(avg("Study_Hours"), 2)
            .alias("Average_Hours")
        )
        .collect()[0]["Average_Hours"]
    )

    maximum_hours = (
        df.select(
            max("Study_Hours")
            .alias("Maximum_Hours")
        )
        .collect()[0]["Maximum_Hours"]
    )

    minimum_hours = (
        df.select(
            min("Study_Hours")
            .alias("Minimum_Hours")
        )
        .collect()[0]["Minimum_Hours"]
    )

    print(f"Average Study Hours : {average_hours}")
    print(f"Maximum Study Hours : {maximum_hours}")
    print(f"Minimum Study Hours : {minimum_hours}")

    print("\nStudents by Study Hours")
    print("-" * 60)

    (
        df.select(
            "Student_ID",
            "Name",
            "Study_Hours"
        )
        .orderBy(
            col("Study_Hours").desc()
        )
        .show(truncate=False)
    )

    print("=" * 60)


# ============================================================
# 7. SUBJECT ANALYSIS
# ============================================================

def subject_analysis():

    print("\n" + "=" * 60)
    print("                    SUBJECT ANALYSIS")
    print("=" * 60)

    df = create_dataframe()

    if df is None:

        print("\nNo student records found.")
        return

    averages = (
        df.select(
            round(avg("Math"), 2).alias("Math"),
            round(avg("Science"), 2).alias("Science"),
            round(avg("English"), 2).alias("English")
        )
        .collect()[0]
    )

    print("Subject Averages")
    print("-" * 40)

    print(f"Math       : {averages['Math']}")
    print(f"Science    : {averages['Science']}")
    print(f"English    : {averages['English']}")

    print("\nTop 3 Math Students")
    print("-" * 40)

    (
        df.select(
            "Student_ID",
            "Name",
            "Math"
        )
        .orderBy(col("Math").desc())
        .limit(3)
        .show(truncate=False)
    )

    print("Top 3 Science Students")
    print("-" * 40)

    (
        df.select(
            "Student_ID",
            "Name",
            "Science"
        )
        .orderBy(col("Science").desc())
        .limit(3)
        .show(truncate=False)
    )

    print("Top 3 English Students")
    print("-" * 40)

    (
        df.select(
            "Student_ID",
            "Name",
            "English"
        )
        .orderBy(col("English").desc())
        .limit(3)
        .show(truncate=False)
    )

    print("=" * 60)


# ============================================================
# 8. SPARK SQL + RDD ANALYTICS
# ============================================================

def spark_analytics():

    print("\n" + "=" * 60)
    print("                 SPARK ANALYTICS")
    print("=" * 60)

    df = create_dataframe()

    if df is None:

        print("\nNo student records found.")
        return

    df = calculate_performance(df)

    register_student_view(df)

    while True:

        print("\n" + "-" * 60)
        print("SPARK SQL & RDD MENU")
        print("-" * 60)

        print("1. Spark SQL Class Summary")
        print("2. Spark SQL Top Students")
        print("3. Spark SQL At-Risk Students")
        print("4. RDD Grade Distribution")
        print("5. RDD Pass/Fail Distribution")
        print("6. Back to Main Menu")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            print("\nSpark SQL Class Summary")
            print("-" * 60)

            sql_class_summary(spark).show(
                truncate=False
            )

        elif choice == "2":

            print("\nTop 5 Students")
            print("-" * 60)

            sql_top_students(spark).show(
                truncate=False
            )

        elif choice == "3":

            print("\nAt-Risk Students")
            print("-" * 60)

            result = sql_at_risk_students(spark)

            if result.count() == 0:

                print("No at-risk students found.")

            else:

                result.show(truncate=False)

        elif choice == "4":

            print("\nGrade Distribution")
            print("-" * 60)

            results = rdd_grade_distribution(df)

            for grade, total in sorted(results):

                print(
                    f"Grade {grade}: "
                    f"{total} student(s)"
                )

        elif choice == "5":

            print("\nPass/Fail Distribution")
            print("-" * 60)

            results = rdd_pass_fail_distribution(df)

            for status, total in sorted(results):

                print(
                    f"{status}: "
                    f"{total} student(s)"
                )

        elif choice == "6":

            print("\nReturning to main menu...")
            break

        else:

            print("\nInvalid choice. Please select 1-6.")


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        print("\n")
        print("=" * 65)
        print("       STUDENT PERFORMANCE ANALYTICS SYSTEM")
        print("=" * 65)

        print("1. Add Student")
        print("2. Find Student")
        print("3. View All Students")
        print("4. Class Statistics")
        print("5. Attendance Analysis")
        print("6. Study Hours Analysis")
        print("7. Subject Analysis")
        print("8. Spark SQL & RDD Analytics")
        print("9. Exit")

        print("=" * 65)

        choice = input("Enter choice: ").strip()

        if choice == "1":

            add_student_menu()

        elif choice == "2":

            find_student()

        elif choice == "3":

            view_all_students()

        elif choice == "4":

            class_statistics()

        elif choice == "5":

            attendance_analysis()

        elif choice == "6":

            study_hours_analysis()

        elif choice == "7":

            subject_analysis()

        elif choice == "8":

            spark_analytics()

        elif choice == "9":

            print("\nThank you for using the Student Performance Analytics System!")
            break

        else:

            print(
                f"\nInvalid choice '{choice}'. "
                "Please enter a number from 1 to 9."
            )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\n\nApplication stopped.")

    finally:

        spark.stop()