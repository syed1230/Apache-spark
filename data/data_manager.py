import csv
import os


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "data/students.csv"

COLUMNS = [
    "Student_ID",
    "Name",
    "Gender",
    "Attendance",
    "Study_Hours",
    "Math",
    "Science",
    "English"
]


# ============================================================
# INITIALIZE CSV
# ============================================================

def initialize_csv():
    """Create the CSV file with headers if it does not exist."""

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()


# ============================================================
# LOAD STUDENTS
# ============================================================

def load_students():
    """Load all students from the CSV file."""

    initialize_csv()

    with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


# ============================================================
# STUDENT EXISTS
# ============================================================

def student_exists(student_id):
    """Check whether a student ID already exists."""

    students = load_students()

    for student in students:
        if student["Student_ID"] == str(student_id):
            return True

    return False


# ============================================================
# ADD STUDENT
# ============================================================

def add_student(student):
    """Add a new student to the CSV file."""

    initialize_csv()

    if student_exists(student["Student_ID"]):
        return False

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writerow(student)

    return True