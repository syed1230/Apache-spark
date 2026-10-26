
import csv
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(__file__).resolve().parent
CSV_FILE = DATA_DIR / "students.csv"

COLUMNS = [
    "Student_ID",
    "Name",
    "Gender",
    "Attendance",
    "Study_Hours",
    "Math",
    "Science",
    "English",
]


# ============================================================
# INITIALIZE CSV
# ============================================================

def initialize_csv():
    """Create the CSV file with headers if it does not exist."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        with CSV_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()


# ============================================================
# LOAD STUDENTS
# ============================================================

def load_students():
    """Load all students from the CSV file."""

    initialize_csv()

    with CSV_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)
        return list(reader)


# ============================================================
# STUDENT EXISTS
# ============================================================

def student_exists(student_id):
    """Check whether a student ID already exists."""

    target_id = str(student_id).strip()

    return any(
        student["Student_ID"].strip() == target_id
        for student in load_students()
    )


# ============================================================
# ADD STUDENT
# ============================================================

def add_student(student):
    """Add a new student unless the ID already exists."""

    initialize_csv()

    student_id = str(student["Student_ID"]).strip()

    if not student_id or student_exists(student_id):
        return False

    new_student = {
        column: student.get(column, "")
        for column in COLUMNS
    }
    new_student["Student_ID"] = student_id

    with CSV_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writerow(new_student)

    return True
