class Literals():
    
    ENROLLMENTS_MASTER_FILE = "enrollments_master.pkl"
    EXCEL_LOCK_KEY = "timetabling@iiitb"

    # ============= Source Literals ========================
    SDATA_SHEET = "Course Roster Report"
    CORE = 'CORE'
    COURSE = "Course"
    COURSE_TYPE = "OTHER/REGULAR-ELECTIVE/REGULAR-CORE"
    OTHER = 'OTHER'
    SRC_BATCH = "Batch"
    SRC_FACULTY1 = "Primary Faculty"
    SRC_FACULTY2 = "Alternate Faculty"
    STUDENT_ID = "Student ID"

    # ================= Target Literals ======================
    COURSES_MASTER_FILE = "courses_master.xlsx"
    COURSES = "Courses" # Sheet name
    BATCH = "Major Batch"
    CODE = "Code"
    SIZE = "Size"
    FACULTY1 = "Faculty1"
    FACULTY2 = "Faculty2"
    ISCORE = "Is Core"
    TITLE = "Title"


    CLASS_PREF = "Class Preferences" # sheet name
    CLASS_TYPE = "Class Type"
    COMP = "Computer Lab"
    LAB = "Physical Lab"
    LECTURE = "Lecture"
    TUTORIAL = "Tutorial"
    CLASS_TYPE_RANGE = "E3:E"
    CLASS_TYPE_NAMES = "ClassTypes"
    NEED_PROJ = "Need Projector"
    NEED_PROJ_RANGE = "F3:F"
    NEED_WACOM = "Need Wacom"
    NEED_WACOM_RANGE = "G3:G"
    NEED_TA = "Need TAs"
    NEED_TA_RANGE = "H3:H"
    DURATION = "Duration (Hrs)"
    SLOTS = 'Slots'
    CLASS_DURATION_RANGE = "I3:I"
    CLASS_DURATION_TO_SLOTS_RANGE = "CDurationToSlots"
    CLASSES_PER_WEEK = "Classes Per Week"
    CLASSES_PER_WEEK_RANGE = "J3:J"
    CLASSES_PER_WEEK_NAMES = "ClassesPerWeek"
    CLASS_PREF_RANGES = "F3:H,K3:O,Q3:Y"


    EXAM_PREF = "Exam Preferences" # sheet name
    NO_EXAM = "** No Exam **"
    EXAM_TYPE = "Exam Type"
    PAPER_PEN = "Paper-Pen"
    ON_SYSTEM = "On System"
    LAB_EXAM = "Lab Exam"
    OTHERS = "Others"
    EXAM_TYPE_RANGE = "E3:E"
    EXAM_TYPE_NAMES = "ExamTypes"
    EXAM_DURATION_RANGE = "F3:F"
    EXAM_DURATION_TO_SLOTS_RANGE = "EDurationToSlots"
    EXAM_PREF_RANGES = "G3:L,N3:T"

    SESSION_DURATION_NAMES = "SessionDurations"


    CONFIG = "Config" # sheet name
    PREF_NAMES = "Pref"
    EXAM_TIMES = "ExamTimes"
    EXAM_TIMES_FINE = "ExamTimesFine"
    CLASS_TIMES = "ClassTimes"
    CLASS_TIMES_FINE = "ClassTimesFine"
    REG_LUNCH_TIME = "CLunchTime"
    EXAM_LUNCH_TIME = "ELunchTime"
    EXAM_DAYS = "ExamDays"
    CLASS_DAYS = "ClassDays"
    YES = "Yes"
    NO = "No"
    FACULTY_NAMES = "Faculty"
    COURSE_GROUPS = 'CourseGroups'


    # classrooms_master.xlsx
    CLASSROOMS_MASTER_FILE = "classrooms_master.xlsx"
    CLASSROOMS = "Classrooms" # sheet name
    ROOM_TYPE = "Room Type"
    FOR_EVENTS = "For Events"
    FOR_EXAMS = "For Exams"
    ROOM_NUM = "Room Num"
    EXAM_CAPACITY = "Exam Capacity"
    CAPACITY = "Capacity"
    CATEGORY = "Category"
    HAS_PROJ = "Has Projector"
    HAS_WACOM = "Has Wacom"


    BLANK = "----------" 
    DAY, BEGIN, END, SLOT, SEATING = 'Day', 'Begin', 'End', 'Slot', 'Seating'
    ANY_DAY, ANY_TIME = "Any Day", "Any Time"
    PREF_DAYS, PREF_TIMES = 'Pref Days', 'Pref Times'