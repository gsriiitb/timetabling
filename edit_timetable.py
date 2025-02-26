import pandas as pd
import nbformat
import os
import nbconvert
from nbconvert.preprocessors import ExecutePreprocessor

# Time Slots
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
n_days = len(days)
times = ['08:00', '08:15', '08:30', '08:45',
         '09:00', '09:15', '09:30', '09:45',
         '10:00', '10:15', '10:30', '10:45',
         '11:00', '11:15', '11:30', '11:45',
         '12:00', '12:15', '12:30', '12:45', '13:00',
         '14:00', '14:15', '14:30', '14:45',
         '15:00', '15:15', '15:30', '15:45',
         '16:00', '16:15', '16:30', '16:45',
         '17:00', '17:15', '17:30', '17:45', '18:00']

def load_excel(file_path):
    return pd.ExcelFile(file_path)

def display_sheet(sheet_df):
    print(sheet_df.to_string(index=True))
def add_entry(sheets):
    print("\nAdding entry to Exam Slot report:")
    course_code = input("Enter Course Code: ")
    title = input("Enter Course Title: ")
    batch_size = int(input("Enter Batch Size: "))  # Changed to int
    dept = input("Enter Department (CSE/ECE): ")
    day = input(f"Enter Day ({', '.join(days)}): ")
    duration = int(input("Enter Duration: "))  # Changed to int
    begin = input("Enter Begin Time: ")
    end = input("Enter End Time: ")
    preferred_time_slot = "true"

    new_exam_entry = pd.DataFrame([[course_code, title, batch_size, dept, day, duration, begin, end, preferred_time_slot]],
                                   columns=sheets["Exam Slot report"].columns)
    sheets["Exam Slot report"] = pd.concat([sheets["Exam Slot report"], new_exam_entry], ignore_index=True)
    
    print("\nExam Slot report entry added successfully.")

    # Ask the user how many entries they want to add in the Seating Report
    while True:
        try:
            num_entries = int(input("\nHow many seating report entries do you want to add (1 or 2)? "))
            if num_entries not in [1, 2]:
                print("Please enter either 1 or 2.")
            else:
                break
        except ValueError:
            print("Invalid input! Please enter a number (1 or 2).")

    for i in range(num_entries):
        print(f"\nAdding Seating Report entry {i+1}:")
        room = input("Enter Room Name: ")
        room_type = input("Enter Room Type: ")
        room_number = int(input("Enter Room Number: "))
        capacity = int(input("Enter Room Capacity: "))  # Changed to int
        assigned_students = int(input("Enter Number of Students Assigned: "))  # Changed to int

        new_seating_entry = pd.DataFrame([[course_code, title, room, room_type, batch_size, room_number, capacity, assigned_students]],
                                          columns=sheets["Seating Report"].columns)
        sheets["Seating Report"] = pd.concat([sheets["Seating Report"], new_seating_entry], ignore_index=True)

    print("\nSeating Report entries added successfully.")
    
    return sheets


def remove_entry(sheets):
    course_codes = sheets["Exam Slot report"]["Course Code"].unique()
    
    if len(course_codes) == 0:
        print("No courses available to remove.")
        return sheets
    
    print("Select a course to remove:")
    for i, code in enumerate(course_codes):
        print(f"{i+1}. {code}")
    
    try:
        choice = int(input("Enter choice: "))
        if 1 <= choice <= len(course_codes):
            course_code = course_codes[choice - 1]
            sheets["Exam Slot report"] = sheets["Exam Slot report"][sheets["Exam Slot report"]["Course Code"] != course_code].reset_index(drop=True)
            sheets["Seating Report"] = sheets["Seating Report"][sheets["Seating Report"]["Course Code"] != course_code].reset_index(drop=True)
        else:
            print("Invalid choice!")
    except ValueError:
        print("Invalid input! Enter a number.")
    
    return sheets

def save_excel(df_dict, file_path):
    with pd.ExcelWriter(file_path) as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print("Changes saved to", file_path)

def execute_notebook(notebook_path, sheets):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Inject the sheets dictionary into the notebook execution
    nb.cells.insert(0, nbformat.v4.new_code_cell("sheets = " + repr(sheets)))
    
    ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})


    
def main():
    file_path = "timetable_report.xlsx"
    xl = load_excel(file_path)
    sheets = {sheet: xl.parse(sheet) for sheet in xl.sheet_names if sheet != "Clash report"}
    
    while True:
        print("\nMenu:")
        print("1. Add course entry")
        print("2. Remove course entry")
        print("3. Execute plot.ipynb")
        print("4. Save changes")
        print("5. Exit")
        
        choice = input("Enter choice: ")
        if choice == '1':
            sheets = add_entry(sheets)
        elif choice == '2':
            sheets = remove_entry(sheets)
        elif choice == '3':
            execute_notebook("plot.ipynb", sheets)
        elif choice == '4':
            save_excel(sheets, file_path)
        elif choice == '5':
            print("Exiting without saving.")
            break
        else:
            print("Invalid choice! Try again.")

if __name__ == "__main__":
    main()
