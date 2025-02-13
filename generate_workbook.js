
// Ensure you have installed the xlsx package: npm install xlsx
const XLSX = require('xlsx');

// Returns dummy data for the Clash Report sheet.
function clash_reprt() {
    // ...existing code if any...
    return [
        ['Event', 'Time', 'Location'],
        ['Exam Clash 1', '09:00', 'Room A'],
        ['Exam Clash 2', '11:00', 'Room B']
    ];
}

// Returns dummy data for the Exam Slot sheet.
function make_exam_slot() {
    // ...existing code if any...
    return [
        ['Slot', 'Exam Name', 'Duration'],
        ['Morning Slot', 'Math', '2 hrs'],
        ['Afternoon Slot', 'English', '1.5 hrs']
    ];
}

function generateWorkbook() {
    const workbook = XLSX.utils.book_new();
    
    const clashData = clash_reprt();
    const examSlotData = make_exam_slot();
    
    const clashSheet = XLSX.utils.aoa_to_sheet(clashData);
    const examSheet = XLSX.utils.aoa_to_sheet(examSlotData);
    
    XLSX.utils.book_append_sheet(workbook, clashSheet, 'Clash Report');
    XLSX.utils.book_append_sheet(workbook, examSheet, 'Exam Slot');
    
    XLSX.writeFile(workbook, 'report_workbook.xlsx');
}

generateWorkbook();
