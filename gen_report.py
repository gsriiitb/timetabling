import os
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pickle
from pandas import ExcelWriter, DataFrame
from PyPDF2 import PdfMerger
from literals import Literals as L


class TimetableReport(object):
    def __init__(self, title:str, for_exams:bool, soln_num:int, timetable_df:DataFrame,
                 time_slots:list, rooms:list, overlapping_pairs:list=None):

        self.timetable_df = timetable_df
        self.soln_num = soln_num
        self.title, self.for_exams = title, for_exams
        self.time_slots = time_slots
        self.rooms, self.capacities = tuple(zip(*sorted(rooms, key=itemgetter(1), reverse=True)))
        self.n_times, self.n_rooms = len(self.time_slots), len(self.rooms)
        self.primary_key_courses = [L.CODE, L.TITLE, L.FACULTY1, L.FACULTY2]
        self.unique_courses = \
            self.timetable_df[self.primary_key_courses].drop_duplicates()
        self.n_courses = len(self.unique_courses)
        if overlapping_pairs is not None:
            self.overlapping_pairs = sorted(overlapping_pairs, key=itemgetter(2,0,1), 
                                            reverse=True)

    @staticmethod
    def output_file(f:str): return os.path.join(os.getcwd(), "../output", 
                                                f.replace(" ", "_"))

    @staticmethod
    def get_cmap(n, name='hsv'):
        # Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        # RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.get_cmap(name, n)

class ExcelReport(TimetableReport):
    REPORT = "Report"
    MET_DAY_PREF, MET_TIME_PREF = "Met Day Pref?", "Met Time Pref?"
    COMMON, SAME_DAY, CLASH = 'Common #', 'Same Day?', 'Clash?'

    def __init__(self, title:str, for_exams:bool, soln_num:int,
                 timetable_df:DataFrame, time_slots:list, rooms:list,
                 overlapping_pairs:list):
        super().__init__(title, for_exams, soln_num, timetable_df,
                         time_slots, rooms, overlapping_pairs)
        self.report_file = self.output_file(f'{self.title} {self.REPORT} {self.soln_num}.xlsx')

    @staticmethod
    def bool_val(b:bool): return L.YES if b else L.NO

    def get_courses_dfs(self):
        def met_day_pref(prefs:str, day_str:str) -> str:
            return self.bool_val((day_str in prefs) or (prefs == L.ANY_DAY))

        def met_time_pref(prefs:str, begin_str:int) -> str:
            return self.bool_val((begin_str in prefs) or (prefs == L.ANY_TIME))

        def make_slots_df():
            slots_cols = [L.CODE, L.TITLE, L.SIZE, L.FACULTY1, L.FACULTY2,
                          L.ISCORE, L.BATCH, L.DAY, L.BEGIN, L.END,
                          L.PREF_DAYS, self.MET_DAY_PREF,
                          L.PREF_TIMES, self.MET_TIME_PREF]
            slots_df = self.timetable_df[slots_cols[:-3]+[L.PREF_TIMES]]
            met_day_prefs = slots_df.apply(lambda r: met_day_pref(r[L.PREF_DAYS], r[L.DAY]),
                                           axis=1)
            slots_df[self.MET_DAY_PREF] = met_day_prefs
            met_time_prefs = slots_df.apply(lambda r: met_time_pref(r[L.PREF_TIMES], r[L.BEGIN]),
                                            axis=1)
            slots_df[self.MET_TIME_PREF] = met_time_prefs
            return slots_df[slots_cols]

        def make_seating_df():
            seating_cols = [L.CODE, L.TITLE, L.SIZE, L.ISCORE, L.BATCH, 
                            L.ROOM_NUM, L.CAPACITY, L.SEATING]
            seating_df = self.timetable_df[seating_cols]
            return seating_df

        return make_slots_df(), make_seating_df()

    def get_clash_df(self, overlapping_pairs:list):
        def is_clash(begin1:str, end1:str, begin2:str, end2:str):
            b1, e1 = self.time_slots.index(begin1), self.time_slots.index(end1)
            b2, e2 = self.time_slots.index(begin2), self.time_slots.index(end2)
            return ((b1 <= b2 <= e1) or (b1 <= e2 <= e1) or 
                    (b2 <= b1 <= e2) or (b2 <= e1 <= e2))

        code_col, title_col, f1_col, f2_col = self.primary_key_courses
        clash_cols = [L.BATCH, L.CODE, L.TITLE, L.DAY, L.BEGIN, L.END]
        course1_cols = [f'{h}-1' for h in clash_cols]
        course2_cols = [f'{h}-2' for h in clash_cols]
        all_clash_cols = (course1_cols + course2_cols +
                          [self.COMMON, self.SAME_DAY, self.CLASH])
        clash_df_dict = {h: [] for h in all_clash_cols}

        for c1, c2, n in overlapping_pairs:
            code1, title1, f11, f12 = c1
            c1_df = self.timetable_df[(self.timetable_df[code_col]==code1) &
                                        (self.unique_courses[title_col]==title1) &
                                        (self.unique_courses[f1_col]==f11) &
                                        (self.unique_courses[f2_col]==f12)][clash_cols]
            code2, title2, f21, f22 = c2
            c2_df = self.timetable_df[(self.timetable_df[code_col]==code2) &
                                        (self.unique_courses[title_col]==title2) &
                                        (self.unique_courses[f1_col]==f21) &
                                        (self.unique_courses[f2_col]==f22)][clash_cols]
            for index1, row1 in c1_df.iterrows():
                for index2, row2 in c2_df.iterrows():
                    for h in clash_cols: clash_df_dict[f'{h}-1'].append(row1[h])
                    for h in clash_cols: clash_df_dict[f'{h}-2'].append(row2[h])
                    clash_df_dict[self.COMMON].append(n)
                    same_day = (row1[L.DAY] == row2[L.DAY])
                    clash_df_dict[self.SAME_DAY].append(self.bool_val(same_day))
                    clashing = is_clash(row1[L.BEGIN], row1[L.END],
                                        row2[L.BEGIN], row2[L.END])
                    clash_df_dict[self.CLASH].append(self.bool_val(same_day and clashing))

        clash_df = DataFrame.from_dict(clash_df_dict).reset_index(drop=True)
        return clash_df[all_clash_cols]

    def get_core_clash_df(self):
        core_clash_pairs = []
        for b, clist in self.gg.batch_cores.items():
            for i, c1 in enumerate(clist):
                for c2 in clist[(i+1):]:
                    s1 = self.gg.courses.enrollments[c1]
                    s2 = self.gg.courses.enrollments[c2]
                    n = len(s1.intersection(s2))
                    core_clash_pairs.append((c1, c2, n))
        return self.get_clash_df(core_clash_pairs)

    def exam_clash_df(self):
        return self.get_clash_df(self.overlapping_pairs)

    def get_noncore_clash_df(self):
        return self.get_clash_df(self.gg.non_core_overlaps)

    def gen_report(self):
        with ExcelWriter(self.report_file) as writer:
            slots_df, seating_df = self.get_courses_dfs()
            slots_df.to_excel(writer, sheet_name='Exam Slots', index=False)
            seating_df.to_excel(writer, sheet_name='Seating', index=False)
            if not self.for_exams:
                pass
                # core_clash_df = self.get_core_clash_df()
                # core_clash_df.to_excel(writer, sheet_name='Core Clashes',
                                    #    index=False)
                # noncore_clash_df = self.get_noncore_clash_df()
                # noncore_clash_df.to_excel(writer, sheet_name='Non-Core Clashes',
                                        #   index=False)
            else:
                pass
                clash_df = self.exam_clash_df()
                clash_df.to_excel(writer, sheet_name='Clashes', index=False)

        print(f'Excel report {self.soln_num} generated successfully!')


class Plotter(TimetableReport):
    PLOT = 'Chart'

    def __init__(self, title:str, for_exams:bool, soln_num:int, 
                 timetable_df:DataFrame, time_slots:list, rooms:list):

        super().__init__(title, for_exams, soln_num, timetable_df,
                         time_slots, rooms)
        self.ticks, self.partitions = self.ticks_and_partitions()

    def ticks_and_partitions(self):
        s = sum(self.capacities)
        partitions = np.cumsum(self.capacities) / s
        ticks = (partitions - (self.capacities / (2*s)))
        partitions = np.insert(partitions, 0, 0)
        return ticks, partitions

    def pdf_file(self, day:str):
        return self.output_file(f'{self.title} {self.PLOT} {day} {self.soln_num}.pdf')

    def contiguous_occupied_fractions(self, r, b, e,
                                      occupied_fractions_up, occupied_fractions_down):
        base_down = occupied_fractions_down[r, b]
        down_clear = all([(occupied_fractions_down[r, t] == base_down) 
                          for t in range(b, (e + 1))])
        return (occupied_fractions_down if down_clear else occupied_fractions_up)

    def add_patch(self, ax, clr, allocation,
                  occupied_fractions_up=None, occupied_fractions_down=None):
        b = self.time_slots.index(allocation[L.BEGIN])
        e = self.time_slots.index(allocation[L.END])
        n_slots = (e - b)
        r = self.rooms.index(allocation[L.ROOM_NUM])
        code, enrolled = allocation[L.CODE], allocation[L.SIZE]
        seated, capacity = allocation[L.SEATING], allocation[L.CAPACITY]
        seating_fraction = (seated / capacity)
        if self.for_exams:
            occupied_fractions = self.contiguous_occupied_fractions(r, b, e,
                                                                    occupied_fractions_up,  
                                                                    occupied_fractions_down)
            offset = occupied_fractions[r, b]
        else:
            offset = 0
        gap = self.partitions[r+1] - self.partitions[r]
        occupancy = gap*seating_fraction
        # width, height = duration, (gap*seating_fraction)
        room_start = (((self.partitions[r] + gap*offset)
                       if (occupied_fractions is occupied_fractions_down) else
                       (self.partitions[r+1] - gap*offset - occupancy))
                      if self.for_exams else self.partitions[r])
        anchor, width, height = (((b, room_start), n_slots, occupancy)
                                 if self.for_exams else 
                                 ((room_start, b), occupancy, n_slots))
        rect = mpatches.Rectangle(anchor, width, height, alpha=0.8, zorder=5,
                                  linewidth=1, edgecolor=clr, facecolor=clr)
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width()/2.0
        cy = ry + rect.get_height()/2.0
        smaller_letters = (width <= 1) or (height <= 0.02)
        fs = (8 if smaller_letters else 10)
        wt = 'bold' # ('bold' if smaller_letters else 'normal')
        text = (f'{code} ({seated}/{enrolled})' if self.for_exams else
                f'{code} ({enrolled})')
        ax.annotate(text, (cx, cy), color='black', weight=wt,
                    fontsize=fs, ha='center', va='center', zorder=6,
                    rotation=(0 if self.for_exams else 90))
        if self.for_exams:
            for t in range(b, (e + 1)):
                occupied_fractions[r, t] += seating_fraction
        return code

    def gen_day_plot(self, day:str, allocation_df:DataFrame):
        occupied_fractions_up, occupied_fractions_down = None, None
        if self.for_exams:
            occupied_fractions_up = np.zeros((self.n_rooms, self.n_times))
            occupied_fractions_down = np.zeros((self.n_rooms, self.n_times))

        fig = plt.figure(num=None, figsize=(8, 10), dpi=120, facecolor='w')
        ax = fig.add_subplot(111)
        ax.grid(axis=("x" if self.for_exams else "y"))
        axx = ax.twiny()
        if self.for_exams: axy = ax.twinx()

        time_ticks = range(self.n_times)
        ax.set_title(f'{self.title} for {day}', fontweight="bold")
        label_font = {'fontsize': 8, 'fontweight': 'bold'}
        if not self.for_exams:
            ax.set_xlabel('Capacity', fontdict=label_font)
            ax.set_xticks(self.ticks, self.capacities)
            ax.set_ylabel('Time', fontdict=label_font)
            ax.set_yticks(time_ticks, self.time_slots)
            axx.set_xlabel('Room', fontdict=label_font)
            axx.set_xticks(self.ticks, self.rooms, rotation=270)
        else:
            ax.set_xlabel('Time', fontdict=label_font)
            ax.set_xticks(time_ticks, self.time_slots)
            ax.set_ylabel('Room', fontdict=label_font)
            ax.set_yticks(self.ticks, self.rooms)
            axx.set_xlabel('Time', fontdict=label_font)
            axx.set_xticks(time_ticks, self.time_slots)
            axy.set_ylabel('Capacity', fontdict=label_font)
            axy.set_yticks(self.ticks, self.capacities)

        for l in self.partitions:
            if self.for_exams:
                plt.axhline(y=l, color='black', linestyle='-')
            else:
                plt.axvline(x=l, color='black', linestyle='-')
        cmap = self.get_cmap(self.n_courses)
        course_legend_handles, day_courses = [], []

        code_col, title_col, f1_col, f2_col = self.primary_key_courses
        for index, allocation in allocation_df.iterrows():
            course_key = tuple(allocation[v] for v in self.primary_key_courses)
            code, title, f1, f2 = course_key
            course_id = self.unique_courses[(self.unique_courses[code_col]==code) &
                                            (self.unique_courses[title_col]==title) &
                                            (self.unique_courses[f1_col]==f1) &
                                            (self.unique_courses[f2_col]==f2)].index[0]
            clr = cmap(course_id)
            code = self.add_patch(ax, clr, allocation,
                                  occupied_fractions_up,
                                  occupied_fractions_down)
            if course_id not in day_courses:
                day_courses.append(course_id)
                course_legend_handles.append(mpatches.Patch(color=clr, label=code))

        if not self.for_exams: plt.gca().invert_yaxis()
        plt.legend(handles=course_legend_handles)
        pdf_file = self.pdf_file(day)
        plt.savefig(pdf_file)
        return pdf_file

    def gen_report(self):
        merger = PdfMerger()
        by_day = self.timetable_df.groupby(L.DAY)
        day_files = []
        for day, entries in by_day:
            day_files.append(self.gen_day_plot(day, entries))
            merger.append(day_files[-1])
        merger.write(self.output_file(f'{self.title} {self.PLOT} {self.soln_num}.pdf'))
        merger.close()
        for f in day_files: os.remove(f)
        print(f'Graphical report {self.soln_num} generated successfully!')


if __name__ == '__main__':
    with open(TimetableReport.output_file('other_info.pkl'), 'rb') as other_info:
        info = pickle.load(other_info)
    timetable_df = pd.read_pickle(TimetableReport.output_file('Term_II_24-25_Midsem_Exam_Timetable_0.pkl'))
    excel = ExcelReport('Test Title', True, 1, timetable_df,
                        info['times'], info['rooms'], info['overlapping_pairs'])
    excel.gen_report()
    plotter = Plotter('Test Title', True, 1, timetable_df,
                      info['times'], info['rooms'])
    plotter.gen_report()