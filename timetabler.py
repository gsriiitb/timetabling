import numpy as np
import pandas as pd
import pickle
import gurobipy as gp
from gurobipy import GRB
from literals import Literals as L
import logging
from abc import ABC, abstractmethod


from globals import ProblemParameters as PP, Globals as GG, add_section_label
from gen_report import TimetableReport, ExcelReport, Plotter


class Timetabler(ABC):
    LINK = 'link'

    # General Comments:
    # Courses, rooms, times, days, etc. are all referred to only by their respective ids, where
    # the id is an index into an array or a dataframe
    # So in all the methods --- c,b,d,r etc. are ids of courses, begin times, days and rooms
    def __init__(self, params:PP):
        self.pp, self.gg = params, GG(params)
        self.model = gp.Model(self.pp.title)

        self.objectives = {self.pp.SAME_DAY_OVERLAP: 0,
                           self.pp.DAY_AVAILABILITY: 0,
                           self.pp.TIME_AVAILABILITY: 0,
                          }
        self.objectives.update({self.pp.COURSE_SPLITS: 0}
                               if self.pp.for_exams else
                                {self.pp.NON_CORE_CLASH: 0,
                                 self.pp.DAY_GAP_BETWEEN_CLASSES: 0})
        
        # The timetables produced by this timetabler will be stored in two forms
        # Tuple ('occupancy', 'occupancy_ind'). 'occupancy' will be a numpy array 
        # showing when, where and how many of
        # each course are occupying at any time / day
        # 'occupancy_ind' will be the 0-1 version of the 'occupancy'
        # 'timetable' will be the timetable as a pandas dataframe 
        self.timetables = []

        (self.vbSlot, self.viSeating, self.vbSeating_ind, 
         self.viLink, self.vbLink_ind) = self.create_variables()

    def print_problem_stats(self):
        logging.info(f'Exams for {len(self.gg.courses_idx)} Courses with total strength '
              f'{sum(self.gg.course_sizes)}')
        for c in self.gg.courses_idx:
            attrs = self.gg.courses.get_attrs(c, [L.CODE, L.SIZE, L.EXAM_TYPE, L.SLOTS])
            logging.info('  '.join([str(a) for a in attrs]) + ' Hrs')

        logging.info(f'To be scheduled across {len(self.gg.rooms_idx)} Rooms with Total Capacity '
            f'{sum(self.gg.rooms.capacities)} and minimum capacity {self.pp.min_capacity}, '
            f'over {len(self.gg.days_idx)} Days')
        for r in range(self.gg.rooms.n_rooms):
            attrs = self.gg.rooms.get_attrs(r, [L.ROOM_NUM, L.CAPACITY, L.CATEGORY])
            logging.info('  '.join([str(a) for a in attrs]))

    def create_variables(self):
        vbSlot = self.model.addVars([(c,d,b) 
                                     for c in self.gg.courses_idx
                                     for d in self.gg.days_idx
                                     for b in self.gg.valid_begins[(c,d)]
                                     ],
                                    vtype=GRB.BINARY, name=L.SLOT)

        viSeating = self.model.addVars([(c,r) 
                                        for c in self.gg.courses_idx
                                        for r in self.gg.matching_rooms[c]],
                                       vtype=GRB.INTEGER,
                                       lb=0,
                                       name=L.SEATING)
        
        vbSeating_ind = self.model.addVars([(c,r)
                                            for c in self.gg.courses_idx
                                            for r in self.gg.matching_rooms[c]],
                                           vtype=GRB.BINARY,
                                           name="seating_ind")

        # Conditions relating 'seating and 'seating_ind' vairables
        for c in self.gg.courses_idx:
            for r in self.gg.matching_rooms[c]:
                self.model.addConstr((vbSeating_ind[c,r] == 1) >> 
                                          (viSeating[c,r] >= self.gg.min_counts[c]),
                                     name=f'seating_ind_1_{c},{r}')
                self.model.addConstr((vbSeating_ind[c,r] == 0) >> 
                                          (viSeating[c,r] == 0),
                                     name=f'seating_ind_0_{c},{r}')

        viLink = self.model.addVars([(c,d,b,r)
                                     for c in self.gg.courses_idx
                                     for d in self.gg.days_idx
                                     for b in self.gg.valid_begins[(c,d)]
                                     for r in self.gg.matching_rooms[c]],
                                    vtype=GRB.INTEGER,
                                    lb=0, name=self.LINK)

        vbLink_ind = self.model.addVars([(c,d,b,r)
                                         for c in self.gg.courses_idx
                                         for d in self.gg.days_idx
                                         for b in self.gg.valid_begins[(c,d)]
                                         for r in self.gg.matching_rooms[c]],
                                        vtype=GRB.BINARY, name="link_ind")

        for c in self.gg.courses_idx:
            for d in self.gg.days_idx:
                for b in self.gg.valid_begins[(c,d)]:
                    for r in self.gg.matching_rooms[c]:
                        self.model.addConstr((vbLink_ind[c,d,b,r] == 
                                                gp.and_(vbSlot[c,d,b], 
                                                        vbSeating_ind[c,r])),
                                             name='link_ind_{c}_{d}_{b}_{r}')
                        self.model.addConstr(((vbLink_ind[c,d,b,r] == 1) >> 
                                                (viLink[c,d,b,r] == viSeating[c,r])),
                                             name=f"Link_1_{c}_{d}_{b}_{r}")
                        self.model.addConstr(((vbLink_ind[c,d,b,r] == 0) >> 
                                                    (viLink[c,d,b,r] == 0)),
                                             name=f"Link_2_{c}_{d}_{b}_{r}")

        return vbSlot, viSeating, vbSeating_ind, viLink, vbLink_ind        


    @staticmethod
    def compliment_dims(dims:tuple): return (i for i in range(5) if i not in dims)

    def error_check(self, errors:list, check_qty:np.ndarray,
                    dims:tuple, message:str, is_hard:bool=True):
        # Reports the violations of the kind given in 'message'
        # The 'check_qty' is a numpy array, typically an aggregate of 'self.occupancy_ind'
        # or 'self.occupancy' depending on the context, over dims 'dims'
        # 'errs' is a list of tuples of the form 
        # (tuple of indexes over dims + erroneous value)
        errs = [((int(v) for v in t) + (check_qty[*t],))
                for t in zip(*errors)]
        return ((message, self.compliment_dims(dims), errs, is_hard) 
                if len(errs) > 0 else None)

    def ensure_course_strength_is_covered(self, verify:bool):
        for c in self.gg.courses_idx:
            self.model.addConstr((self.viSeating.sum(c,'*') == self.gg.course_sizes[c]),
                                 name=f'enrollment_count_{c}')

    @abstractmethod
    def ensure_session_slots_are_covered(self, verify:bool): pass

    @abstractmethod
    def ensure_capacities_are_enough(self, verify:bool): pass
 
    @abstractmethod
    def minimize_same_day_overlapping_courses(self, verify:bool): pass

    @abstractmethod
    def accommodate_day_availability(self, verify:bool): pass

    @abstractmethod
    def accommodate_time_availability(self, verify:bool): pass

    @abstractmethod
    def same_day_overlap_objective(self, slack): pass

    @abstractmethod
    def day_availability_objective(self, slack): pass

    @abstractmethod
    def time_availability_objective(self, slack): pass

    @abstractmethod
    def set_other_objectives(self, slacks:dict): pass

    def set_objective(self, slacks:dict):
        for k, fn in [(self.pp.SAME_DAY_OVERLAP, self.same_day_overlap_objective),
                      (self.pp.DAY_AVAILABILITY, self.day_availability_objective),
                      (self.pp.TIME_AVAILABILITY, self.time_availability_objective)
                      ]:
            slack = slacks.pop(k)
            self.objectives[k] = fn(slack)
        self.set_other_objectives(slacks)

        full_objective = gp.quicksum((w * self.objectives[label])
                                     for label, w in 
                                        self.pp.objective_weights.items()
                                    )
        self.model.setObjective(full_objective, GRB.MINIMIZE)
        self.model.printStats()


    @property
    def constraint_fns(self):
        return [
            self.ensure_course_strength_is_covered,
            self.ensure_session_slots_are_covered,
            self.ensure_capacities_are_enough,
            self.minimize_same_day_overlapping_courses,
            self.accommodate_day_availability,
            self.accommodate_time_availability
        ]

    def build_verify(self, verify:bool):
        logging.info(f'Constraints formed')
        errors_slacks = [f(verify=verify) for f in self.constraint_fns]
        if verify:
                return [e for e in errors_slacks if e is not None]
        else:
            self.set_objective({k: s 
                                for ks in errors_slacks
                                if ks is not None
                                for k, s in ks.items()
                                })
            logging.info(f'Built objective function')

    def run_optimizer(self):
        self.model.setParam('MIPGap', self.pp.gap)
        self.model.setParam('Timelimit', self.pp.time_limit)
        logging.info(f'Running solver ...')
        self.model.optimize()
        logging.info(f'Solution Count: {self.model.SolCount}')
        return min(self.pp.n_solutions_to_report, self.model.SolCount)

    def extract_timetable(self, n_solutions_to_get:int):
        def parse_varname(s:str, prefix:str):
            x = f'{prefix}['
            return  (int(n) for n in tuple(s.lstrip(x)[:-1].split(',')))

        def make_pref_days(c:int) -> str:
            preferred_days_idx = self.gg.courses.get_preferred_days(c)
            any_day = (len(preferred_days_idx) == self.gg.time_slots.n_days)
            preferred_days = (', '.join([self.gg.time_slots.day(d)
                                         for d in preferred_days_idx])
                              if not any_day else L.ANY_DAY)
            return preferred_days

        def make_pref_times(c:int) -> str:
            preferred_times_idx = self.gg.courses.get_preferred_times(c)
            any_time = (len(preferred_times_idx) == self.gg.time_slots.n_pref_times)
            preferred_times = (', '.join([self.gg.time_slots.slot(b)
                                          for b in preferred_times_idx])
                               if not any_time else L.ANY_TIME)
            return preferred_times
        
        for sol_num in range(n_solutions_to_get):
            self.model.params.SolutionNumber = sol_num
            var_dict = {L.SLOT: {}, L.SEATING: {}}
            for variable in self.model.getVars():
                for k, c_dict in var_dict.items():
                    if variable.VarName.startswith(f'{k}[') and variable.Xn > 0.5:
                        indices, val = parse_varname(variable.VarName, k), int(variable.Xn)
                        if k == L.SEATING:
                            c, r = indices
                            if c in c_dict: c_dict[c].append((r, val))
                            else: c_dict[c] = [(r, val)]
                        else:
                            c, d, b = indices
                            if c in c_dict: c_dict[c].append((d, b, val))
                            else: c_dict[c] = [(d, b, val)]

            col_names = [L.CODE, L.TITLE, L.SIZE, L.FACULTY1, L.FACULTY2, L.ISCORE,
                         L.BATCH, L.DAY, L.BEGIN, L.END, L.ROOM_NUM, L.CAPACITY, L.SEATING,
                         L.PREF_DAYS, L.PREF_TIMES]
            df_dict = {h: [] for h in col_names}    
            col_types = [str, str, int, str, str, bool, str, str, str, str, str, int, int,
                         str, str]
            col_types = dict(zip(col_names, col_types))
            for c in self.gg.courses_idx:
                course_attrs = self.gg.courses.get_attrs(c, col_names[:7])
                code = add_section_label(self.gg.sections_and_groups, course_attrs[0], c)
                for d, b, val in var_dict[L.SLOT][c]:
                     for r, v in var_dict[L.SEATING][c]:
                         df_dict[L.CODE].append(code)
                         for i, attr_val in enumerate(course_attrs):
                             if i > 0:
                                df_dict[col_names[i]].append(attr_val)
                         df_dict[L.DAY].append(self.gg.day(d))
                         df_dict[L.BEGIN].append(self.gg.tm(b))
                         n_slots = self.gg.course_slots[c]
                         end_time = self.gg.tm(self.gg.time_slots.end_time(b, n_slots))
                         df_dict[L.END].append(end_time)
                         df_dict[L.ROOM_NUM].append(self.gg.room(r))
                         df_dict[L.CAPACITY].append(self.gg.capacities[r])
                         df_dict[L.SEATING].append(v if self.pp.for_exams else val)
                         df_dict[L.PREF_DAYS].append(make_pref_days(c))
                         df_dict[L.PREF_TIMES].append(make_pref_times(c))
            solution = pd.DataFrame.from_dict(df_dict).astype(col_types).reset_index(drop=True)
            logging.info(f'Generated Timetable')
            pkl_file = TimetableReport.output_file(f'{self.pp.title}_Timetable_{sol_num}.pkl')
            solution.to_pickle(pkl_file)
            logging.info(f'Saved timetable {sol_num} as df into {pkl_file}')
            self.timetables.append(solution)

    def make(self):
        self.build_verify(verify=False)
        n_solutions = self.run_optimizer()
        logging.info(f'Retriving {n_solutions} Solutions')
        if n_solutions > 0:
            self.extract_timetable(n_solutions)
            rooms_capacities = list(zip(self.gg.rooms_list, self.gg.capacities))
            with open(TimetableReport.output_file('other_info.pkl'), 'wb') as other_info:
                pickle.dump({'times': self.gg.time_slots.times,
                            'rooms': rooms_capacities,
                            'overlapping_pairs': self.gg.overlapping_pairs_with_attrs},
                            other_info)
            for i, timetable in enumerate(self.timetables):
                excel_report = ExcelReport(self.pp.title, self.pp.for_exams, i, timetable,
                                           self.gg.time_slots.times, rooms_capacities,
                                           self.gg.overlapping_pairs_with_attrs)
                excel_report.gen_report()
                pdf_report = Plotter(self.pp.title, self.pp.for_exams, i, timetable,
                                     self.gg.time_slots.times, rooms_capacities)
                pdf_report.gen_report()
        else:
            logging.info('Failed to find any feasible solutions.')

    def verify(self, df_file:str, occupancy_file:str):
        self.timetable = pd.read_csv(df_file)
        occupancy = np.loadtxt(occupancy_file)
        self.occupancy = occupancy['occupancy']
        self.occupancy_ind = occupancy['occupancy_ind']
        errors = self.build_verify(verify=True)
        if len(errors) > 0:
            pass
        else:
            logging.info('No issues found in the timetable. Congratulations.')
