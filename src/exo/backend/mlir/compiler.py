import re
from pathlib import Path

from ...core.LoopIR import LoopIR, LoopIR_Do
from ..mem_analysis import MemoryAnalysis
from ..parallel_analysis import ParallelAnalysis
from ..prec_analysis import PrecisionAnalysis
from ..win_analysis import WindowAnalysis
from .ir_gen import IRGenerator


def sanitize_str(s):
    return re.sub(r"\W", "_", s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


operations = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "%": lambda x, y: x % y,
}


class LoopIR_SubProcs(LoopIR_Do):
    def __init__(self, proc):
        self._subprocs = set()
        if proc.instr is None:
            super().__init__(proc)

    def result(self):
        return self._subprocs

    # to improve efficiency
    def do_e(self, e):
        pass

    def do_s(self, s):
        if isinstance(s, LoopIR.Call):
            self._subprocs.add(s.f)
        else:
            super().do_s(s)


def find_all_subprocs(proc_list):
    all_procs = []
    seen = set()

    def walk(proc, visited):
        if proc in seen:
            return

        all_procs.append(proc)
        seen.add(proc)

        for sp in LoopIR_SubProcs(proc).result():
            if sp in visited:
                raise ValueError(f"found call cycle involving {sp.name}")
            walk(sp, visited | {proc})

    for proc in proc_list:
        walk(proc, set())

    # Reverse for C declaration order.
    return list(reversed(all_procs))


class LoopIR_FindMems(LoopIR_Do):
    def __init__(self, proc):
        self._mems = set()
        for a in proc.args:
            if a.mem:
                self._mems.add(a.mem)
        super().__init__(proc)

    def result(self):
        return self._mems

    # to improve efficiency
    def do_e(self, e):
        pass

    def do_s(self, s):
        if isinstance(s, LoopIR.Alloc):
            if s.mem:
                self._mems.add(s.mem)
        else:
            super().do_s(s)

    def do_t(self, t):
        pass


class LoopIR_FindExterns(LoopIR_Do):
    def __init__(self, proc):
        self._externs = set()
        super().__init__(proc)

    def result(self):
        return self._externs

    # to improve efficiency
    def do_e(self, e):
        if isinstance(e, LoopIR.Extern):
            self._externs.add((e.f, e.type.basetype().ctype()))
        else:
            super().do_e(e)

    def do_t(self, t):
        pass


class LoopIR_FindConfigs(LoopIR_Do):
    def __init__(self, proc):
        self._configs = set()
        super().__init__(proc)

    def result(self):
        return self._configs

    # to improve efficiency
    def do_e(self, e):
        if isinstance(e, LoopIR.ReadConfig):
            self._configs.add(e.config)
        else:
            super().do_e(e)

    def do_s(self, s):
        if isinstance(s, LoopIR.WriteConfig):
            self._configs.add(s.config)
        super().do_s(s)

    def do_t(self, t):
        pass


def find_all_mems(proc_list):
    mems = set()
    for p in proc_list:
        mems.update(LoopIR_FindMems(p).result())

    return [m for m in mems]


def find_all_externs(proc_list):
    externs = set()
    for p in proc_list:
        externs.update(LoopIR_FindExterns(p).result())

    return externs


def find_all_configs(proc_list):
    configs = set()
    for p in proc_list:
        configs.update(LoopIR_FindConfigs(p).result())

    return list(configs)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler Entry-points

# top level compiler function called by tests!


def run_compile(proc_list, mlir_file_name):
    file_stem = str(Path(mlir_file_name).stem)
    lib_name = sanitize_str(file_stem)
    body = compile_to_strings(lib_name, proc_list)

    return body


def compile_to_strings(lib_name, proc_list):
    proc_list = list(sorted(find_all_subprocs(proc_list), key=lambda x: x.name))
    instrs_global = []
    analyzed_proc_list = []

    # Compile proc bodies
    seen_procs = set()
    for p in proc_list:
        if p.name in seen_procs:
            raise TypeError(f"multiple procs named {p.name}")
        seen_procs.add(p.name)

        # don't compile instruction procedures, but add a comment.
        if p.instr is not None and p.instr.c_global:
            instrs_global.append(p.instr.c_global)
        else:
            p = ParallelAnalysis().run(p)
            p = PrecisionAnalysis().run(p)
            p = WindowAnalysis().apply_proc(p)
            p = MemoryAnalysis().run(p)

            analyzed_proc_list.append(p)

    return IRGenerator().generate(analyzed_proc_list)
