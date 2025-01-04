import functools
import re
import textwrap
from collections import ChainMap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ...core.LoopIR import LoopIR, LoopIR_Do, get_writes_of_stmts, T, CIR
from ...core.configs import ConfigError
from ..mem_analysis import MemoryAnalysis
from ...core.memory import MemGenError, Memory, DRAM, StaticMemory
from ..parallel_analysis import ParallelAnalysis
from ..prec_analysis import PrecisionAnalysis
from ...core.prelude import *
from ..win_analysis import WindowAnalysis
from ...rewrite.range_analysis import IndexRangeEnvironment


def sanitize_str(s):
    return re.sub(r"\W", "_", s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

CacheDict = lambda: defaultdict(CacheDict)


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


def run_compile(proc_list, h_file_name: str):
    file_stem = str(Path(h_file_name).stem)
    lib_name = sanitize_str(file_stem)
    body = compile_to_strings(lib_name, proc_list)

    return body


def compile_to_strings(lib_name, proc_list):
    # Get transitive closure of call-graph
    orig_procs = [id(p) for p in proc_list]

    def from_lines(x):
        return "\n".join(x)

    proc_list = list(sorted(find_all_subprocs(proc_list), key=lambda x: x.name))

    # Header contents
    ctxt_name, ctxt_def = _compile_context_struct(find_all_configs(proc_list), lib_name)
    struct_defns = set()
    public_fwd_decls = []

    # Body contents
    memory_code = _compile_memories(find_all_mems(proc_list))
    private_fwd_decls = []
    proc_bodies = []
    instrs_global = []
    analyzed_proc_list = []

    # Compile proc bodies
    seen_procs = set()
    for p in proc_list:
        if p.name in seen_procs:
            raise TypeError(f"multiple procs named {p.name}")
        seen_procs.add(p.name)

        # don't compile instruction procedures, but add a comment.
        if p.instr is not None:
            argstr = ",".join([str(a.name) for a in p.args])
            proc_bodies.extend(
                [
                    "",
                    '/* relying on the following instruction..."',
                    f"{p.name}({argstr})",
                    p.instr.c_instr,
                    "*/",
                ]
            )
            if p.instr.c_global:
                instrs_global.append(p.instr.c_global)
        else:
            is_public_decl = id(p) in orig_procs

            p = ParallelAnalysis().run(p)
            p = PrecisionAnalysis().run(p)
            p = WindowAnalysis().apply_proc(p)
            p = MemoryAnalysis().run(p)

            print(p)

            if is_public_decl:
                public_fwd_decls.append(d)
            else:
                private_fwd_decls.append(d)

            proc_bodies.append(b)

            analyzed_proc_list.append(p)

    # Structs are just blobs of code... still sort them for output stability
    struct_defns = [x.definition for x in sorted(struct_defns, key=lambda x: x.name)]

    extern_code = _compile_externs(find_all_externs(analyzed_proc_list))

    body_contents = [
        instrs_global,
        memory_code,
        extern_code,
        private_fwd_decls,
        proc_bodies,
    ]
    body_contents = list(filter(lambda x: x, body_contents))  # filter empty lines
    body_contents = map(from_lines, body_contents)
    body_contents = from_lines(body_contents)
    body_contents += "\n"  # New line at end of file
    return body_contents


def _compile_externs(externs):
    extern_code = []
    for f, t in sorted(externs, key=lambda x: x[0].name() + x[1]):
        if glb := f.globl(t):
            extern_code.append(glb)
    return extern_code


def _compile_memories(mems):
    memory_code = []
    for m in sorted(mems, key=lambda x: x.name()):
        memory_code.append(m.global_())
    return memory_code


def _compile_context_struct(configs, lib_name):
    if not configs:
        return "void", []

    ctxt_name = f"{lib_name}_Context"
    ctxt_def = [f"typedef struct {ctxt_name} {{ ", f""]

    seen = set()
    for c in sorted(configs, key=lambda x: x.name()):
        name = c.name()

        if name in seen:
            raise TypeError(f"multiple configs named {name}")
        seen.add(name)

        if c.is_allow_rw():
            sdef_lines = c.c_struct_def()
            sdef_lines = [f"    {line}" for line in sdef_lines]
            ctxt_def += sdef_lines
            ctxt_def += [""]
        else:
            ctxt_def += [f"// config '{name}' not materialized", ""]

    ctxt_def += [f"}} {ctxt_name};"]
    return ctxt_name, ctxt_def
