from __future__ import annotations

from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp, FunctionType
from xdsl.ir import Block, SSAValue, Region, Operation
from xdsl.utils.scoped_dict import ScopedDict

from .dialect import (
    ConstantOp,
    AllocOp,
    AssignOp,
    BinOp,
    CallOp,
    IfOp,
    ExternOp,
    ForOp,
    FreeOp,
    ProcedureOp,
    ReadOp,
    ReadConfigOp,
    ReduceOp,
    USubOp,
    WindowExprOp,
    WindowStmtOp,
    WriteConfigOp,
    ExoF16,
    ExoF32,
    ExoF64,
    ExoINT8,
    ExoUINT8,
    ExoUINT16,
    ExoINT32,
    ExoBool,
    ExoIndex,
    ExoSize,
    ExoStride,
    ExoError,
    ExoTensor,
    ExoInt,
    ExoNum,
    ExoObject,
    ExoType,
    ExoMem,
)

from ...core.LoopIR import LoopIR


class IRGeneratorError(Exception):
    pass


class IRGenerator:
    module: ModuleOp
    builder: Builder

    symbol_table: ScopedDict[str, SSAValue] | None = None

    # used in testing
    last_op: Operation | None = None

    seen_procs: set[str] = set()

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder.at_end(self.module.body.blocks[0])

    def declare(self, var: str, value: SSAValue) -> bool:
        assert self.symbol_table is not None
        if var in self.symbol_table:
            return False
        self.symbol_table[var] = value
        return True

    def generate(self, procs) -> ModuleOp:
        for proc in procs:
            self.generate_proc(proc)

        # verify module
        # TODO: none of the operations actually implement verify_()
        try:
            self.module.verify()
        except Exception as e:
            print("module verification failed: ", e)
            raise

        return self.module

    def insert(self, op):
        self.last_op = op
        self.builder.insert(op)

    def generate_proc(self, proc):
        # prevent infinite generation of procedures - shouldn't happen, but LoopIR embeds entire procedures in its AST
        # rather than just referring to them.
        if proc.name in self.seen_procs:
            return

        self.seen_procs.add(proc.name)

        parent_builder = self.builder
        self.symbol_table = ScopedDict[str, SSAValue]()

        # initialise function block
        block = Block(arg_types=[self.get_type(arg.type) for arg in proc.args])
        self.builder = Builder.at_end(block)

        # add arguments to symbol table
        for arg, value in zip(proc.args, block.args):
            self.declare(arg.name, value)

        # generate function body
        self.generate_stmt_list(proc.body)

        # cleanup
        self.symbol_table = None
        self.builder = parent_builder

        input_types = [self.get_type(arg.type) for arg in proc.args]
        func_type = FunctionType.from_lists(input_types, [])

        # insert procedure into module
        self.insert(ProcedureOp(proc.name, func_type, Region(block)))

    def generate_stmt_list(self, stmts):
        assert self.symbol_table is not None
        for stmt in stmts:
            self.generate_stmt(stmt)

    def generate_stmt(self, stmt):
        if isinstance(stmt, LoopIR.Assign):
            self.generate_assign_stmt(stmt)
        elif isinstance(stmt, LoopIR.Reduce):
            self.generate_reduce_stmt(stmt)
        elif isinstance(stmt, LoopIR.WriteConfig):
            self.generate_write_config_stmt(stmt)
        elif isinstance(stmt, LoopIR.Pass):
            # do nothing!!
            pass
        elif isinstance(stmt, LoopIR.If):
            self.generate_if_stmt(stmt)
        elif isinstance(stmt, LoopIR.For):
            self.generate_for_stmt(stmt)
        elif isinstance(stmt, LoopIR.Alloc):
            self.generate_alloc_stmt(stmt)
        elif isinstance(stmt, LoopIR.Free):
            self.generate_free_stmt(stmt)
        elif isinstance(stmt, LoopIR.Call):
            # TODO: call stmts are not supported yet
            pass
        elif isinstance(stmt, LoopIR.Window):
            self.generate_window_stmt(stmt)
        else:
            raise IRGeneratorError(f"Unknown statement {stmt}")

    def generate_assign_stmt(self, assign):
        idx = self.generate_expr_list(assign.idx)
        rhs = self.generate_expr(assign.rhs)

        self.insert(AssignOp(assign.name, assign.type, idx, rhs))

    def generate_reduce_stmt(self, reduce):
        idx = self.generate_expr_list(reduce.idx)
        rhs = self.generate_expr(reduce.rhs)

        self.insert(ReduceOp(reduce.name, reduce.type, idx, rhs))

    def generate_write_config_stmt(self, write_config):
        rhs = self.generate_expr(write_config.rhs)
        self.insert(WriteConfigOp(write_config.name, write_config.field, rhs))

    def generate_if_stmt(self, if_stmt):
        cond = self.generate_expr(if_stmt.cond)

        parent_builder = self.builder

        # construct true_block
        true_block = Block()
        self.builder = Builder.at_end(true_block)
        self.generate_stmt_list(if_stmt.true_stmts)

        # construct false_block
        false_block = Block()
        self.builder = Builder.at_end(false_block)
        self.generate_stmt_list(if_stmt.false_stmts)

        # cleanup and construct
        self.builder = parent_builder
        self.insert(IfOp(cond, Region(true_block), Region(false_block)))

    def generate_for_stmt(self, for_stmt):
        lo = self.generate_expr(for_stmt.lo)
        hi = self.generate_expr(for_stmt.hi)

        parent_builder = self.builder
        parent_scope = self.symbol_table

        # construct loop block
        loop_block = Block(
            # TODO: this should be inferred from lo and hi
            arg_types=[ExoIndex]
        )
        self.builder = Builder.at_end(loop_block)
        self.symbol_table = ScopedDict(parent_scope)

        # add loop variable to symbol table
        self.declare(for_stmt.iter, loop_block.args[0])

        # generate loop body
        self.generate_stmt_list(for_stmt.body)

        # cleanup and construct
        self.symbol_table = parent_scope
        self.builder = parent_builder

        self.insert(ForOp(lo, hi, Region(loop_block)))

    def generate_alloc_stmt(self, alloc):
        self.insert(AllocOp(alloc.name, alloc.type, alloc.mem))

    def generate_free_stmt(self, free):
        self.insert(FreeOp(free.name, free.type, free.mem))

    def generate_call_stmt(self, call):
        # TODO: procedure generation should be top-level, then call should simply use a SymRefAttr to refer to the procedure
        self.generate_proc(call.f)
        args = [self.generate_expr(arg) for arg in call.args]
        self.insert(CallOp(call.f.name, args))

    def generate_window_stmt(self, window):
        rhs = self.generate_expr(window.rhs)
        self.insert(WindowStmtOp(window.name, rhs))

    def generate_expr_list(self, exprs):
        assert self.symbol_table is not None

        for expr in exprs:
            self.generate_expr(expr)

    def generate_expr(self, expr):
        if isinstance(expr, LoopIR.Read):
            return self.generate_read_expr(expr)
        elif isinstance(expr, LoopIR.Const):
            return self.generate_const_expr(expr)
        elif isinstance(expr, LoopIR.USub):
            return self.generate_usub_expr(expr)
        elif isinstance(expr, LoopIR.BinOp):
            return self.generate_binop_expr(expr)
        elif isinstance(expr, LoopIR.Extern):
            return self.generate_extern_expr(expr)
        elif isinstance(expr, LoopIR.WindowExpr):
            return self.generate_window_expr(expr)
        elif isinstance(expr, LoopIR.Stride):
            return self.generate_stride_expr(expr)
        elif isinstance(expr, LoopIR.ReadConfig):
            return self.generate_read_config_expr(expr)
        else:
            raise IRGeneratorError(f"Unknown expression {expr}")

    def generate_read_expr(self, read):
        idx = self.generate_expr_list(read.idx)
        read = ReadOp(read.name, idx)
        self.insert(read)
        return read.res

    def generate_const_expr(self, const):
        const = ConstantOp(const.val)
        self.insert(const)
        return const.res

    def generate_usub_expr(self, usub):
        usub = USubOp(self.generate_expr(usub.arg))
        self.insert(usub)
        return usub.res

    def generate_binop_expr(self, binop):
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)
        binop = BinOp(binop.op, lhs, rhs)
        self.insert(binop)
        return binop.res

    def generate_extern_expr(self, extern):
        args = self.generate_expr_list(extern.args)
        extern = ExternOp(extern.f.name(), args)
        self.insert(extern)
        return extern.res

    def generate_window_expr(self, window):
        pass

    def generate_stride_expr(self, stride):
        pass

    def generate_read_config_expr(self, read_config):
        pass

    def get_type(self, t):
        if isinstance(t, LoopIR.Num):
            return ExoNum
        elif isinstance(t, LoopIR.F16):
            return ExoF16
        elif isinstance(t, LoopIR.F32):
            return ExoF32
        elif isinstance(t, LoopIR.F64):
            return ExoF64
        elif isinstance(t, LoopIR.INT8):
            return ExoINT8
        elif isinstance(t, LoopIR.UINT8):
            return ExoUINT8
        elif isinstance(t, LoopIR.UINT16):
            return ExoUINT16
        elif isinstance(t, LoopIR.INT32):
            return ExoINT32
        elif isinstance(t, LoopIR.Bool):
            return ExoBool
        elif isinstance(t, LoopIR.Int):
            return ExoInt
        elif isinstance(t, LoopIR.Index):
            return ExoIndex
        elif isinstance(t, LoopIR.Size):
            return ExoSize
        elif isinstance(t, LoopIR.Stride):
            return ExoStride
        elif isinstance(t, LoopIR.Error):
            return ExoError
        elif isinstance(t, LoopIR.Tensor):
            return ExoTensor
        else:
            # TODO: add more types, but stupid exam regulations dont let me use GPT to make this quick :(
            raise IRGeneratorError(f"Unknown type {t}")
