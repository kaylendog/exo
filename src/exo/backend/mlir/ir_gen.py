from __future__ import annotations

from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp, FunctionType
from xdsl.ir import Block, SSAValue, Region, Operation, BlockArgument
from xdsl.utils.test_value import TestSSAValue
from xdsl.utils.scoped_dict import ScopedDict
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    I8,
    I32,
    IntegerType,
    IndexType,
)

from ...core.prelude import Sym
from ...core.LoopIR import LoopIR, T

from .dialect import (
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
    ReduceOp,
    USubOp,
    U8,
    U16,
    TensorTypeF16,
    TensorTypeF32,
    TensorTypeF64,
    TensorTypeI8,
    TensorTypeU16,
    TensorTypeU8,
)


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

    def with_empty_scope(self):
        """
        Return this IRGenerator with an empty symbol table.
        """
        self.symbol_table = ScopedDict()
        return self

    def with_declared_arg(self, sym: Sym, type):
        """
        Return this IRGenerator with a symbol declared in the symbol table.
        """
        self.declare_arg(sym, type)
        return self

    def declare_arg(self, sym: Sym, type) -> BlockArgument:
        """
        Declare a symbol in the symbol table.
        """
        raise NotImplementedError

    def with_declared_test_arg(self, sym: Sym, type):
        """
        Return this IRGenerator with a test value declared in the symbol table.
        """
        self.declare_test_arg(sym, type)
        return self

    def declare_test_arg(self, sym: Sym, type) -> TestSSAValue:
        """
        Declare a test value in the symbol table.
        """
        assert self.symbol_table is not None
        value = TestSSAValue(self.get_type(type))
        self.symbol_table[sym.name()] = value
        return value

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

    def symbol(self, sym: Sym) -> SSAValue:
        assert self.symbol_table is not None

        if sym.name() not in self.symbol_table:
            raise IRGeneratorError(f"Unknown symbol {sym.name()}")

        return self.symbol_table[sym.name()]

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
            self.declare_arg(arg.name, value)

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

        self.insert(AssignOp(self.symbol(assign.name), assign.type, idx, rhs))

    def generate_reduce_stmt(self, reduce):
        idx = self.generate_expr_list(reduce.idx)
        rhs = self.generate_expr(reduce.rhs)

        self.insert(ReduceOp(self.symbol(reduce.name), reduce.type, idx, rhs))

    def generate_write_config_stmt(self, write_config):
        # rhs = self.generate_expr(write_config.rhs)
        # self.insert(WriteConfigOp(write_config.name, write_config.field, rhs))
        raise NotImplementedError

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
            arg_types=[IndexType]
        )
        self.builder = Builder.at_end(loop_block)
        self.symbol_table = ScopedDict(parent_scope)

        # add loop variable to symbol table
        self.declare_arg(for_stmt.iter, loop_block.args[0])

        # generate loop body
        self.generate_stmt_list(for_stmt.body)

        # cleanup and construct
        self.symbol_table = parent_scope
        self.builder = parent_builder

        self.insert(ForOp(lo, hi, Region(loop_block)))

    def generate_alloc_stmt(self, alloc):
        self.insert(
            AllocOp(
                self.symbol(alloc.name), self.get_type(alloc.type), alloc.mem.name()
            )
        )

    def generate_free_stmt(self, free):
        self.insert(
            FreeOp(self.symbol(free.name), self.get_type(free.type), free.mem.name())
        )

    def generate_call_stmt(self, call):
        # TODO: procedure generation should be top-level, then call should simply use a SymRefAttr to refer to the procedure
        self.generate_proc(call.f)
        args = [self.generate_expr(arg) for arg in call.args]
        self.insert(CallOp(call.f.name, args))

    # def generate_window_stmt(self, window):
    #     rhs = self.generate_expr(window.rhs)
    #     self.insert(WindowStmtOp(self.symbol(window.name), rhs))

    def generate_expr_list(self, exprs):
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
        read = ReadOp(self.symbol(read.name), idx)
        self.insert(read)
        return read.res

    def generate_const_expr(self, const):
        const = ConstantOp.from_int_and_width(const.val, self.get_type(const.type))
        self.insert(const)
        return const.result

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
        if isinstance(t, T.F16):
            return Float16Type
        elif isinstance(t, T.F32) or isinstance(t, T.Num):
            return Float32Type
        elif isinstance(t, T.F64):
            return Float64Type
        elif isinstance(t, T.INT8):
            return I8
        elif isinstance(t, T.UINT8):
            return U8
        elif isinstance(t, T.UINT16):
            return U16
        elif isinstance(t, T.INT32) or isinstance(t, T.Int):
            return I32
        elif isinstance(t, T.Bool):
            return IntegerType(1)
        elif isinstance(t, T.Index):
            return IndexType
        elif isinstance(t, T.Tensor):
            inner = self.get_type(t.type)
            if isinstance(inner, Float16Type):
                return TensorTypeF16
            elif isinstance(inner, Float32Type):
                return TensorTypeF32
            elif isinstance(inner, Float64Type):
                return TensorTypeF64
            else:
                raise IRGeneratorError(f"Unknown tensor type {t}")
        else:
            raise IRGeneratorError(f"Unknown type {t}")
