from __future__ import annotations
from typing import TypeAlias

from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp, FunctionType
from xdsl.ir import Block, SSAValue, Region, Operation, BlockArgument, OpResult
from xdsl.utils.test_value import TestSSAValue
from xdsl.utils.scoped_dict import ScopedDict
from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i1,
    MemRefType,
    FloatAttr,
    IntegerAttr,
    IntAttr,
    BoolAttr,
)
from xdsl.dialects.arith import (
    ConstantOp,
    AddfOp,
    SubfOp,
    MulfOp,
    DivfOp,
    NegfOp,
    AddiOp,
    SubiOp,
    MuliOp,
    DivSIOp,
    RemSIOp,
)
from xdsl.dialects.func import FuncOp, CallOp
from xdsl.dialects.memref import LoadOp, StoreOp, AllocOp, DeallocOp
from xdsl.dialects.scf import IfOp, ForOp, YieldOp

from ...core.prelude import Sym
from ...core.LoopIR import LoopIR, T


MemRefTypeF16: TypeAlias = MemRefType[Float16Type]
MemRefTypeF32: TypeAlias = MemRefType[Float32Type]
MemRefTypeF64: TypeAlias = MemRefType[Float64Type]


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

    def declare_arg(self, sym: Sym, type, block, idx) -> BlockArgument:
        """
        Declare a symbol in the symbol table.
        """
        assert self.symbol_table is not None
        arg = BlockArgument(self.get_type(type), block, idx)
        self.symbol_table[sym.name()] = arg
        return arg

    def declare_value(self, sym: Sym, value: SSAValue) -> SSAValue:
        """
        Declare a value in the symbol table.
        """
        assert self.symbol_table is not None
        self.symbol_table[sym.name()] = value
        return value

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
            self.generate_procedure(proc)

        # verify module
        # TODO: none of the operations actually implement verify_()
        try:
            self.module.verify()
        except Exception as e:
            print("module verification failed: ", e)
            raise

        return self.module

    def get_sym(self, sym: Sym) -> SSAValue:
        """Get the SSAValue for a symbol."""
        assert self.symbol_table is not None

        if sym.name() not in self.symbol_table:
            raise IRGeneratorError(f"Unknown symbol {sym.name()}")

        return self.symbol_table[sym.name()]

    def insert(self, op):
        """Insert an operation into the module and set it as the last operation."""
        self.last_op = op
        self.builder.insert(op)

    def cast_to_index(self, value: SSAValue) -> SSAValue:
        self.em

    def generate_procedure(self, procedure):
        if procedure.name in self.seen_procs:
            return

        self.seen_procs.add(procedure.name)

        parent_builder = self.builder
        self.symbol_table = ScopedDict[str, SSAValue]()

        # initialise function block
        block = Block(arg_types=[self.get_type(arg.type) for arg in procedure.args])
        self.builder = Builder.at_end(block)

        # add arguments to symbol table
        for idx, arg, value in enumerate(zip(procedure.args, block.args)):
            self.declare_arg(arg.name, value, block, idx)

        # generate function body
        self.generate_stmt_list(procedure.body)

        # cleanup
        self.symbol_table = None
        self.builder = parent_builder

        input_types = [self.get_type(arg.type) for arg in procedure.args]
        func_type = FunctionType.from_lists(input_types, [])

        # insert procedure into module
        self.insert(FuncOp(procedure.name, func_type, Region(block)))

    def generate_stmt_list(self, stmts):
        """Generate a list of statements."""
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

        self.insert(StoreOp(operands=[self.get_sym(assign.name), rhs, idx]))

    def generate_reduce_stmt(self, reduce):
        idx = self.generate_expr_list(reduce.idx)
        rhs = self.generate_expr(reduce.rhs)

        memref = self.get_sym(reduce.name)

        # load value from memory, add rhs, store back - could use AtomicRMWOp here?
        load = LoadOp(operands=[memref, idx], result_types=[self.get_type(memref)])
        inc = AddfOp(load.res, rhs)
        store = StoreOp(operands=[inc, memref, idx])

        self.insert(load)
        self.insert(inc)
        self.insert(store)

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
        self.generate_stmt_list(if_stmt.body)
        self.insert(YieldOp())

        # construct false_block
        false_block = Block()
        self.builder = Builder.at_end(false_block)
        self.generate_stmt_list(if_stmt.orelse)
        self.insert(YieldOp())

        # cleanup and construct
        self.builder = parent_builder
        self.insert(IfOp(cond, [], Region(true_block), Region(false_block)))

    def generate_for_stmt(self, for_stmt):
        lo = self.generate_expr(for_stmt.lo)
        hi = self.generate_expr(for_stmt.hi)

        parent_builder = self.builder
        parent_scope = self.symbol_table

        # construct loop block
        loop_block = Block(
            # TODO: this should be inferred from lo and hi
            arg_types=[lo.type],
        )
        self.builder = Builder.at_end(loop_block)
        self.symbol_table = ScopedDict(parent_scope)

        # add loop variable to symbol table
        self.declare_arg(for_stmt.iter, loop_block.args[0], loop_block, 0)

        # generate loop body
        self.generate_stmt_list(for_stmt.body)
        self.insert(YieldOp())

        # cleanup and construct
        self.symbol_table = parent_scope
        self.builder = parent_builder

        self.insert(ForOp(lo, hi, ConstantOp(1, lo.type), [], Region(loop_block)))

    def generate_alloc_stmt(self, alloc):
        op = AllocOp([], [], result_type=self.get_type(alloc.type))
        self.insert(op)
        self.declare_value(alloc.name, op.results[0])
        return op.results[0]

    def generate_free_stmt(self, free):
        self.insert(DeallocOp(operands=[self.get_sym(free.name)], result_types=[]))

    def generate_call_stmt(self, call):
        # TODO: procedure generation should be top-level, then call should simply use a SymRefAttr to refer to the procedure
        self.generate_procedure(call.f)
        args = [self.generate_expr(arg) for arg in call.args]
        self.insert(CallOp(call.f.name, args, []))

    # def generate_window_stmt(self, window):
    #     rhs = self.generate_expr(window.rhs)
    #     self.insert(WindowStmtOp(self.symbol(window.name), rhs))

    def generate_expr_list(self, exprs) -> list[OpResult]:
        return [self.generate_expr(expr) for expr in exprs]

    def generate_expr(self, expr) -> OpResult:
        if isinstance(expr, LoopIR.Read):
            return self.generate_read_expr(expr)
        elif isinstance(expr, LoopIR.Const):
            return self.generate_const_expr(expr)
        elif isinstance(expr, LoopIR.BinOp):
            return self.generate_binop_expr(expr)
        else:
            raise IRGeneratorError(f"Unknown expression {expr}")

    def generate_read_expr(self, read):
        idx = self.generate_expr_list(read.idx)
        read = LoadOp(
            operands=[self.get_sym(read.name), idx],
            result_types=[self.get_type(read.type)],
        )
        self.insert(read)
        return read.res

    def generate_const_expr(self, const):
        type = self.get_type(const.type)

        # construct attribute depending on type
        if type in [f16, f32, f64]:
            attr = FloatAttr(const.val, type)
        elif type in [i8, i16, i32]:
            attr = IntegerAttr(const.val, type)
        elif type == i1:
            attr = BoolAttr(const.val, i1)
        else:
            raise IRGeneratorError(f"Unknown type {type} passed to Const")

        const = ConstantOp(attr, self.get_type(const.type))
        self.insert(const)
        return const.result

    def generate_usub_expr(self, usub):
        expr = self.generate_expr(usub.arg)
        # float case
        if self.get_type(usub.type) in [f16, f32, f64]:
            usub = NegfOp(expr)
        # integer case
        elif self.get_type(usub.type) in [i8, i16, i32]:
            zero = ConstantOp(IntegerAttr(0, self.get_type(usub.type)))
            usub = SubiOp(zero.result, expr, result_type=self.get_type(usub.type))
            self.insert(zero)
        else:
            raise IRGeneratorError(f"Bad type {type} passed to USub")

        self.insert(usub)
        return usub.result

    def generate_binop_expr(self, binop):
        type = self.get_type(binop.type)

        if type in [f16, f32, f64]:
            return self.generate_binop_expr_float(binop)
        elif type in [i8, i16, i32]:
            return self.generate_binop_expr_int(binop)
        else:
            raise IRGeneratorError(f"Unknown type {type}")

    def generate_binop_expr_float(self, binop):
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)
        type = self.get_type(binop.type)

        if binop.op == "+":
            binop = AddfOp(lhs, rhs, result_type=type)
        elif binop.op == "-":
            binop = SubfOp(lhs, rhs, result_type=type)
        elif binop.op == "*":
            binop = MulfOp(lhs, rhs, result_type=type)
        elif binop.op == "/":
            binop = DivfOp(lhs, rhs, result_type=type)
        else:
            raise IRGeneratorError(f"Unknown binop {binop.op}")

        self.insert(binop)
        return binop.result

    def generate_binop_expr_int(self, binop):
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)
        type = self.get_type(binop.type)

        if binop.op == "+":
            binop = AddiOp(lhs, rhs, result_type=type)
        elif binop.op == "-":
            binop = SubiOp(lhs, rhs, result_type=type)
        elif binop.op == "*":
            binop = MuliOp(lhs, rhs, result_type=type)
        elif binop.op == "/":
            binop = DivSIOp(lhs, rhs, result_type=type)
        elif binop.op == "%":
            binop = RemSIOp(lhs, rhs, result_type=type)
        else:
            raise IRGeneratorError(f"Unknown binop {binop.op}")

        self.insert(binop)
        return binop.result

    def generate_extern_expr(self, extern):
        args = self.generate_expr_list(extern.args)
        extern = CallOp(extern.f.name, args, [])
        self.insert(extern)
        return extern.res

    def generate_window_expr(self, window):
        pass

    def generate_stride_expr(self, stride):
        pass

    def generate_read_config_expr(self, read_config):
        pass

    def get_type(self, t):
        # mlir
        if isinstance(t, SSAValue):
            return t.type
        # exo
        if isinstance(t, T.F16):
            return f16
        elif isinstance(t, T.F32) or isinstance(t, T.Num):
            return f32
        elif isinstance(t, T.F64):
            return f64
        elif isinstance(t, T.INT8) or isinstance(t, T.UINT8):
            return i8
        elif isinstance(t, T.UINT16):
            return i16
        elif isinstance(t, T.INT32) or isinstance(t, T.Int) or isinstance(t, T.Index):
            return i32
        elif isinstance(t, T.Bool):
            return i1
        elif isinstance(t, T.Tensor):
            inner = self.get_type(t.type)
            if inner == f16:
                return MemRefTypeF16(f16, self.get_shape(t))
            elif inner == f32:
                return MemRefTypeF32(f32, self.get_shape(t))
            elif inner == f64:
                return MemRefTypeF64(f64, self.get_shape(t))
            else:
                raise IRGeneratorError(f"Unknown tensor type {t}")
        else:
            raise IRGeneratorError(f"Unknown type {t}")

    def get_shape(self, type) -> list[IntegerAttr]:
        assert isinstance(type, T.Tensor)

        def attr_from_expr(expr):
            if isinstance(expr, LoopIR.Const):
                return IntAttr(expr.val)
            else:
                raise IRGeneratorError(f"Invalid shape argument {expr}")

        return [attr_from_expr(expr) for expr in type.shape()]
