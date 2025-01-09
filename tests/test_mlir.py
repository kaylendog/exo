from exo.backend.mlir.ir_gen import IRGenerator
from exo.core.LoopIR import LoopIR, T
from exo.core.memory import DRAM
from exo.core.prelude import SrcInfo, Sym
from exo.core.extern import Extern

from xdsl.utils.scoped_dict import ScopedDict


def test_emit_assign_op():
    srcinfo = SrcInfo("test_mlir.py", 0)

    # x[0] = 1
    ir = LoopIR.Assign(
        Sym("x"),
        T.Tensor(
            [
                LoopIR.Const(32, T.index, srcinfo),
            ],
            False,
            T.f32,
        ),
        [LoopIR.Const(0, T.index, srcinfo)],
        LoopIR.Const(0.0, T.f32, srcinfo),
        srcinfo,
    )

    gen = (
        IRGenerator()
        .with_empty_scope()
        .with_declared_test_arg(
            Sym("x"),
            T.Tensor(
                [
                    LoopIR.Const(32, T.index, srcinfo),
                ],
                False,
                T.f32,
            ),
        )
    )
    gen.generate_assign_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_reduce_op():
    srcinfo = SrcInfo("test_mlir.py", 0)

    # x[0] += 1
    ir = LoopIR.Reduce(
        Sym("x"),
        T.Tensor(
            [
                LoopIR.Const(32, T.index, srcinfo),
            ],
            False,
            T.f32,
        ),
        [LoopIR.Const(0, T.int, srcinfo)],
        LoopIR.Const(0.0, T.f32, srcinfo),
        srcinfo,
    )

    gen = (
        IRGenerator()
        .with_empty_scope()
        .with_declared_test_arg(
            Sym("x"),
            T.Tensor(
                [
                    LoopIR.Const(32, T.index, srcinfo),
                ],
                False,
                T.f32,
            ),
        )
    )
    gen.generate_reduce_stmt(ir)

    print(gen.module)
    gen.module.verify()


# def test_emit_write_config_op():
#     # TODO: discover what exactly WriteConfig does
#     raise NotImplementedError


def test_emit_if_op():
    srcinfo = SrcInfo("test_mlir.py", 0)

    # if True:
    #     pass
    # else:
    #     pass
    ir = LoopIR.If(LoopIR.Const(True, T.bool, srcinfo), [], [], srcinfo)

    gen = IRGenerator()
    gen.generate_if_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_for_op():
    srcinfo = SrcInfo("test_mlir.py", 0)

    # for i in seq(0, 10):
    #   pass
    ir = LoopIR.For(
        Sym("i"),
        LoopIR.Const(0, T.int, srcinfo),
        LoopIR.Const(10, T.int, srcinfo),
        [],
        LoopIR.Seq(),
        srcinfo,
    )

    gen = IRGenerator().with_empty_scope()
    gen.generate_for_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_alloc_op():
    srcinfo = SrcInfo("test_mlir.py", 0)

    ir = LoopIR.Alloc(
        Sym("x"),
        T.Tensor(
            [
                LoopIR.Const(32, T.index, srcinfo),
            ],
            False,
            T.f32,
        ),
        DRAM,
        srcinfo,
    )

    gen = IRGenerator().with_empty_scope()
    gen.generate_alloc_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_free_op():
    srcinfo = SrcInfo("test_mlir.py", 0)

    ir = LoopIR.Free(
        Sym("x"),
        T.Tensor(
            [
                LoopIR.Const(32, T.index, srcinfo),
            ],
            False,
            T.f32,
        ),
        DRAM,
        srcinfo,
    )

    gen = (
        IRGenerator()
        .with_empty_scope()
        .with_declared_test_arg(
            Sym("x"),
            T.Tensor(
                [
                    LoopIR.Const(32, T.index, srcinfo),
                ],
                False,
                T.f32,
            ),
        )
    )
    gen.generate_free_stmt(ir)

    print(gen.module)
    gen.module.verify()


# def test_emit_call_op():
#     srcinfo = SrcInfo("test_mlir.py", 0)

#     ir = LoopIR.Call(
#         LoopIR.proc("blank", [], [], [], None, srcinfo),
#         [],
#         srcinfo,
#     )

#     gen = IRGenerator().with_empty_scope()
#     gen.generate_call_stmt(ir)

#     gen.last_op.verify()
#     print(gen.last_op)


# def test_emit_window_stmt_op():
#     # TODO: window statement op seems wrong
#     raise NotImplementedError


def test_read_op():
    srcinfo = SrcInfo("test_mlir.py", 0)

    ir = LoopIR.Read(Sym("x"), [LoopIR.Const(0, T.index, srcinfo)], T.f32, srcinfo)

    gen = (
        IRGenerator()
        .with_empty_scope()
        .with_declared_test_arg(
            Sym("x"),
            T.Tensor(
                [
                    LoopIR.Const(32, T.index, srcinfo),
                ],
                False,
                T.f32,
            ),
        )
    )
    gen.generate_read_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_const_op_int():
    srcinfo = SrcInfo("test_mlir.py", 0)

    ir = LoopIR.Const(0, T.int, srcinfo)

    gen = IRGenerator()
    gen.generate_const_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_const_op_float():
    srcinfo = SrcInfo("test_mlir.py", 0)

    ir = LoopIR.Const(0.0, T.f32, srcinfo)

    gen = IRGenerator()
    gen.symbol_table = ScopedDict()
    gen.generate_const_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_const_op_bool():
    srcinfo = SrcInfo("test_mlir.py", 0)

    ir = LoopIR.Const(True, T.bool, srcinfo)

    gen = IRGenerator()
    gen.symbol_table = ScopedDict()
    gen.generate_const_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_usub_op():
    srcinfo = SrcInfo("test_mlir.py", 0)

    ir = LoopIR.USub(LoopIR.Const(0, T.int, srcinfo), T.int, srcinfo)

    gen = IRGenerator()
    gen.generate_usub_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_bin_op():
    srcinfo = SrcInfo("test_mlir.py", 0)

    ir = LoopIR.BinOp(
        "+",
        LoopIR.Const(0, T.int, srcinfo),
        LoopIR.Const(0, T.int, srcinfo),
        T.int,
        srcinfo,
    )

    gen = IRGenerator()
    gen.generate_binop_expr(ir)

    print(gen.module)
    gen.module.verify()


# def test_emit_extern_op():
#     srcinfo = SrcInfo("test_mlir.py", 0)

#     ir = LoopIR.Extern(Extern("example"), [], T.int, srcinfo)

#     gen = IRGenerator()
#     gen.generate_extern_expr(ir)

#     gen.last_op.verify()
#     print(gen.last_op)


# def test_emit_window_expr_op():
#     raise NotImplementedError


# def test_emit_stride_op():
#     raise NotImplementedError


# def test_read_config_op():
#     raise NotImplementedError
