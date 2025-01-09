from __future__ import annotations
from typing import TypeAlias, Annotated

from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    I8,
    I32,
    IntegerType,
    Signedness,
    StringAttr,
    SymbolRefAttr,
    FunctionType,
    TensorType,
    IndexType,
)

from xdsl.ir import Attribute, SSAValue, Region, Dialect
from xdsl.irdl import (
    base,
    Block,
    IRDLOperation,
    Operation,
    OpResult,
    Sequence,
    attr_def,
    irdl_op_definition,
    operand_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import (
    Pure,
    RecursiveMemoryEffect,
    RecursivelySpeculatable,
    SymbolOpInterface,
)


# -----------------------------------------------------------------------------
# Exo Type Aliases
# -----------------------------------------------------------------------------


u8 = IntegerType(8, Signedness.UNSIGNED)
u16 = IntegerType(16, Signedness.UNSIGNED)

U8: TypeAlias = Annotated[IntegerType, u8]
U16: TypeAlias = Annotated[IntegerType, u16]

TensorTypeF16: TypeAlias = TensorType[Float16Type]
TensorTypeF32: TypeAlias = TensorType[Float32Type]
TensorTypeF64: TypeAlias = TensorType[Float64Type]

TensorTypeI8: TypeAlias = TensorType[I8]
TensorTypeU8: TypeAlias = TensorType[U8]
TensorTypeU16: TypeAlias = TensorType[U16]

AnyNumericConstr = (
    base(U8) | base(U16) | base(Float16Type) | base(Float32Type) | base(Float64Type)
)


@irdl_op_definition
class ProcedureOp(IRDLOperation):
    name = "exo.proc"
    body = region_def()
    sym_name = attr_def(StringAttr)
    function_type = attr_def(FunctionType)

    # TODO: may need ProcedureOpCallableInterface or similar
    traits = traits_def(SymbolOpInterface())

    def __init__(
        self,
        name: str,
        ftype: FunctionType,
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ):
        attributes: dict[str, Attribute] = {"sym_name": StringAttr(name)}
        if not isinstance(region, Region):
            region = Region(Block(arg_types=ftype.input))

        return super().__init__(
            regions=[region],
            attributes=attributes,
        )


# -----------------------------------------------------------------------------
# Exo Stmt Operations
# -----------------------------------------------------------------------------


@irdl_op_definition
class AssignOp(IRDLOperation):
    name = "exo.assign"

    rhs = operand_def(AnyNumericConstr)
    indices = var_operand_def(IndexType)

    sym_name = attr_def(StringAttr)
    type = attr_def(StringAttr)

    def __init__(
        self,
        sym_name: str,
        type: str,
        indices: list[SSAValue | Operation],
        rhs: SSAValue | Operation,
    ):
        sym_name = StringAttr(sym_name)
        type = StringAttr(type)

        return super().__init__(
            operands=[rhs, indices],
            attributes={"sym_name": sym_name, "type": type},
        )


@irdl_op_definition
class ReduceOp(IRDLOperation):
    name = "exo.reduce"

    rhs = operand_def(AnyNumericConstr)
    idx = var_operand_def(IndexType)

    sym_name = attr_def(StringAttr)
    type = attr_def(StringAttr)

    def __init__(
        self,
        sym_name: str,
        type: str,
        idx: SSAValue | Operation,
        rhs: SSAValue | Operation,
    ):
        sym_name = StringAttr(sym_name)
        type = StringAttr(type)

        return super().__init__(
            operands=[rhs, idx],
            attributes={"sym_name": sym_name, "type": type},
        )


# @irdl_op_definition
# class WriteConfigOp(IRDLOperation):
#     name = "exo.write_config"

#     sym_name = attr_def(StringAttr)
#     field = attr_def(StringAttr)

#     value = operand_def(AnyNumericConstr)

#     def __init__(
#         self,
#         operand: str,
#         field: str,
#         value: SSAValue | Operation,
#     ):
#         operand = StringAttr(operand)
#         field = StringAttr(field)

#         return super.__init__(
#             operands=[value],
#             attributes={"operand": operand, "field": field},
#         )


@irdl_op_definition
class IfOp(IRDLOperation):
    name = "exo.if"

    condition = operand_def(IntegerType(1))

    true_region = region_def("single_block")
    false_region = region_def("single_block")

    traits = traits_def(RecursiveMemoryEffect, RecursivelySpeculatable)

    def __init__(
        self,
        cond: SSAValue | Operation,
        true_region: Region | Sequence[Block] | Sequence[Operation],
        false_region: Region | Sequence[Block] | Sequence[Operation],
        attr_dict: dict[str, Attribute] = {},
    ):
        return super.__init__(
            operands=[cond],
            regions=[true_region, false_region],
            attributes=attr_dict,
        )


# TODO: needs loop mode
@irdl_op_definition
class ForOp(IRDLOperation):
    name = "exo.for"

    lo = operand_def(IndexType)
    hi = operand_def(IndexType)

    body = region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
    )

    def __init__(
        self,
        lo: SSAValue | Operation,
        hi: SSAValue | Operation,
        body: Region | Sequence[Block] | Sequence[Operation],
        attr_dict: dict[str, Attribute] = {},
    ):
        if isinstance(body, Block):
            body = [body]

        return super().__init__(
            operands=[lo, hi],
            regions=[body],
            attributes=attr_dict,
        )


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "exo.alloc"

    target = attr_def(StringAttr)
    type = attr_def(StringAttr)
    mem = attr_def(StringAttr)

    def __init__(
        self,
        target: str,
        type: str,
        mem: str,
    ):
        target = StringAttr(target)
        type = StringAttr(type)
        mem = StringAttr(mem)

        return super().__init__(
            attributes={"mem": mem, "type": type, "target": target},
        )


@irdl_op_definition
class FreeOp(IRDLOperation):
    name = "exo.free"

    target = attr_def(StringAttr)
    type = attr_def(StringAttr)
    mem = attr_def(StringAttr)

    def __init__(
        self,
        target: str,
        type: str,
        mem: str,
    ):
        target = StringAttr(target)
        type = StringAttr(type)
        mem = StringAttr(mem)

        return super().__init__(
            attributes={"mem": mem, "type": type, "target": target},
        )


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "exo.call"
    arguments = var_operand_def()
    callee = attr_def(SymbolRefAttr)

    def __init__(
        self,
        callee: ProcedureOp,
        arguments: list[SSAValue | OpResult],
    ):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)

        return super().__init__(
            operands=[arguments],
            attributes={"callee": callee},
        )


# @irdl_op_definition
# class WindowStmtOp(IRDLOperation):
#     name = "exo.window_stmt"

#     sym_name = attr_def(StringAttr)
#     rhs = operand_def(AnyExoValue)

#     def __init__(self, sym_name: StringAttr, rhs: SSAValue | Operation):
#         sym_name = StringAttr(sym_name)

#         return super().__init__(
#             operands=[rhs],
#             attributes={"sym_name": sym_name},
#         )


# -----------------------------------------------------------------------------
# Exo Expr Operations
# -----------------------------------------------------------------------------


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "exo.read"

    sym_name = attr_def(StringAttr)
    idx = var_operand_def()
    res = result_def(AnyNumericConstr)

    def __init__(self, sym_name: str, idx: list[SSAValue | OpResult]):
        sym_name = StringAttr(sym_name)
        return super().__init__(
            operands=[idx],
            result_types=[AnyNumericConstr],
            attributes={"sym_name": sym_name},
        )


# @irdl_op_definition
# class ConstantOp(IRDLOperation):
#     name = "exo.constant"
#     value = attr_def(AnyValueAttr)
#     res = result_def(ExoObject)

#     traits = traits_def(Pure(), ConstantLike())

#     def __init__(self, value: int | float | bool):
#         value = cast(AnyIntegerAttr | FloatAttr[AnyFloat] | BoolAttr, value)
#         value_type = value.type

#         return super().__init__(
#             result_types=[value_type],
#             attributes={"value": value},
#         )

#     def verify_(self) -> None:
#         if not self.res.type == self.value.type:
#             raise VerifyException(
#                 "Expected value and result types to be equal: "
#                 f"{self.res.type}, {self.value.type}"
#             )

#     def get_type(self):
#         return self.res.type


@irdl_op_definition
class USubOp(IRDLOperation):
    name = "exo.usub"

    operand = operand_def(AnyNumericConstr)

    res = result_def(AnyNumericConstr)

    traits = traits_def(Pure())

    def __init__(self, operand: SSAValue | Operation):
        return super().__init__(operands=[operand], result_types=[operand.type])


@irdl_op_definition
class BinOp(IRDLOperation):
    name = "exo.binop"

    lhs = operand_def(AnyNumericConstr)
    rhs = operand_def(AnyNumericConstr)
    res = result_def(AnyNumericConstr)

    operation = attr_def(StringAttr)

    traits = traits_def(Pure())

    def __init__(
        self,
        operation: str,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
    ):
        if isinstance(operation, str):
            operation = StringAttr(operation)

        return super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type],
            attributes={"operation": operation},
        )


@irdl_op_definition
class ExternOp(IRDLOperation):
    name = "exo.extern"

    callee = attr_def(SymbolRefAttr)

    args = var_operand_def()

    res = result_def(AnyNumericConstr)

    def __init__(self, callee: str | SymbolRefAttr, args: list[SSAValue | OpResult]):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)

        return super().__init__(
            operands=[args],
            result_types=[AnyNumericConstr],
            attributes={"callee": callee},
        )


# @irdl_op_definition
# class WindowExprOp(IRDLOperation):
#     name = "exo.window_expr"

#     access = operand_def(ExoWindowAccess)
#     res = result_def(AnyExoValue)

#     operand = attr_def(StringAttr)

#     def __init__(self, sym_name: str, access: SSAValue | Operation):
#         sym_name = StringAttr(sym_name)

#         return super().__init__(
#             operands=[access],
#             result_types=[AnyExoValue],
#             attributes={"sym_name": sym_name},
#         )


@irdl_op_definition
class StrideExprOp(IRDLOperation):
    name = "exo.stride_expr"

    dim = operand_def(AnyNumericConstr)
    res = result_def(AnyNumericConstr)

    operand = attr_def(StringAttr)

    def __init__(self, sym_name: str, dim: SSAValue | Operation):
        sym_name = StringAttr(sym_name)

        return super().__init__(
            operands=[dim],
            result_types=[AnyNumericConstr],
            attributes={"sym_name": sym_name},
        )


@irdl_op_definition
class ReadConfigOp(IRDLOperation):
    name = "exo.read_config"

    sym_name = attr_def(StringAttr)
    field = attr_def(StringAttr)

    res = result_def(AnyNumericConstr)

    def __init__(self, sym_name: str, field: str):
        sym_name = StringAttr(sym_name)
        field = StringAttr(field)
        return super().__init__(
            result_types=[AnyNumericConstr],
            attributes={"sym_name": sym_name, "field": field},
        )


Exo = Dialect(
    "exo",
    [
        AllocOp,
        AssignOp,
        BinOp,
        CallOp,
        IfOp,
        ExternOp,
        ForOp,
        FreeOp,
        ReadConfigOp,
        ReadOp,
        ReduceOp,
        StrideExprOp,
        USubOp,
        # WindowExprOp,
        # WindowStmtOp,
        # WriteConfigOp,
    ],
)
