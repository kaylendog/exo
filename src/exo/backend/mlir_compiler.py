from __future__ import annotations
from typing import TypeAlias, Annotated

from xdsl.dialects.builtin import (
    BoolAttr,
    IntegerType,
    Signedness,
    I8,
    I32,
    Float16Type,
    Float32Type,
    Float64Type,
    NoneType,
    TensorType,
    SymbolRefAttr,
)

from xdsl.ir import Attribute, SSAValue, Region
from xdsl.irdl import (
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
)
from xdsl.utils.exceptions import VerifyException


# -----------------------------------------------------------------------------
# Exo Type Aliases
# -----------------------------------------------------------------------------

u8 = IntegerType(8, Signedness.UNSIGNED)
u16 = IntegerType(16, Signedness.UNSIGNED)

ExoF16: TypeAlias = Float16Type
ExoF32: TypeAlias = Float32Type
ExoF64: TypeAlias = Float64Type
ExoINT8: TypeAlias = I8
ExoUINT8 = Annotated[IntegerType, u8]
ExoUINT16 = Annotated[IntegerType, u16]
ExoINT32: TypeAlias = I32
ExoBool: TypeAlias = BoolAttr
ExoIndex: TypeAlias = IntegerType
ExoSize: TypeAlias = IntegerType
ExoStride: TypeAlias = IntegerType
ExoError: TypeAlias = NoneType  # needs investigation
ExoTensor: TypeAlias = TensorType[Float64Type]

# union types - might be wrong
ExoInt: TypeAlias = ExoINT8 | ExoUINT8 | ExoUINT16 | ExoINT32
ExoNum: TypeAlias = ExoF16 | ExoF32 | ExoF64 | ExoINT8 | ExoINT32

ExoObject: TypeAlias = (
    ExoF16 | ExoF32 | ExoF64 | ExoINT8 | ExoUINT8 | ExoUINT16 | ExoINT32 | ExoBool
)

ExoType: TypeAlias = SymbolRefAttr
ExoMem: TypeAlias = SymbolRefAttr


# -----------------------------------------------------------------------------
# Exo Stmt Operations
# -----------------------------------------------------------------------------


@irdl_op_definition
class AssignOp(IRDLOperation):
    name = "exo.assign"
    lhs = operand_def(ExoMem)
    rhs = operand_def(SSAValue)


@irdl_op_definition
class ReduceOp(IRDLOperation):
    name = "exo.reduce"
    lhs = operand_def(ExoMem)
    rhs = operand_def(SSAValue)


@irdl_op_definition
class WriteConfig(IRDLOperation):
    name = "exo.write_config"


@irdl_op_definition
class ConditionalOp(IRDLOperation):
    name = "exo.conditional"
    condition = operand_def(ExoBool)

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
        super.__init__(
            operands=[cond],
            result_types=[NoneType],
            regions=[true_region, false_region],
            attributes=attr_dict,
        )


# TODO: needs loop mode
@irdl_op_definition
class ForOp(IRDLOperation):
    name = "exo.for"

    lo = operand_def(SSAValue)
    hi = operand_def(SSAValue)

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

        super.__init__(
            operands=[lo, hi],
            result_types=[NoneType],
            regions=[body],
            attributes=attr_dict,
        )


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "exo.alloc"

    target = attr_def(SymbolRefAttr)
    type = attr_def(SymbolRefAttr)
    mem = attr_def(SymbolRefAttr)

    def __init__(
        self,
        target: str | SymbolRefAttr,
        type: str | SymbolRefAttr,
        mem: str | SymbolRefAttr,
    ):
        if isinstance(target, str):
            target = SymbolRefAttr(target)

        if isinstance(type, str):
            type = SymbolRefAttr(type)

        if isinstance(mem, str):
            mem = SymbolRefAttr(mem)

        super.__init__(
            result_types=[NoneType],
            attributes={"mem": mem, "type": type, "target": target},
        )


@irdl_op_definition
class FreeOp(IRDLOperation):
    name = "exo.free"

    target = attr_def(SymbolRefAttr)
    type = attr_def(SymbolRefAttr)
    mem = attr_def(SymbolRefAttr)

    def __init__(
        self,
        target: str | SymbolRefAttr,
        type: str | SymbolRefAttr,
        mem: str | SymbolRefAttr,
    ):
        if isinstance(target, str):
            target = SymbolRefAttr(target)

        if isinstance(type, str):
            type = SymbolRefAttr(type)

        if isinstance(mem, str):
            mem = SymbolRefAttr(mem)

        super.__init__(
            result_types=[NoneType],
            attributes={"mem": mem, "type": type, "target": target},
        )


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "exo.call"
    arguments = var_operand_def()
    callee = attr_def(SymbolRefAttr)

    def __init__(
        self,
        callee: str | SymbolRefAttr,
        arguments: list[SSAValue | OpResult],
    ):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)

        super.__init__(
            operands=arguments,
            result_types=[NoneType],
            attributes={"callee": callee},
        )


@irdl_op_definition
class WindowStmtOp(IRDLOperation):
    name = "exo.window_stmt"


# -----------------------------------------------------------------------------
# Exo Expr Operations
# -----------------------------------------------------------------------------


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "exo.read"


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "exo.constant"
    value = attr_def(ExoObject)
    res = result_def(ExoObject)

    traits = traits_def(Pure())

    def __init__(self, value: ExoObject):
        self.value = value
        self.res = ExoObject

    def verify_(self) -> None:
        if not self.res.type == self.value.type:
            raise VerifyException(
                "Expected value and result types to be equal: "
                f"{self.res.type}, {self.value.type}"
            )

    def get_type(self):
        return self.res.type


@irdl_op_definition
class USubOp(IRDLOperation):
    name = "exo.usub"


@irdl_op_definition
class BinOp(IRDLOperation):
    name = "exo.binop"

    lhs = operand_def(ExoObject)
    rhs = operand_def(ExoObject)
    result = result_def(ExoObject)

    operation = attr_def(SymbolRefAttr)

    traits = traits_def(Pure())

    def __init__(
        self,
        operation: str | SymbolRefAttr,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
    ):
        if isinstance(operation, str):
            operation = SymbolRefAttr(operation)

        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type],
            attributes={"operation": operation},
        )


@irdl_op_definition
class Extern(IRDLOperation):
    name = "exo.extern"


@irdl_op_definition
class WindowExprOp(IRDLOperation):
    name = "exo.window_expr"


@irdl_op_definition
class StrideExprOp(IRDLOperation):
    name = "exo.stride_expr"


@irdl_op_definition
class CastOp(IRDLOperation):
    name = "exo.cast"
