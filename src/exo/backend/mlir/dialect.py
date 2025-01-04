from __future__ import annotations
from typing import TypeAlias, Annotated

from xdsl.dialects.builtin import (
    BoolAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    I32,
    I8,
    IntegerType,
    NoneType,
    Signedness,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    TupleType,
    FunctionType,
)

from xdsl.ir import Attribute, SSAValue, Region, Dialect
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
    SymbolOpInterface,
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
ExoUINT8: TypeAlias = Annotated[IntegerType, u8]
ExoUINT16: TypeAlias = Annotated[IntegerType, u16]
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

# the LoopIR references object a lot - operating under the assumption that this
# is any well-typed value.
ExoObject: TypeAlias = (
    ExoF16 | ExoF32 | ExoF64 | ExoINT8 | ExoUINT8 | ExoUINT16 | ExoINT32 | ExoBool
)

ExoWindowAccess = TupleType([ExoObject, ExoObject]) | ExoObject

ExoType: TypeAlias = SymbolRefAttr
ExoMem: TypeAlias = SymbolRefAttr


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

    operand = attr_def(SymbolRefAttr)
    type = attr_def(SymbolRefAttr)

    idx = var_operand_def(ExoObject)
    rhs = operand_def(ExoObject)

    def __init__(
        self,
        operand: str | SymbolRefAttr,
        type: str | SymbolRefAttr,
        idx: list[SSAValue | Operation],
        rhs: SSAValue | Operation,
    ):
        if isinstance(operand, str):
            operand = SymbolRefAttr(operand)

        if isinstance(type, str):
            type = SymbolRefAttr(type)

        return super().__init__(
            operands=[idx, rhs],
            result_types=[NoneType],
            attributes={"operand": operand, "type": type},
        )


@irdl_op_definition
class ReduceOp(IRDLOperation):
    name = "exo.reduce"

    operand = attr_def(SymbolRefAttr)
    type = attr_def(SymbolRefAttr)

    idx = operand_def(ExoObject)
    rhs = operand_def(ExoObject)

    def __init__(
        self,
        operand: str | SymbolRefAttr,
        type: str | SymbolRefAttr,
        idx: SSAValue | Operation,
        rhs: SSAValue | Operation,
    ):
        if isinstance(operand, str):
            operand = SymbolRefAttr(operand)

        if isinstance(type, str):
            type = SymbolRefAttr(type)

        return super.__init__(
            operands=[idx, rhs],
            result_types=[NoneType],
            attributes={"operand": operand, "type": type},
        )


@irdl_op_definition
class WriteConfigOp(IRDLOperation):
    name = "exo.write_config"

    operand = attr_def(SymbolRefAttr)
    field = attr_def(StringAttr)

    value = operand_def(ExoObject)

    def __init__(
        self,
        operand: str | SymbolRefAttr,
        field: str,
        value: SSAValue | Operation,
    ):
        if isinstance(operand, str):
            operand = SymbolRefAttr(operand)

        return super.__init__(
            operands=[value],
            result_types=[NoneType],
            attributes={"operand": operand, "field": field},
        )


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
        return super.__init__(
            operands=[cond],
            result_types=[NoneType],
            regions=[true_region, false_region],
            attributes=attr_dict,
        )


# TODO: needs loop mode
@irdl_op_definition
class ForOp(IRDLOperation):
    name = "exo.for"

    lo = operand_def(ExoObject)
    hi = operand_def(ExoObject)

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

        return super.__init__(
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

        return super.__init__(
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

        return super.__init__(
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
        callee: ProcedureOp,
        arguments: list[SSAValue | OpResult],
    ):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)

        return super.__init__(
            operands=arguments,
            result_types=[NoneType],
            attributes={"callee": callee},
        )


@irdl_op_definition
class WindowStmtOp(IRDLOperation):
    name = "exo.window_stmt"

    sym_name = attr_def(SymbolRefAttr)
    rhs = operand_def(ExoObject)

    def __init__(self, sym_name: str | SymbolRefAttr, rhs: SSAValue | Operation):
        if isinstance(sym_name, str):
            sym_name = SymbolRefAttr(sym_name)

        return super().__init__(
            operands=[rhs],
            result_types=[NoneType],
            attributes={"sym_name": sym_name},
        )


# -----------------------------------------------------------------------------
# Exo Expr Operations
# -----------------------------------------------------------------------------


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "exo.read"

    operand = attr_def(SymbolRefAttr)

    idx = var_operand_def()

    res = result_def(ExoObject)

    def __init__(self, operand: str | SymbolRefAttr, idx: list[SSAValue | OpResult]):
        if isinstance(operand, str):
            operand = SymbolRefAttr(operand)

        return super().__init__(
            operands=[idx], result_types=[ExoObject], attributes={"operand": operand}
        )


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "exo.constant"
    value = attr_def(ExoObject)
    res = result_def(ExoObject)

    traits = traits_def(Pure())

    def __init__(self, value: ExoObject):
        return super().__init__(result_types=[ExoObject], attributes={"value": value})

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

    operand = operand_def(ExoObject)

    res = result_def(ExoObject)

    traits = traits_def(Pure())

    def __init__(self, operand: SSAValue | Operation):
        return super().__init__(operands=[operand], result_types=[operand.type])


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

    res = result_def(ExoObject)

    def __init__(self, callee: str | SymbolRefAttr, args: list[SSAValue | OpResult]):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)

        return super().__init__(
            operands=args,
            result_types=[ExoObject],
            attributes={"callee": callee},
        )


@irdl_op_definition
class WindowExprOp(IRDLOperation):
    name = "exo.window_expr"

    operand = attr_def(SymbolRefAttr)

    access = operand_def(ExoWindowAccess)

    res = result_def(ExoObject)

    def __init__(self, operand: str | SymbolRefAttr, access: SSAValue | Operation):
        if isinstance(operand, str):
            operand = SymbolRefAttr(operand)

        return super().__init__(
            operands=[access],
            result_types=[ExoObject],
            attributes={"operand": operand},
        )


@irdl_op_definition
class StrideExprOp(IRDLOperation):
    name = "exo.stride_expr"

    operand = attr_def(SymbolRefAttr)

    dim = operand_def(ExoObject)

    res = result_def(ExoObject)

    def __init__(self, operand: str | SymbolRefAttr, dim: SSAValue | Operation):
        if isinstance(operand, str):
            operand = SymbolRefAttr(operand)

        return super().__init__(
            operands=[dim], result_types=[ExoObject], attributes={"operand": operand}
        )


@irdl_op_definition
class ReadConfigOp(IRDLOperation):
    name = "exo.read_config"

    operand = attr_def(SymbolRefAttr)
    field = attr_def(SymbolRefAttr)

    res = result_def(ExoObject)

    def __init__(self, operand: str | SymbolRefAttr, field: str | SymbolRefAttr):
        if isinstance(operand, str):
            operand = SymbolRefAttr(operand)

        if isinstance(field, str):
            field = SymbolRefAttr(field)

        return super().__init__(
            result_types=[ExoObject],
            attributes={"operand": operand, "field": field},
        )


Exo = Dialect(
    "exo",
    [
        AllocOp,
        AssignOp,
        BinOp,
        CallOp,
        ConditionalOp,
        ConstantOp,
        ExternOp,
        ForOp,
        FreeOp,
        ReadConfigOp,
        ReadOp,
        ReduceOp,
        StrideExprOp,
        USubOp,
        WindowExprOp,
        WindowStmtOp,
        WriteConfigOp,
    ],
)
