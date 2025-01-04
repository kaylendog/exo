from __future__ import annotations


from abc import ABC, abstractmethod
from typing import TypeAlias, cast, Annotated

from xdsl.dialects.builtin import (
    BoolAttr,
    IntegerType,
    Signedness,
    I8,
    I32,
    I64,
    Float16Type,
    Float32Type,
    Float64Type,
    NoneType,
    TensorType,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    OpResult,
    OpTraits,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    base,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import (
    CallableOpInterface,
    HasCanonicalizationPatternsTrait,
    IsTerminator,
    OpTrait,
    Pure,
    SymbolOpInterface,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.isattr import isattr


# -----------------------------------------------------------------------------
# LoopIR Type Aliases
# -----------------------------------------------------------------------------

u8 = IntegerType(8, Signedness.UNSIGNED)
u16 = IntegerType(16, Signedness.UNSIGNED)

LoopIR_F16: TypeAlias = Float16Type
LoopIR_F32: TypeAlias = Float32Type
LoopIR_F64: TypeAlias = Float64Type
LoopIR_INT8: TypeAlias = I8
LoopIR_UINT8 = Annotated[IntegerType, u8]
LoopIR_UINT16 = Annotated[IntegerType, u16]
LoopIR_INT32: TypeAlias = I32
LoopIR_Bool: TypeAlias = BoolAttr
LoopIR_Index: TypeAlias = IntegerType
LoopIR_Size: TypeAlias = IntegerType
LoopIR_Stride: TypeAlias = IntegerType
LoopIR_Error: TypeAlias = NoneType  # needs investigation
LoopIR_Tensor: TypeAlias = TensorType[Float64Type]

# union types - might be wrong
LoopIR_Int: TypeAlias = LoopIR_INT8 | LoopIR_UINT8 | LoopIR_UINT16 | LoopIR_INT32
LoopIR_Num: TypeAlias = (
    LoopIR_F16 | LoopIR_F32 | LoopIR_F64 | LoopIR_INT8 | LoopIR_INT32
)

LoopIR_Object: TypeAlias = (
    LoopIR_F16
    | LoopIR_F32
    | LoopIR_F64
    | LoopIR_INT8
    | LoopIR_UINT8
    | LoopIR_UINT16
    | LoopIR_INT32
    | LoopIR_Bool
)

# -----------------------------------------------------------------------------
# LoopIR Stmt Operations
# -----------------------------------------------------------------------------


@irdl_op_definition
class AssignOp(IRDLOperation):
    name = "loop_ir.assign"


@irdl_op_definition
class ReduceOp(IRDLOperation):
    name = "loop_ir.reduce"


@irdl_op_definition
class WriteOp(IRDLOperation):
    name = "loop_ir.write"


@irdl_op_definition
class PassOp(IRDLOperation):
    name = "loop_ir.pass"


@irdl_op_definition
class ConditionalOp(IRDLOperation):
    name = "loop_ir.conditional"


@irdl_op_definition
class ForOp(IRDLOperation):
    name = "loop_ir.for"


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "loop_ir.alloc"


@irdl_op_definition
class FreeOp(IRDLOperation):
    name = "loop_ir.free"


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "loop_ir.call"


@irdl_op_definition
class WindowStmtOp(IRDLOperation):
    name = "loop_ir.window_stmt"


# -----------------------------------------------------------------------------
# LoopIR Expr Operations
# -----------------------------------------------------------------------------


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "loop_ir.read"


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "loop_ir.constant"
    value = attr_def(LoopIR_Object)
    res = result_def(LoopIR_Object)

    traits = traits_def(Pure())

    def __init__(self, value: LoopIR_Object):
        self.value = value
        self.res = LoopIR_Object

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
    name = "loop_ir.usub"


@irdl_op_definition
class BinOp(IRDLOperation):
    name = "loop_ir.binop"


@irdl_op_definition
class Extern(IRDLOperation):
    name = "loop_ir.extern"


@irdl_op_definition
class WindowExprOp(IRDLOperation):
    name = "loop_ir.window_expr"


@irdl_op_definition
class StrideExprOp(IRDLOperation):
    name = "loop_ir.stride_expr"


@irdl_op_definition
class CastOp(IRDLOperation):
    name = "loop_ir.cast"
