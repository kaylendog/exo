// RUN: exocc ./filecheck.py -t mlir -o - | filecheck %s

builtin.module {
  func.func @filecheck(%0 : i32, %1 : i32) {
    %2 = arith.constant 0 : i32
    %3 = arith.cmpi sgt, %4, %2 : i32
    scf.if %3 {
    } else {
    }
    %5 = arith.constant 0 : i32
    %6 = arith.cmpi sgt, %4, %5 : i32
    %7 = arith.constant 0 : i32
    %8 = arith.cmpi sgt, %9, %7 : i32
    %10 = arith.andi %6, %8 : i1
    scf.if %10 {
    } else {
    }
    func.return
  }
}
