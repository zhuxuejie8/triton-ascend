/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_TYPES_H
#define TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_TYPES_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace gmload {

// One marked op and the topo-sorted dependency chain needed to reproduce it.
struct MarkedLoad {
<<<<<<< HEAD
  mlir::Operation *markedOp = nullptr;
  llvm::SmallVector<mlir::Operation *> chain;
  mlir::memref::AllocOp allocOp;
=======
    mlir::Operation *markedOp = nullptr;
    llvm::SmallVector<mlir::Operation *> chain;
    mlir::memref::AllocOp allocOp;
>>>>>>> release-3.2.2-0625-b79d137
};

// Each marked load gets its own independent LoadGroup with its own
// producer/consumer counter pair.
struct LoadGroup {
<<<<<<< HEAD
  llvm::SmallVector<MarkedLoad> loads;
  int depth = 0;
  llvm::SmallVector<llvm::SmallVector<mlir::Value>>
      bufSlots; // bufSlots[slot][load]
  llvm::SmallVector<mlir::Operation *> mergedChain;
=======
    llvm::SmallVector<MarkedLoad> loads;
    int depth = 0;
    llvm::SmallVector<llvm::SmallVector<mlir::Value>> bufSlots; // bufSlots[slot][load]
    llvm::SmallVector<mlir::Operation *> mergedChain;
>>>>>>> release-3.2.2-0625-b79d137
};

// Transformation context for one scf.for.
struct ForBufferCtx {
<<<<<<< HEAD
  mlir::scf::ForOp forOp;
  llvm::SmallVector<LoadGroup> groups;
=======
    mlir::scf::ForOp forOp;
    llvm::SmallVector<LoadGroup> groups;
>>>>>>> release-3.2.2-0625-b79d137
};

// Iter-arg handles for one LoadGroup inside the extended scf.for.
struct GroupIterArgs {
<<<<<<< HEAD
  llvm::SmallVector<mlir::Value> flagArgs;
  mlir::Value prodCounter;
  mlir::Value consCounter;
=======
    llvm::SmallVector<mlir::Value> flagArgs;
    mlir::Value prodCounter;
    mlir::Value consCounter;
>>>>>>> release-3.2.2-0625-b79d137
};

// Output of buildExtendedFor: new loop + per-group iter_arg handles.
struct ExtendedForInfo {
<<<<<<< HEAD
  mlir::scf::ForOp newForOp;
  mlir::Block *oldBody = nullptr;
  mlir::Block *newBody = nullptr;
  mlir::IRMapping mapping;
  llvm::SmallVector<GroupIterArgs> groupArgs;
  mlir::Value falseVal;
  int numOrig = 0;
  int depth = 0;
=======
    mlir::scf::ForOp newForOp;
    mlir::Block *oldBody = nullptr;
    mlir::Block *newBody = nullptr;
    mlir::IRMapping mapping;
    llvm::SmallVector<GroupIterArgs> groupArgs;
    mlir::Value falseVal;
    int numOrig = 0;
    int depth = 0;
>>>>>>> release-3.2.2-0625-b79d137
};

// One consecutive top-level run in an old loop body.
struct BodyRun {
<<<<<<< HEAD
  bool hasBlockId = false;
  int32_t blockId = 0;
  llvm::SmallVector<mlir::Operation *> ops;
=======
    bool hasBlockId = false;
    int32_t blockId = 0;
    llvm::SmallVector<mlir::Operation *> ops;
>>>>>>> release-3.2.2-0625-b79d137
};

} // namespace gmload

#endif // TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_TYPES_H
