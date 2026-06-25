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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/CloneOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
<<<<<<< HEAD
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "CloneOps";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::outs() << __VA_ARGS__;                                               \
    llvm::outs() << "\n";                                                      \
  })
=======
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

static constexpr const char *DEBUG_TYPE = "CloneOps";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::outs() << __VA_ARGS__; \
  llvm::outs() << "\n"; \
})
>>>>>>> release-3.2.2-0625-b79d137

using namespace mlir;
using namespace triton;
using namespace hivm;

<<<<<<< HEAD
// Update op operands using value mapping, skip yield values of forOp
static LogicalResult
updateCloneMapping(Operation *op, llvm::DenseMap<Value, Value> &valueMap,
                   const llvm::DenseSet<Value> &yieldValues) {
=======
using MemDepGraph = std::unique_ptr<CVPipeline::MemoryDependenceGraph>;
using MemDepGraphT = CVPipeline::MemoryDependenceGraph;

// Update op operands using value mapping, skip yield values of forOp
static LogicalResult updateCloneMapping(Operation *op,
    llvm::DenseMap<Value, Value> &valueMap,
    const llvm::DenseSet<Value> &yieldValues)
{
>>>>>>> release-3.2.2-0625-b79d137
  if (!op) {
    return failure();
  }

  for (OpOperand &operand : op->getOpOperands()) {
    // Only skip if this operand is a yield value from the main_loop forOp
    // Nested yield ops (in scf.if/scf.for) should have their operands updated
    Value v = operand.get();
    if (yieldValues.contains(v)) {
      continue;
    }

    auto it = valueMap.find(v);
    if (it != valueMap.end()) {
      if (it->second.getType() != v.getType()) {
<<<<<<< HEAD
        LDBG("[Error]: type mismatch in value mapping: "
             << v.getType() << " vs " << it->second.getType() << "\n");
=======
        LDBG("[Error]: type mismatch in value mapping: " << v.getType() << " vs " << it->second.getType() << "\n");
>>>>>>> release-3.2.2-0625-b79d137
        return failure();
      }
      operand.set(it->second);
    }
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (failed(updateCloneMapping(&nestedOp, valueMap, yieldValues))) {
          return failure();
        }
      }
    }
  }

  return success();
}

// Clone a single op with IRMapping
<<<<<<< HEAD
static Operation *cloneOpWithMapping(Operation *op, OpBuilder &builder,
                                     llvm::DenseMap<Value, Value> &valueMap) {
  IRMapping mapper;
  for (auto result : op->getResults()) {
    if (valueMap.count(result)) {
      mapper.map(result, valueMap[result]);
    }
=======
static Operation *cloneOpWithMapping(Operation *op,
    OpBuilder &builder,
    llvm::DenseMap<Value, Value> &valueMap)
{
  IRMapping mapper;
  // Populate mapper with ALL previously cloned values (not just the current op's results). 
  for (const auto &entry : valueMap) {
    mapper.map(entry.first, entry.second);
>>>>>>> release-3.2.2-0625-b79d137
  }

  Operation *cloned = builder.clone(*op, mapper);
  for (auto it : llvm::zip(op->getResults(), cloned->getResults())) {
    valueMap[std::get<0>(it)] = std::get<1>(it);
  }

  return cloned;
}

// Clone ops for a single block in vector/cube mode
<<<<<<< HEAD
static LogicalResult
cloneOpsForBlock(int curId, SmallVector<Operation *> &curOps,
                 const SmallVector<int> &earlierIds,
                 const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
                 scf::ForOp forOp) {
=======
static LogicalResult cloneOpsForBlock(int curId, SmallVector<Operation *> &curOps,
    const SmallVector<int> &earlierIds,
    const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
    scf::ForOp forOp)
{
>>>>>>> release-3.2.2-0625-b79d137
  if (curOps.empty() || earlierIds.empty()) {
    return success();
  }

  // Collect all ops from earlier blocks to clone
  SmallVector<Operation *> toClone;
  for (int eid : earlierIds) {
    llvm::append_range(toClone, blockOps.lookup(eid));
  }
  if (toClone.empty()) {
    return success();
  }

  llvm::DenseMap<Value, Value> valueMap;
  SmallVector<Operation *> clonedOps;
  OpBuilder builder(curOps.front());

  for (Operation *op : toClone) {
    Operation *cloned = cloneOpWithMapping(op, builder, valueMap);
<<<<<<< HEAD
    cloned->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(curId));
    if (auto origBlockIdAttr =
            op->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
      cloned->setAttr("ssbuffer.clone", origBlockIdAttr);
=======
    cloned->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(curId));
    if (auto origBlockIdOpt = CVPipeline::getOpBlockId(op)) {
      cloned->setAttr(CVPipeline::kClone, builder.getI32IntegerAttr(static_cast<int32_t>(*origBlockIdOpt)));
>>>>>>> release-3.2.2-0625-b79d137
    }
    clonedOps.push_back(cloned);
  }

  curOps.insert(curOps.begin(), clonedOps.begin(), clonedOps.end());

  llvm::DenseSet<Value> yieldValues;
  if (auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
    for (Value operand : yieldOp.getOperands()) {
      yieldValues.insert(operand);
    }
  }

  for (Operation *op : curOps) {
    if (failed(updateCloneMapping(op, valueMap, yieldValues))) {
      return failure();
    }
  }

  if (failed(topologicalSort(curOps))) {
    return failure();
  }

  return success();
}

// Clone ops for all blocks in main loop
<<<<<<< HEAD
LogicalResult CloneOpsPass::cloneOpsInMainLoop(scf::ForOp forOp) {
=======
LogicalResult CloneOpsPass::cloneOpsInMainLoop(scf::ForOp forOp)
{
>>>>>>> release-3.2.2-0625-b79d137
  llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
  if (failed(collectOpsByBlockId(forOp, blockOps))) {
    return failure();
  }

  SmallVector<int> idsInOrder = getBlockIdsInOrder(forOp);

  for (int i = idsInOrder.size() - 1; i >= 0; --i) {
    int curId = idsInOrder[i];
    SmallVector<int> earlierIds(idsInOrder.begin(), idsInOrder.begin() + i);

<<<<<<< HEAD
    if (failed(cloneOpsForBlock(curId, blockOps[curId], earlierIds, blockOps,
                                forOp))) {
=======
    if (failed(cloneOpsForBlock(curId, blockOps[curId], earlierIds, blockOps, forOp))) {
>>>>>>> release-3.2.2-0625-b79d137
      return failure();
    }
  }

  return success();
}

// Check if an op should be erased during cleanup (for cube)
<<<<<<< HEAD
static bool shouldEraseOpForCube(Operation *op) {
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (ifOp->hasAttr("ssbuffer.cross_buffer") ||
        ifOp->hasAttr("ssbuffer.intra_buffer") ||
        ifOp->hasAttr("ssbuffer.load_store")) {
      return true;
    }
  }

=======
// memGraph: optional pointer to MemoryDependenceGraph for checking exec-after dependencies
static bool shouldEraseOpForCube(Operation *op, const CVPipeline::MemoryDependenceGraph *memGraph = nullptr)
{
  // Rule 1: SyncBlockWaitOp, SyncBlockSetOp, FixpipeOp -> directly erase
>>>>>>> release-3.2.2-0625-b79d137
  if (isa<SyncBlockWaitOp>(op) || isa<SyncBlockSetOp>(op) ||
      isa<hivm::FixpipeOp>(op)) {
    return true;
  }

<<<<<<< HEAD
  // Copy and fill ops: erase if operand not used elsewhere
  if (isa<memref::CopyOp>(op) ||
      (isa<linalg::FillOp>(op) && op->getNumResults() == 0)) {
    if (op->getNumOperands() > 1) {
      Value secondOperand = op->getOperand(1);
      bool usedByOtherOp =
          llvm::any_of(secondOperand.getUsers(),
                       [&](Operation *user) { return user != op; });
      if (!usedByOtherOp) {
        return true;
      }
    }
  }

  // Erase ops with no used results
  return llvm::none_of(op->getResults(),
                       [](auto result) { return !result.use_empty(); });
}

// Check if an op should be erased (for vector)
static bool shouldEraseOpForVector(Operation *op) {
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (ifOp->hasAttr("ssbuffer.cross_buffer") ||
        ifOp->hasAttr("ssbuffer.intra_buffer") ||
        ifOp->hasAttr("ssbuffer.load_store")) {
      return true;
    }
  }

=======
  auto opBlockId = getForDirectChildBlockId(op);

  // Rule 2: If op has results, check via SSA if result is used by later ops in same block_id
  // Use for-direct-child block_id (the immediate child of scf.for) for comparison
  if (op->getNumResults() > 0) {
    for (auto result : op->getResults()) {
      if (result.use_empty()) {
        // Result not used by anyone, can erase
        continue;
      }
      if (!opBlockId) {
        // No block_id but result is used, be conservative and keep
        return false;
      }
      // Check if any user is in the same for-direct-child block_id
      bool usedInSameBlockId = llvm::any_of(result.getUsers(), [&](Operation *user) {
        auto userBlockId = getForDirectChildBlockId(user);
        return userBlockId && *userBlockId == *opBlockId;
      });
      if (usedInSameBlockId) {
        // Result used in same for-direct-child block, cannot erase
        return false;
      }
    }
    // All results are either unused or not used in same for-direct-child block, can erase
    return true;
  }

  // Rule 3: If op has no results, check getExecAfter for same block_id dependencies
  if (memGraph) {
    auto execAfterOps = memGraph->getExecAfter(op);
    bool hasCloneExecAfterInSameBlockId = llvm::any_of(execAfterOps, [&](Operation *execOp) {
      // sync_block_wait/sync_block_set ops are not memory side effects in analyzing cleanup ops,
      // therefore, we need to skip their judgments
      if (isa<SyncBlockWaitOp>(execOp) || isa<SyncBlockSetOp>(execOp)) {
        return false;
      }
      auto execBlockId = CVPipeline::getOpBlockId(execOp);
      return execBlockId && opBlockId && *execBlockId == *opBlockId;
    });
    if (hasCloneExecAfterInSameBlockId) {
      return false;
    }
  }

  // Can erase if no results and no exec-after dependencies in same block
  return true;
}

// Check if an op should be erased (for vector)
static bool shouldEraseOpForVector(Operation *op)
{
>>>>>>> release-3.2.2-0625-b79d137
  return llvm::none_of(op->getResults(),
                       [](auto result) { return !result.use_empty(); });
}

// Cleanup for cloned ops in a forOp
<<<<<<< HEAD
static LogicalResult
cleanupClonedOps(scf::ForOp forOp,
                 llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
                 const SmallVector<int> &idsInOrder, bool isCube) {
=======
// memGraphFactory: callable that rebuilds MemoryDependenceGraph for current IR state
static LogicalResult cleanupClonedOps(scf::ForOp forOp,
    llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
    const SmallVector<int> &idsInOrder,
    bool isCube,
    std::function<MemDepGraph(scf::ForOp)> memGraphFactory)
{
>>>>>>> release-3.2.2-0625-b79d137
  for (int i = idsInOrder.size() - 1; i >= 0; --i) {
    auto &curOps = blockOps[idsInOrder[i]];
    if (curOps.empty()) {
      continue;
    }

    // Find last index of cloned ops
    int startIdx = -1;
    for (int j = curOps.size() - 1; j >= 0; --j) {
<<<<<<< HEAD
      if (curOps[j]->hasAttr("ssbuffer.clone")) {
=======
      if (curOps[j]->hasAttr(CVPipeline::kClone)) {
>>>>>>> release-3.2.2-0625-b79d137
        startIdx = j;
        break;
      }
    }
    if (startIdx < 0) {
      continue;
    }

<<<<<<< HEAD
    // Erase cloned ops that are not needed
    for (int j = startIdx; j >= 0; --j) {
      Operation *op = curOps[j];
      if (!op->hasAttr("ssbuffer.clone")) {
        break;
      }
      bool shouldErase =
          isCube ? shouldEraseOpForCube(op) : shouldEraseOpForVector(op);
=======
    // Erase cloned ops from bottom to top
    // This ensures that when checking if an op can be erased, its users in same block_id
    // have already been processed (and erased if applicable)
    for (int j = startIdx; j >= 0; --j) {
      Operation *op = curOps[j];
      if (!op->hasAttr(CVPipeline::kClone)) {
        break;
      }
      // Rebuild memGraph after each erasure to reflect current IR state
      auto memGraph = memGraphFactory(forOp);
      bool shouldErase = isCube ? shouldEraseOpForCube(op, memGraph.get()) : shouldEraseOpForVector(op);
>>>>>>> release-3.2.2-0625-b79d137
      if (shouldErase) {
        op->erase();
      }
    }
  }

  // Check whether new forOp is valid after cleanup
  for (Operation &op : forOp.getBody()->without_terminator()) {
<<<<<<< HEAD
    if (op.hasAttr("ssbuffer.clone")) {
      if (isa<SyncBlockWaitOp>(op) || isa<SyncBlockSetOp>(op) ||
          isa<hivm::FixpipeOp>(op)) {
        LDBG("[ERROR]: Cloned sync/fixpipe op should have been erased: "
             << op.getName() << "\n");
=======
    if (op.hasAttr(CVPipeline::kClone)) {
      if (isa<SyncBlockWaitOp>(op) || isa<SyncBlockSetOp>(op) ||
          isa<hivm::FixpipeOp>(op)) {
        LDBG("[ERROR]: Cloned sync/fixpipe op should have been erased: " << op.getName() << "\n");
>>>>>>> release-3.2.2-0625-b79d137
        return failure();
      }
    }
  }

  return success();
}

// Cleanup cloned ops for a single main loop
<<<<<<< HEAD
LogicalResult CloneOpsPass::cleanupClonedOpsInMainLoop(scf::ForOp forOp) {
=======
LogicalResult CloneOpsPass::cleanupClonedOpsInMainLoop(scf::ForOp forOp)
{
>>>>>>> release-3.2.2-0625-b79d137
  ModuleOp module = getOperation();
  scope::ScopeOp scopeOp = forOp->getParentOfType<scope::ScopeOp>();
  if (!scopeOp) {
    return success();
  }

<<<<<<< HEAD
  auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
=======
  auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(CVPipeline::kTcoreType);
>>>>>>> release-3.2.2-0625-b79d137
  if (!attr) {
    return success();
  }

<<<<<<< HEAD
  bool isCube = (attr == hivm::TCoreTypeAttr::get(module.getContext(),
                                                  hivm::TCoreType::CUBE));
=======
  bool isCube = (attr == hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE));
>>>>>>> release-3.2.2-0625-b79d137

  llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
  if (failed(collectOpsByBlockId(forOp, blockOps))) {
    return failure();
  }

  SmallVector<int> idsInOrder = getBlockIdsInOrder(forOp);
<<<<<<< HEAD
  if (failed(cleanupClonedOps(forOp, blockOps, idsInOrder, isCube))) {
=======
  if (failed(cleanupClonedOps(forOp, blockOps, idsInOrder, isCube, [&](scf::ForOp forOp) -> MemDepGraph {
    if (!isCube) {
      return nullptr;
    }
    auto &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    return std::make_unique<MemDepGraphT>(forOp, aliasAnalysis);
  }))) {
>>>>>>> release-3.2.2-0625-b79d137
    return failure();
  }

  return success();
}

<<<<<<< HEAD
// Validate that each block_id's ops form contiguous ranges (not interleaved
// with other ids) e.g., [1,1,2,2] is valid, but [1,2,1,2] is invalid
static bool areBlockIdsConsecutive(scf::ForOp forOp) {
  SmallVector<int> idsInOrder;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr) {
=======
// Validate that each block_id's ops form contiguous ranges (not interleaved with other ids)
// e.g., [1,1,2,2] is valid, but [1,2,1,2] is invalid
static bool areBlockIdsConsecutive(scf::ForOp forOp)
{
  SmallVector<int> idsInOrder;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto blockIdOpt = CVPipeline::getOpBlockId(&op);
    if (!blockIdOpt) {
>>>>>>> release-3.2.2-0625-b79d137
      LDBG("[ERROR]: Op missing ssbuffer.block_id: " << op.getName() << "\n");
      return false;
    }

<<<<<<< HEAD
    idsInOrder.push_back(blockIdAttr.getInt());
=======
    idsInOrder.push_back(static_cast<int>(*blockIdOpt));
>>>>>>> release-3.2.2-0625-b79d137
  }

  // Check that each block_id forms a contiguous range
  for (size_t i = 0; i < idsInOrder.size();) {
    int currentId = idsInOrder[i];
    size_t j = i;

    while (j < idsInOrder.size() && idsInOrder[j] == currentId) {
      ++j;
    }

    for (size_t k = j; k < idsInOrder.size(); ++k) {
      if (idsInOrder[k] == currentId) {
        LDBG("[ERROR]: block_id: " << currentId << " is interleaved\n");
        return false;
      }
    }

    i = j;
  }

  return true;
}

<<<<<<< HEAD
LogicalResult CloneOpsPass::validateBlockIdsConsecutive(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
=======
LogicalResult CloneOpsPass::validateBlockIdsConsecutive(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(CVPipeline::kMainLoop)) {
>>>>>>> release-3.2.2-0625-b79d137
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }
    if (!areBlockIdsConsecutive(forOp)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  return success();
}

<<<<<<< HEAD
void CloneOpsPass::runOnOperation() {
=======
// Check that no op in a VECTOR scope's main_loop forOp has a tensor result \
// carrying the ssbuffer.clone attribute.
LogicalResult CloneOpsPass::validateClonedOpsInVector(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(CVPipeline::kMainLoop)) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }

    scope::ScopeOp scopeOp = forOp->getParentOfType<scope::ScopeOp>();
    if (!scopeOp) {
      return WalkResult::advance();
    }

    auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(CVPipeline::kTcoreType);
    if (!attr) {
      return WalkResult::advance();
    }

    if (attr != hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR)) {
      return WalkResult::advance();
    }

    for (Operation &bodyOp : forOp.getBody()->without_terminator()) {
      if (!bodyOp.hasAttr(CVPipeline::kClone)) {
        continue;
      }
      bool hasTensorDep = llvm::any_of(bodyOp.getResults(), [](Value result) {
        return isa<RankedTensorType>(result.getType());
      });
      if (hasTensorDep) {
        LDBG("[Error]: VECTOR main_loop contains cloned op with tensor type: "
             << bodyOp.getName() << "\n");
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}

void CloneOpsPass::runOnOperation()
{
>>>>>>> release-3.2.2-0625-b79d137
  ModuleOp module = getOperation();

  LDBG("before cloneOps:\n" << module << "\n");

  // Validate block_ids are consecutive before cloning
<<<<<<< HEAD
  if (failed(validateBlockIdsConsecutive(module)))
    return;
=======
  if (failed(validateBlockIdsConsecutive(module))) {
    signalPassFailure();
    return;
  }
>>>>>>> release-3.2.2-0625-b79d137

  // Clone ops in vector/cube to ensure that each block_id has its own
  // ops without sharing
  module.walk([&](Operation *op) -> WalkResult {
<<<<<<< HEAD
    if (!op->hasAttr("ssbuffer.main_loop")) {
=======
    if (!op->hasAttr(CVPipeline::kMainLoop)) {
>>>>>>> release-3.2.2-0625-b79d137
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      signalPassFailure();
      return WalkResult::interrupt();
    }

    if (failed(cloneOpsInMainLoop(forOp))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }

    if (failed(cleanupClonedOpsInMainLoop(forOp))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
<<<<<<< HEAD

  LDBG("after cloneOps:\n" << module << "\n");
=======
  
  LDBG("after cloneOps:\n" << module << "\n");

  // Validate no cloned tensor ops remaining in VECTOR main_loop forOp
  if (failed(validateClonedOpsInVector(module))) {
    signalPassFailure();
    return;
  }

>>>>>>> release-3.2.2-0625-b79d137
}

namespace mlir {
namespace triton {

<<<<<<< HEAD
std::unique_ptr<OperationPass<ModuleOp>> createCloneOpsPass() {
=======
std::unique_ptr<OperationPass<ModuleOp>> createCloneOpsPass()
{
>>>>>>> release-3.2.2-0625-b79d137
  return std::make_unique<CloneOpsPass>();
}

} // namespace triton
} // namespace mlir
