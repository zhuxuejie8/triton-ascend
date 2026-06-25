set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name AddDynamicCVPipeline)
<<<<<<< HEAD
add_public_tablegen_target(AddDynamicCVPipelinePassIncGen)
=======
add_public_tablegen_target(AddDynamicCVPipelinePassIncGen)
>>>>>>> release-3.2.2-0625-b79d137
