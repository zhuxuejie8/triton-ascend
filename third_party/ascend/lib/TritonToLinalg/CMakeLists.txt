add_triton_library(TritonToLinalg
        TritonToLinalgPass.cpp
        LoadStoreConverter.cpp
        FunctionConverter.cpp
        ArgMinMaxConverter.cpp
        TritonOpConverter.cpp
        HoistBroadcast.cpp
        BlockPtrAnalysis.cpp
        MaskAnalysis.cpp
        UseAnalysis.cpp
        ImplicitPermute.cpp
<<<<<<< HEAD
        DescriptorConverter.cpp
=======
        StridedLoadStoreRewrite.cpp
        StridedAxisCoalescing.cpp
        TileChunkCoalescing.cpp
        DescriptorConverter
>>>>>>> release-3.2.2-0625-b79d137
        MarkTensorKindPass.cpp
        DevicePrintOffsetRewrite.cpp

  DEPENDS
  TritonToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  BiShengIRHIVMDialect
  TritonIR
  TritonTransforms
  TritonAnalysis
  MLIRTritonNPUUtils
  MLIRSCFTransforms
  MLIRLinalgTransforms
)
