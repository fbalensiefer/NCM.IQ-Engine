#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/quantized/PackedParams.h>
#include <algorithm>
#include <vector>

using Packed = c10::intrusive_ptr<LinearPackedParamsBase>;

extern "C" void* ncm_int8_pack(const at::Tensor* weight, const at::Tensor* bias) {
    try {
        auto& context = at::globalContext();
        const auto& supported = context.supportedQEngines();
        if (context.qEngine() == at::QEngine::NoQEngine) {
            for (auto engine : {at::QEngine::X86, at::QEngine::FBGEMM, at::QEngine::QNNPACK}) {
                if (std::find(supported.begin(), supported.end(), engine) != supported.end()) {
                    context.setQEngine(engine);
                    break;
                }
            }
        }
        auto scale = std::max(weight->abs().max().item<double>() / 127.5, 1.1920928955078125e-7);
        auto quantized = at::quantize_per_tensor(*weight, scale, 0, at::kQInt8);
        std::vector<c10::IValue> stack{quantized, bias ? c10::IValue(*bias) : c10::IValue()};
        c10::Dispatcher::singleton().findSchemaOrThrow("quantized::linear_prepack", "").callBoxed(&stack);
        return new Packed(stack.back().toCustomClass<LinearPackedParamsBase>());
    } catch (...) {
        return nullptr;
    }
}

extern "C" at::Tensor* ncm_int8_linear(void* packed, const at::Tensor* input) {
    try {
        return new at::Tensor((*static_cast<Packed*>(packed))->apply_dynamic(*input, true));
    } catch (...) {
        return nullptr;
    }
}

extern "C" void ncm_int8_free(void* packed) {
    delete static_cast<Packed*>(packed);
}
