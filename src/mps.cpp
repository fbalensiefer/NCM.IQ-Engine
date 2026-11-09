#include <ATen/detail/MPSHooksInterface.h>

extern "C" void ncm_mps_synchronize() {
    at::detail::getMPSHooks().deviceSynchronize();
}

extern "C" size_t ncm_mps_allocated() {
    return at::detail::getMPSHooks().getCurrentAllocatedMemory();
}

extern "C" size_t ncm_mps_driver_allocated() {
    return at::detail::getMPSHooks().getDriverAllocatedMemory();
}
