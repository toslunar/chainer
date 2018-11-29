#include <memory>
#include <string>

#include "chainerx/context.h"
#include "chainerx/native/native_backend.h"

namespace {

class Backend0 : public chainerx::native::NativeBackend {
public:
    using NativeBackend::NativeBackend;

    std::string GetName() const override { return "backend0"; }
};

}  // namespace

extern "C" chainerx::Backend* CreateBackend(chainerx::Context& ctx) { return new Backend0{ctx}; }

extern "C" void DestroyBackend(chainerx::Backend* backend) { delete backend; }
