// CPU feature probe for the KleidiAI kernel families in `quantized::kleidiai`.
//
// In C rather than Rust because the probes are OS-specific syscalls (`sysctlbyname`,
// `getauxval`) that would otherwise make `libc` a non-optional dependency for one function.
// The bit values are mirrored in `kleidiai.rs`.
#include <stddef.h>
#include <stdint.h>

#define XN_KAI_DOTPROD 1u
#define XN_KAI_I8MM 2u
#define XN_KAI_SME 4u
#define XN_KAI_SME2 8u

#if defined(__APPLE__)
#include <sys/sysctl.h>

static int have(const char* name) {
    int v = 0;
    size_t sz = sizeof v;
    return sysctlbyname(name, &v, &sz, NULL, 0) == 0 && v != 0;
}

uint32_t xn_kleidiai_cpu_features(void) {
    uint32_t f = 0;
    if (have("hw.optional.arm.FEAT_DotProd")) f |= XN_KAI_DOTPROD;
    if (have("hw.optional.arm.FEAT_I8MM")) f |= XN_KAI_I8MM;
    if (have("hw.optional.arm.FEAT_SME")) f |= XN_KAI_SME;
    if (have("hw.optional.arm.FEAT_SME2")) f |= XN_KAI_SME2;
    return f;
}

// Cores in the fastest performance level (the performance cluster on Apple silicon), which
// is how many threads share its SME unit before the efficiency cores would join in. 0 when
// the OS does not say.
uint32_t xn_kleidiai_fast_cluster_cores(void) {
    int v = 0;
    size_t sz = sizeof v;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &v, &sz, NULL, 0) != 0 || v <= 0) return 0;
    return (uint32_t)v;
}

#elif defined(__linux__)
#include <sys/auxv.h>
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif
// Bit positions from the kernel's arch/arm64/include/uapi/asm/hwcap.h, for headers that
// predate a feature.
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#ifndef HWCAP2_SME
#define HWCAP2_SME (1UL << 23)
#endif
#ifndef HWCAP2_SME2
#define HWCAP2_SME2 (1UL << 37)
#endif

uint32_t xn_kleidiai_cpu_features(void) {
    const unsigned long h1 = getauxval(AT_HWCAP);
    const unsigned long h2 = getauxval(AT_HWCAP2);
    uint32_t f = 0;
    if (h1 & HWCAP_ASIMDDP) f |= XN_KAI_DOTPROD;
    if (h2 & HWCAP2_I8MM) f |= XN_KAI_I8MM;
    if (h2 & HWCAP2_SME) f |= XN_KAI_SME;
    if (h2 & HWCAP2_SME2) f |= XN_KAI_SME2;
    return f;
}

// Not derived here: Linux exposes SMIDR_EL1 per CPU under sysfs, which llama.cpp parses for
// the same purpose, but no SME Linux machine was available to check it against.
uint32_t xn_kleidiai_fast_cluster_cores(void) {
    return 0;
}

#else
// No probe for this OS: report nothing, and the storage stays off.
uint32_t xn_kleidiai_cpu_features(void) {
    return 0;
}

uint32_t xn_kleidiai_fast_cluster_cores(void) {
    return 0;
}
#endif
