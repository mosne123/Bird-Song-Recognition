#ifndef _ARM_COMPILER_SPECIFIC_H
#define _ARM_COMPILER_SPECIFIC_H

/* This is the bridge file missing from the Realtek/CMSIS-DSP mix */
#include "cmsis_compiler.h"

/* Ensure common macros are defined for the GCC compiler */
#ifndef __STATIC_FORCEINLINE
  #define __STATIC_FORCEINLINE __attribute__((always_inline)) static inline
#endif

#ifndef __ALIGNED
  #define __ALIGNED(x) __attribute__((aligned(x)))
#endif

#endif