/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "common.hpp"
#include <vector>     // std::vector<>

// Global variable for output control - can be set by applications to disable output
// Default is true (output enabled)
bool outputEnabled_ = true;

// Set output enabled/disabled
void setOutputEnabled(bool enabled)
{
    outputEnabled_ = enabled;
}

// Output function that respects outpuEnabled_ flag
// This allows applications to suppress verbose output when needed
void output(const char* format, ...)
{
    if (!outputEnabled_)
        return;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    fflush(stdout);
}

// Output single character that respects outputEnabled_ flag
// Useful for progress indicators
void outputChar(char c)
{
    if (!outputEnabled_)
        return;
    putchar(c);
    fflush(stdout);
}

// Output a full integer vector, respecting outputEnabled_ flag
void outputVectorDump(const char* label, const std::vector<int32_t>& vec)
{
    if (!outputEnabled_)
        return;
    output("%s: [", label);
    for (size_t i = 0; i < vec.size(); ++i)
        output("%d%s", vec[i], (i < vec.size() - 1) ? " " : "");
    output("]\n");
}
