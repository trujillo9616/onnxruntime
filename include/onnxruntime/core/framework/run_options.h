
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <atomic>
#include "core/session/onnxruntime_c_api.h"

/**
 * Configuration information for a single Run.
 */
struct OrtRunOptions {
  unsigned run_log_verbosity_level = 0;  ///< applies to a particular Run() invocation
  std::string run_tag;                   ///< to identify logs generated by a particular Run() invocation

  /// set to 'true' to terminate any currently executing Run() calls that are using this
  /// OrtRunOptions instance. the individual calls will exit gracefully and return an error status.
  bool terminate = false;
  OrtRunOptions() = default;
  ~OrtRunOptions() = default;

  // disable copy, move and assignment. we don't want accidental copies, to ensure that the instance provided to
  // the Run() call never changes and the terminate mechanism will work.
  OrtRunOptions(const OrtRunOptions&) = delete;
  OrtRunOptions(OrtRunOptions&&) = delete;
  OrtRunOptions& operator=(const OrtRunOptions&) = delete;
  OrtRunOptions& operator=(OrtRunOptions&&) = delete;
};

namespace onnxruntime {
using RunOptions = OrtRunOptions;
}
