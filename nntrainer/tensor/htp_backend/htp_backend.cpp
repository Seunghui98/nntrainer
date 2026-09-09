// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_backend.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP backend lifecycle implementation (HexKL micro-API FastRPC).
 */

#ifdef ENABLE_HEXKL

#include <htp_backend.h>

#include <nntrainer_log.h>

#include <cstring>
#include <string>

// remote_handle64, CDSP_DOMAIN_ID, remote_session_control -- Hexagon SDK,
// not the HexKL addon. nntr_hvx.h is generated from test/htp/nntr_hvx.idl by
// this directory's meson.build (the same IDL the DSP skel is built from).
#include <remote.h>

#include <nntr_hvx.h>

namespace nntrainer {

HtpBackend &HtpBackend::global() {
  static HtpBackend instance;
  return instance;
}

HtpBackend::HtpBackend() {
  // Enables the unsigned-PD CDSP session the dev/bring-up skel needs.
  // Not fatal if it fails -- a signed production skel does not need it,
  // and nntr_hvx_open below is the real pass/fail signal either way.
  remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
  int pd_err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                      &unsigned_pd, sizeof(unsigned_pd));
  if (pd_err != AEE_SUCCESS) {
    ml_logw("remote_session_control(unsigned PD) failed (err=%d); "
            "continuing -- a signed skel does not need it.",
            pd_err);
  }

  const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
  remote_handle64 h = 0;
  int err = nntr_hvx_open(uri.c_str(), &h);
  if (err != AEE_SUCCESS) {
    // Graceful disable: leave enabled_ = false so supports_*() reports
    // false and callers fall back to CPU. Not fatal.
    ml_logw("nntr_hvx_open failed (err=%d); HTP backend disabled, falling "
            "back to CPU. Is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?",
            err);
    return;
  }

  handle_ = static_cast<uint64_t>(h);
  enabled_ = true;

  // Poll-mode QoS: the host half of the measured 90 -> 3,900 us transport
  // spread (docs/htp_attention/34_fc_measured.md section4 item F; the
  // control call itself is test/unittest/htp_rpc_bench.h's
  // htp_set_latency_qos, ported here because this is the first production
  // caller -- every 34_fc_measured.md transport number assumes this is on,
  // and until now nothing in the model path turned it on). The DSP-side
  // half (DCVS/apptype vote) already runs unconditionally in
  // nntr_hvx_open's session setup; this is the other half of the same pair.
  // Best effort: an SDK or device without the control just keeps the old
  // interrupt-driven behavior, logged so a slow transport number is
  // explainable rather than silently misread as the kernel being slow.
  struct remote_rpc_control_latency lat;
  std::memset(&lat, 0, sizeof(lat));
  lat.enable = RPC_POLL_QOS;
  lat.latency = 100;
  int qos_err =
    remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, &lat, sizeof(lat));
  if (qos_err == AEE_SUCCESS) {
    qos_mode_ = 2;
  } else {
    std::memset(&lat, 0, sizeof(lat));
    lat.enable = RPC_PM_QOS;
    lat.latency = 100;
    qos_err =
      remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, &lat, sizeof(lat));
    qos_mode_ = (qos_err == AEE_SUCCESS) ? 1 : 0;
  }
  if (qos_mode_ == 0) {
    ml_logw("HtpBackend: poll and PM latency QoS both rejected (err=%d); "
            "transport will pay the interrupt-wake tail (34_fc_measured.md "
            "section4 item F).",
            qos_err);
  }
}

HtpBackend::~HtpBackend() {
  if (enabled_) {
    nntr_hvx_close(static_cast<remote_handle64>(handle_));
    enabled_ = false;
  }
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
