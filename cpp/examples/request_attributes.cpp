/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file request_attributes.cpp
 * @brief Demonstrates how to read `ucxx::Request::queryAttributes()` and interpret the
 *        debug strings UCX returns.
 *
 * The example submits one tag-send and one tag-recv over a loopback worker, then prints
 * the raw debug string for each and a best-effort parse of the protocol, memory-type and
 * transport fields.
 */

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <ucxx/api.h>

namespace {

/**
 * @brief Parsed fields extracted from a UCX request debug string.
 *
 * Each field is best-effort: missing fields are left empty and the caller should treat
 * the contents as informational rather than authoritative.
 */
struct TransportSummary {
  std::string protocolMarker;  ///< Leading `{...}` marker, e.g. `{proto|cb|init}` or
                               ///< `{cb|strm_r_wtall}`
  std::string protocolName;    ///< Protocol token after the marker, e.g. `stream`, `tag`,
                               ///< `am` (best-effort)
  std::string direction;       ///< Memory-direction phrase, e.g. `from host memory` or
                               ///< `to cuda memory`
  std::string lengthBytes;     ///< Numeric byte count, or empty if not present
  std::string transports;      ///< Transport identifiers, e.g. `tcp/eth0`, `cuda_ipc/cuda0`,
                               ///< `rc/mlx5_0:1`
  std::string bandwidth;       ///< Bandwidth report when present, e.g. `1017.7 MB/s`
  std::string latency;         ///< Latency report when present, e.g. `3930.62 us`
  bool noDebugInfo{false};     ///< Set when UCX returned the literal `<no debug info>`
};

/**
 * @brief Best-effort parse of a UCX debug string into a `TransportSummary`.
 *
 * UCX's debug string layout is informal, so the markers here are heuristics that cover
 * the common cases observed in our testing. Anything we don't recognise is reported as
 * empty.
 *
 * @param[in] debugString  the raw debug string returned by `Request::queryAttributes()`.
 * @return a `TransportSummary` populated with whichever fields could be recognised.
 */
TransportSummary parseDebugString(const std::string& debugString)
{
  TransportSummary out;

  // The protocol marker is the first {...} group at the start of the string,
  // e.g. "{proto|cb|init}" or "{cb|strm_r_wtall}".
  std::regex markerRe(R"(^\{[^}]+\})");
  std::smatch m;
  if (std::regex_search(debugString, m, markerRe)) out.protocolMarker = m.str();

  if (debugString.find("<no debug info>") != std::string::npos) {
    out.noDebugInfo = true;
    return out;
  }

  // After the marker UCX usually emits a "<proto> <direction> length N ..."
  // sequence. We pull the first identifier-looking token after the marker as the
  // protocol name, and find "length N" and the memory-type direction by simple
  // substring search.
  std::regex protoNameRe(R"(\}\s+([a-z_]+)\s+)");
  if (std::regex_search(debugString, m, protoNameRe)) out.protocolName = m[1].str();

  std::regex lengthRe(R"(length\s+(\d+))");
  if (std::regex_search(debugString, m, lengthRe)) out.lengthBytes = m[1].str();

  // The direction phrase varies: send paths usually say "from/to host memory" or
  // "from/to cuda memory", while receive paths often emit just "host memory" /
  // "cuda memory". Match both forms and keep whichever appears first.
  std::regex memDirRe(R"(((?:from |to )?(?:host|cuda) memory))");
  if (std::regex_search(debugString, m, memDirRe)) out.direction = m[1].str();

  // Transport identifier convention: name/device, e.g. tcp/eth0, cuda_ipc/cuda0,
  // rc/mlx5_0:1, sysv/memory, posix/memory, self/memory.
  std::regex transportRe(R"(\b([a-z][a-z0-9_]*)/([A-Za-z0-9_:\.]+))");
  std::vector<std::string> transportTokens;
  auto begin = std::sregex_iterator(debugString.begin(), debugString.end(), transportRe);
  auto end   = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    // Skip the protocol-name match if it slipped through.
    std::string token = it->str();
    if (std::find(transportTokens.begin(), transportTokens.end(), token) == transportTokens.end())
      transportTokens.push_back(std::move(token));
  }
  if (!transportTokens.empty()) {
    std::ostringstream oss;
    for (size_t i = 0; i < transportTokens.size(); ++i) {
      if (i) oss << ", ";
      oss << transportTokens[i];
    }
    out.transports = oss.str();
  }

  // Ending with bandwidth and latency reports.
  std::regex bwRe(R"(([\d.]+\s*[KMG]?B/s))");
  if (std::regex_search(debugString, m, bwRe)) out.bandwidth = m[1].str();

  std::regex latRe(R"(([\d.]+\s*[un]?s)\b)");
  if (std::regex_search(debugString, m, latRe)) out.latency = m[1].str();

  return out;
}

/**
 * @brief Print a labeled summary of a request's debug string to `std::cout`.
 *
 * Prints the raw debug string followed by each parsed field. When UCX returned no
 * per-protocol payload (the `<no debug info>` marker), only the marker line is printed
 * along with an explanatory note.
 *
 * @param[in] label        a human-readable label identifying which request is being
 *                         summarised (e.g. `"tag send"`).
 * @param[in] debugString  the raw debug string returned by `Request::queryAttributes()`.
 */
void printSummary(const std::string& label, const std::string& debugString)
{
  std::cout << "=== " << label << " ===\n";
  std::cout << "raw debug_string: " << (debugString.empty() ? "<empty>" : debugString) << "\n";

  auto t = parseDebugString(debugString);
  std::cout << "  protocol marker  : " << (t.protocolMarker.empty() ? "-" : t.protocolMarker)
            << "\n";
  if (t.noDebugInfo) {
    std::cout << "  (UCX emitted no per-protocol payload for this request; the marker above is\n"
                 "   the only attribute available. This is the typical shape for legacy stream\n"
                 "   recv and other request types that don't carry a debug payload.)\n\n";
    return;
  }
  std::cout << "  protocol name    : " << (t.protocolName.empty() ? "-" : t.protocolName) << "\n";
  std::cout << "  memory direction : " << (t.direction.empty() ? "-" : t.direction) << "\n";
  std::cout << "  length (bytes)   : " << (t.lengthBytes.empty() ? "-" : t.lengthBytes) << "\n";
  std::cout << "  transports       : " << (t.transports.empty() ? "-" : t.transports) << "\n";
  std::cout << "  bandwidth        : " << (t.bandwidth.empty() ? "-" : t.bandwidth) << "\n";
  std::cout << "  latency          : " << (t.latency.empty() ? "-" : t.latency) << "\n\n";
}

}  // namespace

/**
 * @brief Entry point: build a loopback worker, exchange one tag pair, print attributes.
 *
 * Constructs a `ucxx::Context` and builds a `ucxx::Worker` with `requestAttributes(true)`
 * so that `ucp_request_query` is invoked on the UCP handle of every submitted request.
 * A loopback endpoint is created from the worker's own address and one host-buffer
 * tag-send / tag-recv pair is exchanged. The raw debug string and a best-effort parse
 * are printed for each.
 *
 * Use the `UCX_TLS` environment variable to constrain the transport set (e.g.
 * `UCX_TLS=tcp`, `UCX_TLS=sm,self`). The printed `transports` line will reflect what
 * UCX actually picked.
 *
 * @return 0 on success.
 */
int main()
{
  std::cout << "UCXX request attributes example\n"
               "-------------------------------\n"
               "Builds a worker with requestAttributes(true), exchanges one host\n"
               "tag pair on a loopback endpoint, and prints both the raw debug\n"
               "string returned by `Request::queryAttributes()` and a best-effort\n"
               "parse of the protocol and transport fields.\n\n"
               "Use `UCX_TLS` in the environment to force a specific transport set\n"
               "(e.g. `UCX_TLS=tcp ./ucxx_example_request_attributes`,\n"
               "      `UCX_TLS=sm,self ./ucxx_example_request_attributes`).\n\n";

  auto context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
  auto worker  = ucxx::experimental::createWorker(context).requestAttributes(true).build();
  auto ep      = worker->createEndpointFromWorkerAddress(worker->getAddress());

  constexpr size_t kSize = 1024 * 1024;  // 1 MiB (above rndv threshold using default settings)
  std::vector<char> sendBuf(kSize, 'A');
  std::vector<char> recvBuf(kSize, 0);
  const ucxx::Tag tag{42};

  auto sendReq = ep->tagSend(sendBuf.data(), kSize, tag);
  auto recvReq = ep->tagRecv(recvBuf.data(), kSize, tag, ucxx::TagMaskFull);
  for (auto& r : {sendReq, recvReq}) {
    while (!r->isCompleted())
      worker->progress();
    r->checkError();
  }

  std::string sendDebug;
  std::string recvDebug;
  try {
    sendDebug = sendReq->queryAttributes().debugString;
  } catch (const ucxx::NoElemError&) {
    sendDebug = "<no UCP request to query: UCX took an inline path>";
  }
  try {
    recvDebug = recvReq->queryAttributes().debugString;
  } catch (const ucxx::NoElemError&) {
    recvDebug = "<no UCP request to query: UCX took an inline path>";
  }

  printSummary("tag send (proto v2, usually rich)", sendDebug);
  printSummary("tag recv (legacy path, often just a marker)", recvDebug);

  std::cout << "Tips:\n"
               "  - Run with UCX_TLS=<transports> to constrain the transport set, e.g.\n"
               "      UCX_TLS=tcp        (TCP only)\n"
               "      UCX_TLS=sm,self    (shared memory + self loopback)\n"
               "      UCX_TLS=tcp,cuda_copy,cuda_ipc  (CUDA copy + cuda_ipc)\n"
               "      UCX_TLS=rc         (Reliable Connection over InfiniBand)\n"
               "    The `transports` line above will reflect what UCX actually picked.\n"
               "  - The debug-string wording is informal and version-dependent. This parser uses\n"
               "    heuristics; treat the parsed fields as hints, not as a contract.\n";

  return 0;
}
