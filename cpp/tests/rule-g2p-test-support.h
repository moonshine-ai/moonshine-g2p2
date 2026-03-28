#ifndef MOONSHINE_G2P_TESTS_RULE_G2P_TEST_SUPPORT_H
#define MOONSHINE_G2P_TESTS_RULE_G2P_TEST_SUPPORT_H

/// Shared helpers for rule-G2P parity tests (Python reference scripts, wiki samples).

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace moonshine_g2p::rule_g2p_test {

inline std::filesystem::path repo_root_from_tests_cpp(const char* tests_cpp_file) {
  return std::filesystem::path(tests_cpp_file).parent_path().parent_path().parent_path();
}

inline std::string shell_capture(const std::string& cmd) {
  FILE* pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    return {};
  }
  std::string out;
  char buf[8192];
  while (fgets(buf, sizeof(buf), pipe) != nullptr) {
    out += buf;
  }
  (void)pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  return out;
}

inline std::vector<std::string> split_unix_lines(std::string block) {
  std::vector<std::string> lines;
  std::istringstream iss(block);
  std::string L;
  while (std::getline(iss, L)) {
    if (!L.empty() && L.back() == '\r') {
      L.pop_back();
    }
    lines.push_back(std::move(L));
  }
  return lines;
}

/// Run ``python3`` on a ref script under ``cpp/tests/`` with ``PYTHONPATH`` set to the repo root.
/// *before_text_path* / *after_text_path* bracket the quoted wiki (or line) file path on the command line.
inline std::vector<std::string> python_ref_first_lines(
    const std::filesystem::path& repo, const char* script_name, const std::filesystem::path& text_file, int n,
    const std::vector<std::string>& before_text_path = {},
    const std::vector<std::string>& after_text_path = {}) {
  const std::filesystem::path script = repo / "cpp" / "tests" / script_name;
  std::ostringstream cmd;
  cmd << "env PYTHONPATH=" << std::quoted(repo.string()) << " python3 " << std::quoted(script.string());
  for (const std::string& a : before_text_path) {
    cmd << " " << a;
  }
  cmd << " " << std::quoted(text_file.string());
  for (const std::string& a : after_text_path) {
    cmd << " " << a;
  }
  cmd << " --first-lines " << n;
  return split_unix_lines(shell_capture(cmd.str()));
}

inline std::string python_ref_from_utf8_file(const std::filesystem::path& repo, const char* script_name,
                                             const std::filesystem::path& utf8_file,
                                             const std::vector<std::string>& extra_args_after_path = {}) {
  const std::filesystem::path script = repo / "cpp" / "tests" / script_name;
  std::ostringstream cmd;
  cmd << "env PYTHONPATH=" << std::quoted(repo.string()) << " python3 " << std::quoted(script.string()) << " "
      << std::quoted(utf8_file.string());
  for (const std::string& a : extra_args_after_path) {
    cmd << " " << a;
  }
  return shell_capture(cmd.str());
}

inline bool python_import_ok(const std::filesystem::path& repo, const char* dash_c_body) {
  std::ostringstream cmd;
  cmd << "env PYTHONPATH=" << std::quoted(repo.string()) << " python3 -c " << std::quoted(dash_c_body);
  return std::system(cmd.str().c_str()) == 0;
}

inline std::vector<std::string> read_text_first_lines(const std::filesystem::path& p, std::size_t n) {
  std::ifstream in(p);
  std::vector<std::string> src;
  std::string line;
  while (src.size() < n && std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    src.push_back(std::move(line));
  }
  return src;
}

}  // namespace moonshine_g2p::rule_g2p_test

#endif  // MOONSHINE_G2P_TESTS_RULE_G2P_TEST_SUPPORT_H
