#include "session_logits.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using edgeflow::SessionLogits;

void check(bool condition) {
  if (!condition) {
    throw std::runtime_error("check failed");
  }
}

void rejects_argmax(SessionLogits &logits) {
  try {
    (void)logits.argmax();
  } catch (const std::runtime_error &) {
    return;
  }
  throw std::runtime_error("invalid logits were accepted");
}

int main(int argc, char **argv) {
  if (argc != 2) {
    return 2;
  }
  const std::string test = argv[1];
  const float first[] = {-2.0f, 5.0f, 5.0f, -0.0f};
  const float second[] = {8.0f, 1.0f, 2.0f, 3.0f};
  SessionLogits logits;
  logits.assign(first, 4);
  if (test == "argmax") {
    check(logits.argmax() == 1);
    check(logits.argmax() == 1);
    const float zeros[] = {-0.0f, 0.0f};
    logits.assign(zeros, 2);
    check(logits.argmax() == 0);
    const float negatives[] = {-8.0f, -2.0f, -2.0f, -3.0f};
    logits.assign(negatives, 4);
    check(logits.argmax() == 1);
    for (const size_t count : {1u, 3u, 4u, 5u, 31u, 151936u}) {
      std::vector<float> values(count, -3.0f);
      const size_t first_best = count / 3;
      values[first_best] = 7.0f;
      values.back() = 7.0f;
      logits.assign(values.data(), values.size());
      check(logits.argmax() == static_cast<int32_t>(first_best));
      values.back() = std::numeric_limits<float>::quiet_NaN();
      logits.assign(values.data(), values.size());
      rejects_argmax(logits);
    }
  } else if (test == "reuse") {
    const float *storage = logits.values().data();
    logits.assign(second, 4);
    check(logits.values().data() == storage);
    check(logits.values()[0] == 8.0f);
    check(logits.values().size() == 4);
  } else if (test == "invalidate") {
    check(logits.argmax() == 1);
    logits.assign(second, 4);
    check(logits.argmax() == 0);
  } else if (test == "fork") {
    check(logits.argmax() == 1);
    SessionLogits fork = logits;
    logits.assign(second, 4);
    check(fork.argmax() == 1);
    check(logits.argmax() == 0);
    check(fork.values().data() != logits.values().data());
    SessionLogits moved = std::move(fork);
    check(moved.argmax() == 1);
    check(moved.values()[1] == 5.0f);
  } else if (test == "nonfinite") {
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()}) {
      check(logits.argmax() == 1);
      const float values[] = {bad, 5.0f, 3.0f};
      logits.assign(values, 3);
      rejects_argmax(logits);
      rejects_argmax(logits);
      const float late_bad[] = {9.0f, 5.0f, bad};
      logits.assign(late_bad, 3);
      rejects_argmax(logits);
      logits.assign(first, 4);
    }
  } else if (test == "exclude") {
    check(logits.argmax_excluding({}) == 1);
    check(logits.argmax_excluding({1}) == 2);
    check(logits.argmax_excluding({1, 2}) == 3);
    check(logits.argmax() == 1);
    bool rejected = false;
    try { (void)logits.argmax_excluding({0, 1, 2, 3}); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    for (const int excluded : {0, 1}) {
      const float bad[] = {1.0f, std::numeric_limits<float>::quiet_NaN()};
      logits.assign(bad, 2);
      rejected = false;
      try { (void)logits.argmax_excluding({excluded}); }
      catch (const std::runtime_error &) { rejected = true; }
      check(rejected);
    }
    logits.clear();
    rejected = false;
    try { (void)logits.argmax_excluding({}); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
  } else if (test == "exclude-cache") {
    check(logits.argmax_excluding({1}) == 2);
    // A test-only sentinel distinguishes cache reuse from another full scan.
    // Production changes the owned values only through assign()/clear().
    float *storage = const_cast<float *>(logits.values().data());
    storage[0] = 20.0f;
    check(logits.argmax_excluding({1}) == 2);
    storage[0] = first[0];
    check(logits.argmax_excluding({1, 2}) == 3);
    check(logits.argmax_excluding({1}) == 2);
    check(logits.argmax() == 1);
    SessionLogits fork = logits;
    logits.assign(second, 4);
    check(logits.argmax_excluding({1}) == 0);
    check(fork.argmax_excluding({1}) == 2);
    logits.clear();
    bool rejected = false;
    try { (void)logits.argmax_excluding({1}); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    logits.assign(first, 4);
    check(logits.argmax_excluding({1}) == 2);
  } else if (test == "clear") {
    check(logits.argmax() == 1);
    logits.clear();
    check(logits.values().empty());
    rejects_argmax(logits);
    logits.assign(second, 4);
    check(logits.argmax() == 0);
  } else {
    return 2;
  }
  std::cout << "PASS\n";
}
