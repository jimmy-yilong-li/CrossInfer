#pragma once

#include <streambuf>
#include <time.h>

inline long long first_token_monotonic_us() {
    timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return (long long)value.tv_sec * 1000000LL + value.tv_nsec / 1000;
}

// MNN flushes its output stream after sampling each token, including EOS.
// Discard text without altering generation; timestamp only the first flush.
class FirstTokenClock : public std::streambuf {
public:
    long long first_us = 0;
    void reset() { first_us = 0; }

protected:
    std::streamsize xsputn(const char*, std::streamsize count) override { return count; }
    int_type overflow(int_type value) override { return traits_type::not_eof(value); }
    int sync() override {
        if (first_us == 0) first_us = first_token_monotonic_us();
        return 0;
    }
};
