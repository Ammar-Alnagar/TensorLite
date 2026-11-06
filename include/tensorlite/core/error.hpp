#pragma once

#include <exception>
#include <string>
#include <utility>
#include <iostream>
#include <chrono>
#include <format>

namespace tensorlite {

// Base exception for TensorLite runtime errors
class TensorLiteException : public std::exception {
public:
    explicit TensorLiteException(std::string message) : message_(std::move(message)) {}

    [[nodiscard]] const char* what() const noexcept override {
        return message_.c_str();
    }

private:
    std::string message_;
};

class OutOfMemoryException : public TensorLiteException {
public:
    explicit OutOfMemoryException(const std::string& msg)
        : TensorLiteException("CUDA/Host OutOfMemory: " + msg) {}
};

class InvalidArgumentException : public TensorLiteException {
public:
    explicit InvalidArgumentException(const std::string& msg)
        : TensorLiteException("InvalidArgument: " + msg) {}
};

class ModelLoadException : public TensorLiteException {
public:
    explicit ModelLoadException(const std::string& msg)
        : TensorLiteException("ModelLoadException: " + msg) {}
};

// Logging levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

inline LogLevel g_current_log_level = LogLevel::INFO;

inline void set_log_level(LogLevel level) {
    g_current_log_level = level;
}

template <typename... Args>
void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
    if (static_cast<int>(level) < static_cast<int>(g_current_log_level)) {
        return;
    }

    const char* tag = "INFO";
    switch (level) {
        case LogLevel::DEBUG: tag = "DEBUG"; break;
        case LogLevel::INFO:  tag = "INFO "; break;
        case LogLevel::WARN:  tag = "WARN "; break;
        case LogLevel::ERROR: tag = "ERROR"; break;
    }

    auto now = std::chrono::system_clock::now();
    std::string msg = std::format(fmt, std::forward<Args>(args)...);
    std::cout << "[" << tag << "] " << msg << "\n";
}

} // namespace tensorlite
