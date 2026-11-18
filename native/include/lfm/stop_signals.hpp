#pragma once
// SIGTERM and SIGINT go to one waiting thread (sigwait). main blocks them before it starts any thread, so every thread
// the program starts inherits the block. A thread started without that mask (a sanitizer's, a library's) could take a
// process-directed SIGTERM with the default action and kill the process on the spot instead of the orderly shutdown:
// image-cache-v1 quiet window 13, liquid-lfm3-server under compute-sanitizer memcheck, exit 15 with one such thread.
//
// become_waiter, called on the waiting thread before its first sigwait, records that thread and installs a handler for
// the waited signals. The handler runs only on a thread that does not block them (every thread the program started
// blocks them; the waiter takes them through sigwait, which takes precedence over a handler) and does one thing: it
// re-sends the signal to the waiter (tgkill), whose sigwait then returns it as if it had been delivered there.
// Async-signal-safe: a lock-free atomic load and two syscalls; errno is kept.
#include <atomic>
#include <cerrno>
#include <csignal>
#include <stdexcept>

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

namespace lfm::stop_signals {

inline std::atomic<pid_t>& waiter() {
  static std::atomic<pid_t> tid{0};
  return tid;
}

inline void forward(int sig) {
  const int saved = errno;
  const pid_t tid = waiter().load();
  if (tid > 0) ::syscall(SYS_tgkill, ::getpid(), tid, sig);
  errno = saved;
}

// On the waiting thread, with `signals` (SIGTERM, SIGINT) blocked: before the handler exists, so it always has a
// thread to forward to.
inline void become_waiter(const sigset_t& signals) {
  static_assert(std::atomic<pid_t>::is_always_lock_free);
  waiter() = pid_t(::syscall(SYS_gettid));
  struct sigaction action{};
  action.sa_handler = forward;
  sigfillset(&action.sa_mask);
  action.sa_flags = SA_RESTART;
  for (const int s : {SIGTERM, SIGINT})
    if (sigismember(&signals, s) == 1 && ::sigaction(s, &action, nullptr) != 0)
      throw std::runtime_error("cannot install the stop-signal handler");
}

}  // namespace lfm::stop_signals
