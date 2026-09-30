// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   page_cache_evict.c
 * @date   30 Sep 2026
 * @brief  Drops a file's pages from the page cache without root, for the
 *         cold cells of a device sitting (plan 201 S0 / S2)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The phone refuses /proc/sys/vm/drop_caches without root (doc 52: "cold
 * was not measured"), but posix_fadvise(POSIX_FADV_DONTNEED) on a file any
 * process can open drops that file's clean, unmapped pages. With an
 * interval it repeats until killed, so every expert a run reads from the
 * model file comes from flash, not only the first read of each.
 *
 * Build: aarch64-linux-android26-clang -O2 -o page_cache_evict <this file>
 *        (the NDK's toolchains/llvm/prebuilt/linux-x86_64/bin)
 * Usage: page_cache_evict <file> [interval_ms]   (no interval: once)
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
  const int fd = argc > 1 ? open(argv[1], O_RDONLY) : -1;
  const long ms = argc > 2 ? atol(argv[2]) : 0;
  if (fd < 0) {
    fprintf(stderr, "usage: %s <file> [interval_ms]\n", argv[0]);
    return 2;
  }
  for (;;) {
    const int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (rc != 0) {
      fprintf(stderr, "posix_fadvise: %d\n", rc);
      return 1;
    }
    if (ms <= 0)
      return 0;
    usleep((useconds_t)(ms * 1000));
  }
}
