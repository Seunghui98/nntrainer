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
 * Usage: page_cache_evict <file> [interval_ms]   (no interval: once, and
 *        the file's own resident pages before and after, by mincore, on
 *        stdout: "resident <a> -> <b> MiB of <c> MiB"); interval -1:
 *        only report "resident <a> MiB of <c> MiB", evict nothing
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/** @brief The file's pages in the page cache, MiB (mincore on a mapping
 *  that is never touched, so counting faults nothing in); -1 on error. */
static long resident_mib(int fd, size_t len) {
  const long pg = sysconf(_SC_PAGESIZE);
  const size_t n = (len + (size_t)pg - 1) / (size_t)pg;
  unsigned char *v = malloc(n ? n : 1);
  void *m = len ? mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
  long in = -1;
  if (v != NULL && m != MAP_FAILED && mincore(m, len, v) == 0) {
    size_t i, c = 0;
    for (i = 0; i < n; ++i)
      c += v[i] & 1u;
    in = (long)((c * (size_t)pg) >> 20);
  }
  if (m != MAP_FAILED)
    munmap(m, len);
  free(v);
  return in;
}

int main(int argc, char **argv) {
  const int fd = argc > 1 ? open(argv[1], O_RDONLY) : -1;
  const long ms = argc > 2 ? atol(argv[2]) : 0;
  if (fd < 0) {
    fprintf(stderr, "usage: %s <file> [interval_ms]\n", argv[0]);
    return 2;
  }
  struct stat st;
  const size_t len = fstat(fd, &st) == 0 ? (size_t)st.st_size : 0u;
  const long before = ms <= 0 ? resident_mib(fd, len) : 0;
  if (ms < 0) {
    printf("resident %ld MiB of %zu MiB\n", before, len >> 20);
    return 0;
  }
  for (;;) {
    const int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (rc != 0) {
      fprintf(stderr, "posix_fadvise: %d\n", rc);
      return 1;
    }
    if (ms <= 0) {
      printf("resident %ld -> %ld MiB of %zu MiB\n", before,
             resident_mib(fd, len), len >> 20);
      return 0;
    }
    usleep((useconds_t)(ms * 1000));
  }
}
