// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   pread_probe.c
 * @date   09 Oct 2026
 * @brief  Cold expert-read rate of the model file on the phone, by
 *         destination x mode x threads x request shape (plan 266 S0)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Reads n "experts" -- a gate_up range of 1 993 728 B and a down range of
 * 1 013 760 B each (the Gemma-4 26B QS4CX_WH sizes with scales and sums),
 * at seeded random 64 B-aligned offsets in the file's expert region, so
 * like the real ones they are not 4 KiB-aligned -- after dropping the
 * file's pages (POSIX_FADV_DONTNEED, as page_cache_evict does), under an
 * optional anon pressure. A round is `round` experts read together, then
 * a barrier (the pool's miss round: 3.32 misses on E p512 G512); 0 = all
 * at once.
 *
 *   dest   malloc | ion (rpcmem_alloc heap 25, uncached, as the arena)
 *   mode   pread | random (pread on an fd with POSIX_FADV_RANDOM) |
 *          bounce (O_DIRECT into a 4 KiB-aligned per-thread buffer, then
 *          memcpy) | direct (O_DIRECT straight into dest)
 *   req    slices (today: each weight split in T page-aligned slices read
 *          together, experts one after the other) | 1m (1 MiB requests) |
 *          weight | expert (one thread reads both ranges of an expert)
 *
 * Cell k reads stripe k mod 64 of the expert region (see cell()).
 * One line per cell: GiB/s over the requested bytes, ms per expert (wall
 * / n), ms per round, pgpgin MiB per expert (/proc/vmstat, system-wide),
 * /proc/pressure/io some avg10 at the end, and the first errno.
 *
 * Build: aarch64-linux-android26-clang -O2 -o pread_probe <this file> -ldl
 *        -lpthread (the NDK's toolchains/llvm/prebuilt/linux-x86_64/bin)
 * Usage: pread_probe <file> [--pressure MiB] [--n experts] [--reps r]
 *        [--round R] [--cells all|<dest>,<mode>,<T>,<req>[;...]]
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define GU_BYTES 1993728u
#define DN_BYTES 1013760u
#define SLOT_BYTES (3u << 20)
#define RING 16u
#define MAXJ 4096
#define STRIPES 64u

static const char *DEST[] = {"malloc", "ion"};
static const char *MODE[] = {"pread", "random", "bounce", "direct"};
static const char *REQ[] = {"slices", "1m", "weight", "expert"};

typedef struct {
  uint64_t off; /**< file offset */
  uint32_t len;
  uint8_t *dst;
  uint64_t off2; /**< expert jobs: the down range, else len2 = 0 */
  uint32_t len2;
  uint8_t *dst2;
} job_t;

static struct {
  int fd;
  int mode;
  job_t jobs[MAXJ];
  int njobs;
  volatile int next;
  volatile int err;
  int threads;
  pthread_barrier_t start, end;
  volatile int quit;
} G;

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static long vm_pgpgin_kib(void) {
  FILE *f = fopen("/proc/vmstat", "r");
  char k[64];
  long v = 0, n;
  if (!f)
    return 0;
  while (fscanf(f, "%63s %ld", k, &n) == 2)
    if (!strcmp(k, "pgpgin")) {
      v = n;
      break;
    }
  fclose(f);
  return v;
}

static double psi_io_avg10(void) {
  FILE *f = fopen("/proc/pressure/io", "r");
  double a = -1;
  if (f) {
    if (fscanf(f, "some avg10=%lf", &a) != 1)
      a = -1;
    fclose(f);
  }
  return a;
}

static int read_all(int fd, uint8_t *p, size_t len, uint64_t off) {
  while (len) {
    ssize_t n = pread(fd, p, len, (off_t)off);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return errno;
    }
    if (n == 0)
      return -1;
    p += n;
    len -= (size_t)n;
    off += (uint64_t)n;
  }
  return 0;
}

/** One range as the mode reads it; bounce is this thread's aligned buffer. */
static int read_range(uint64_t off, uint32_t len, uint8_t *dst,
                      uint8_t *bounce) {
  if (G.mode < 2)
    return read_all(G.fd, dst, len, off);
  const uint64_t a0 = off & ~4095ull, a1 = (off + len + 4095ull) & ~4095ull;
  if (G.mode == 2) {
    int rc = read_all(G.fd, bounce, a1 - a0, a0);
    if (rc == 0)
      memcpy(dst, bounce + (off - a0), len);
    return rc;
  }
  /* direct: dst is page-aligned with room for the rounded range */
  return read_all(G.fd, dst, a1 - a0, a0);
}

static void *worker(void *arg) {
  uint8_t *bounce = NULL;
  (void)arg;
  if (posix_memalign((void **)&bounce, 4096, (2u << 20) + 8192))
    return NULL;
  memset(bounce, 0, (2u << 20) + 8192);
  for (;;) {
    pthread_barrier_wait(&G.start);
    if (G.quit)
      break;
    for (;;) {
      int i = __atomic_fetch_add(&G.next, 1, __ATOMIC_RELAXED);
      if (i >= G.njobs)
        break;
      job_t *j = &G.jobs[i];
      int rc = read_range(j->off, j->len, j->dst, bounce);
      if (rc == 0 && j->len2)
        rc = read_range(j->off2, j->len2, j->dst2, bounce);
      if (rc)
        __atomic_compare_exchange_n(&G.err, &(int){0}, rc, 0, __ATOMIC_RELAXED,
                                    __ATOMIC_RELAXED);
    }
    pthread_barrier_wait(&G.end);
  }
  free(bounce);
  return NULL;
}

static void run_round(void) {
  G.next = 0;
  pthread_barrier_wait(&G.start);
  pthread_barrier_wait(&G.end);
}

static uint64_t lcg(uint64_t *s) {
  *s = *s * 6364136223846793005ull + 1442695040888963407ull;
  return *s >> 11;
}

/** Splits [off, off + len) into pieces of `piece` bytes (page-aligned in
 *  the destination, as readWeight's slices are) and appends them. */
static void add_pieces(uint64_t off, uint32_t len, uint8_t *dst,
                       uint32_t piece) {
  for (uint32_t b = 0; b < len && G.njobs < MAXJ; b += piece) {
    job_t *j = &G.jobs[G.njobs++];
    memset(j, 0, sizeof(*j));
    j->off = off + b;
    j->len = len - b < piece ? len - b : piece;
    j->dst = dst + b;
  }
}

typedef struct {
  int dest, mode, T, req;
} cell_t;

static uint8_t *g_ring[2]; /* per destination: RING slots */

static void cell(const char *file, cell_t c, int n, int round, int rep,
                 uint64_t fsize) {
  const int flags = O_RDONLY | (c.mode >= 2 ? O_DIRECT : 0);
  int fd = open(file, flags);
  pthread_t th[8];
  if (fd < 0) {
    printf("cell dest=%s mode=%s T=%d req=%s round=%d rep=%d open_errno=%d\n",
           DEST[c.dest], MODE[c.mode], c.T, REQ[c.req], round, rep, errno);
    return;
  }
  (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  if (c.mode == 1)
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
  G.fd = fd;
  G.mode = c.mode;
  G.err = 0;
  G.quit = 0;
  G.threads = c.T;
  pthread_barrier_init(&G.start, NULL, (unsigned)c.T + 1);
  pthread_barrier_init(&G.end, NULL, (unsigned)c.T + 1);
  for (int t = 0; t < c.T; ++t)
    pthread_create(&th[t], NULL, worker, NULL);

  /* each cell reads its own stripe of the expert region: on the S25 a
   * range re-read after POSIX_FADV_DONTNEED shows less pgpgin than its
   * first read (not resident by mincore; source not found), so no cell
   * re-reads the previous ones' ranges */
  static unsigned stripe_next;
  const unsigned stripe = stripe_next++ % STRIPES;
  uint64_t seed = 0x266u + (uint64_t)stripe * 7919u + (uint64_t)rep;
  const uint64_t span = (fsize - fsize / 16 - (64ull << 20)) / STRIPES;
  const uint64_t lo = fsize / 16 + stripe * span;
  const int R = round > 0 ? round : n;
  int rounds = 0;
  const long pg0 = vm_pgpgin_kib();
  const double t0 = now_s();
  for (int e0 = 0; e0 < n; e0 += R) {
    const int e1 = e0 + R < n ? e0 + R : n;
    G.njobs = 0;
    for (int e = e0; e < e1; ++e) {
      const uint64_t gu = lo + (lcg(&seed) % span & ~63ull);
      const uint64_t dn = lo + (lcg(&seed) % span & ~63ull);
      uint8_t *slot = g_ring[c.dest] + (size_t)(e % RING) * SLOT_BYTES;
      /* direct: each range gets its own page-aligned window */
      uint8_t *dgu = slot, *ddn = slot + ((GU_BYTES + 8191u) & ~4095u);
      if (c.req == 0) { /* slices: weight by weight, one barrier each */
        const uint32_t pgu = ((GU_BYTES + c.T - 1) / c.T + 4095u) & ~4095u;
        const uint32_t pdn = ((DN_BYTES + c.T - 1) / c.T + 4095u) & ~4095u;
        G.njobs = 0;
        add_pieces(gu, GU_BYTES, dgu, pgu);
        run_round();
        G.njobs = 0;
        add_pieces(dn, DN_BYTES, ddn, pdn);
        run_round();
        rounds += 2;
        continue;
      }
      if (c.req == 1) {
        add_pieces(gu, GU_BYTES, dgu, 1u << 20);
        add_pieces(dn, DN_BYTES, ddn, 1u << 20);
      } else if (c.req == 2) {
        add_pieces(gu, GU_BYTES, dgu, GU_BYTES);
        add_pieces(dn, DN_BYTES, ddn, DN_BYTES);
      } else {
        job_t *j = &G.jobs[G.njobs++];
        j->off = gu;
        j->len = GU_BYTES;
        j->dst = dgu;
        j->off2 = dn;
        j->len2 = DN_BYTES;
        j->dst2 = ddn;
      }
    }
    if (c.req != 0) {
      run_round();
      ++rounds;
    }
  }
  const double dt = now_s() - t0;
  const long pg1 = vm_pgpgin_kib();
  G.quit = 1;
  pthread_barrier_wait(&G.start);
  for (int t = 0; t < c.T; ++t)
    pthread_join(th[t], NULL);
  pthread_barrier_destroy(&G.start);
  pthread_barrier_destroy(&G.end);
  (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  close(fd);
  const double bytes = (double)n * (GU_BYTES + DN_BYTES);
  printf("cell dest=%s mode=%s T=%d req=%s round=%d rep=%d n=%d "
         "GiBps=%.3f ms_per_expert=%.3f ms_per_round=%.3f "
         "pgpgin_mib_per_expert=%.3f psi_io_avg10=%.2f err=%d\n",
         DEST[c.dest], MODE[c.mode], c.T, REQ[c.req], c.req == 0 ? 0 : round,
         rep, n, bytes / dt / (1u << 30), dt * 1e3 / n,
         dt * 1e3 / (rounds ? rounds : 1), (pg1 - pg0) / 1024.0 / n,
         psi_io_avg10(), G.err);
  fflush(stdout);
}

static int idx(const char **names, int n, const char *s) {
  for (int i = 0; i < n; ++i)
    if (!strcmp(names[i], s))
      return i;
  return -1;
}

int main(int argc, char **argv) {
  const char *file = argc > 1 ? argv[1] : NULL;
  long pressure = 0;
  int n = 256, reps = 1, round = 3;
  const char *cells = "all";
  for (int i = 2; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--pressure"))
      pressure = atol(argv[i + 1]);
    else if (!strcmp(argv[i], "--n"))
      n = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--reps"))
      reps = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--round"))
      round = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--cells"))
      cells = argv[i + 1];
  }
  struct stat st;
  int fd0 = file ? open(file, O_RDONLY) : -1;
  if (fd0 < 0 || fstat(fd0, &st) != 0 || n < 1 || n * 4 > MAXJ) {
    fprintf(stderr,
            "usage: %s <file> [--pressure MiB] [--n experts<=1024] "
            "[--reps r] [--round R] [--cells all|d,m,T,r;...]\n",
            argv[0]);
    return 2;
  }
  close(fd0);

  /* pressure: anon, touched with non-constant data so zram cannot fold it */
  if (pressure > 0) {
    uint64_t s = 1;
    uint64_t *p = malloc((size_t)pressure << 20);
    if (!p) {
      fprintf(stderr, "pressure: malloc %ld MiB failed\n", pressure);
      return 1;
    }
    for (size_t i = 0; i < ((size_t)pressure << 20) / 8; ++i)
      p[i] = lcg(&s);
    printf("pressure %ld MiB touched\n", pressure);
  }

  /* destinations: a ring of RING 3 MiB slots each */
  const size_t ring = (size_t)RING * SLOT_BYTES;
  if (posix_memalign((void **)&g_ring[0], 4096, ring))
    return 1;
  memset(g_ring[0], 1, ring);
  void *lib = dlopen("libcdsprpc.so", RTLD_NOW);
  void *(*ra)(int, uint32_t, int) =
    lib ? (void *(*)(int, uint32_t, int))dlsym(lib, "rpcmem_alloc") : NULL;
  void (*ri)(void) = lib ? (void (*)(void))dlsym(lib, "rpcmem_init") : NULL;
  if (ri)
    ri();
  g_ring[1] = ra ? ra(25, 0, (int)ring) : NULL; /* heap 25, uncached */
  if (g_ring[1])
    memset(g_ring[1], 1, ring);
  printf("ion %s (%zu MiB)\n", g_ring[1] ? "ok" : "unavailable", ring >> 20);

  cell_t list[256];
  int nc = 0;
  if (!strcmp(cells, "all")) {
    /* ion (the arena's memory) x pread / random / bounce x T x req; direct
     * into ion once (it cannot work: EFAULT); malloc as a control */
    for (int r = 0; r < 4; ++r)
      for (int m = 0; m < 3; ++m)
        for (int t = 1; t <= 8; t *= 2)
          list[nc++] = (cell_t){1, m, t, r};
    list[nc++] = (cell_t){1, 3, 4, 3};
    for (int m = 0; m < 4; m += 2)
      for (int t = 4; t <= 8; t *= 2) {
        list[nc++] = (cell_t){0, m, t, 0};
        list[nc++] = (cell_t){0, m, t, 3};
      }
  } else {
    char *dup = strdup(cells), *save = NULL;
    for (char *tok = strtok_r(dup, ";", &save); tok && nc < 256;
         tok = strtok_r(NULL, ";", &save)) {
      char d[16], m[16], r[16];
      int t;
      if (sscanf(tok, "%15[^,],%15[^,],%d,%15s", d, m, &t, r) == 4 &&
          idx(DEST, 2, d) >= 0 && idx(MODE, 4, m) >= 0 && idx(REQ, 4, r) >= 0 &&
          t >= 1 && t <= 8)
        list[nc++] =
          (cell_t){idx(DEST, 2, d), idx(MODE, 4, m), t, idx(REQ, 4, r)};
      else
        fprintf(stderr, "bad cell '%s'\n", tok);
    }
  }
  for (int rep = 0; rep < reps; ++rep)
    for (int i = 0; i < nc; ++i)
      if (list[i].dest == 0 || g_ring[1])
        cell(file, list[i], n, round, rep, (uint64_t)st.st_size);
  return 0;
}
