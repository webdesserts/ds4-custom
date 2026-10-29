/* A failed initial unlink leaves a named cleanup obligation even if fd is gone. */
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
static int fail_unlink(const char *path);
#define unlink fail_unlink
#define DS4_SCR_ENGINE_RAW_STORE
#include "../ds4_scr_engine.inc"
#undef unlink
static int refuse_unlink = 1;
static int fail_unlink(const char *path) {
    if (refuse_unlink) { refuse_unlink = 0; errno = EACCES; return -1; }
    return unlink(path);
}
int main(void) {
    char directory[] = "/tmp/ds4-close-test.XXXXXX", path[PATH_MAX];
    if (!mkdtemp(directory)) return 2;
    ds41_raw_store s = DS41_RAW_STORE_INIT;
    if (ds41_raw_open(&s, directory, 128, ds41_raw_bytes(128)) || s.fd < 0 || !s.cleanup_path[0]) return 3;
    memcpy(path, s.cleanup_path, strlen(s.cleanup_path) + 1u);
    if (close(s.fd)) return 4;
    bool reported = ds41_raw_close(&s);
    bool remains = access(path, F_OK) == 0;
    bool pass = reported && !remains && !s.cleanup_path[0];
    /* Clean only this test's verified original inode after observing the result. */
    if (remains) {
        struct stat st;
        if (lstat(path, &st) || st.st_dev != s.device || st.st_ino != s.inode || unlink(path)) return 5;
    }
    if (rmdir(directory)) return 6;
    printf("close-probe: reported=%d named_path_remained=%d cleanup_record_cleared=%d\n",
           reported, remains, !s.cleanup_path[0]);
    return pass ? 0 : 42;
}
