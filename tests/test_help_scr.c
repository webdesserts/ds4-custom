/* Server help lists exactly the surviving SCR options; retired selectors are
 * still accepted by the parser but no longer documented. */
#include "../ds4_help.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../ds4.h"
/* Match an option's own help entry, not a description that mentions it. */
static int option_entry(const char *text, const char *token) {
    const char *at = text;
    while (*at) {
        const char *line = at;
        while (*line == ' ' || *line == '\t') ++line;
        size_t n = strlen(token);
        if (!strncmp(line, token, n) && (line[n] == ' ' || line[n] == '\t' || line[n] == '\0')) return 1;
        at = strchr(at, '\n');
        if (!at) break;
        ++at;
    }
    return 0;
}
int main(void) {
    FILE *f = tmpfile(); assert(f);
    ds4_help_print(f, DS4_HELP_SERVER, "api"); rewind(f);
    char buf[65536]; size_t n = fread(buf, 1, sizeof(buf) - 1, f); buf[n] = 0; fclose(f);
    const char *kept[] = {"--scr", "--scr-tail-tokens", "--scr-max-ranges", "--scr-planner-budget-ms"};
    for (size_t i = 0; i < sizeof(kept) / sizeof(kept[0]); ++i) assert(option_entry(buf, kept[i]));
    const char *retired[] = {"--scr-capture-batch", "--scr-capture-batch-trial", "--scr-capture-row-chunks",
        "--scr-logical-coordinates", "--scr-padding-policy", "--scr-batch-suffix", "--scr-noncapturing-fresh-once"};
    for (size_t i = 0; i < sizeof(retired) / sizeof(retired[0]); ++i) assert(!option_entry(buf, retired[i]));
    char range[32]; snprintf(range, sizeof(range), "1..%u", DS4_SCR_MULTI_MAX_RANGES);
    assert(strstr(buf, range) != NULL);
    assert(strstr(buf, "known unequal logits") != NULL);
    f = tmpfile(); assert(f);
    ds4_help_print(f, DS4_HELP_SERVER, "kv-cache"); rewind(f);
    n = fread(buf, 1, sizeof(buf) - 1, f); buf[n] = 0; fclose(f);
    assert(strstr(buf, "persistent SCR checkpoint pool") != NULL);
    puts("help: surviving SCR options, retired selectors hidden, K bound, unequal-logit warning, checkpoint pool");
    return 0;
}
