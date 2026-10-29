#include "ds4_scr.h"
#include <stdio.h>
int main(void) {
    uint32_t v = 0;
    if (DS4_SCR_PLANNER_DEFAULT_BUDGET_MS != 5000u) return 1;
    if (ds4_scr_parse_planner_budget("5000", &v) || v != 5000u) return 2;
    if (ds4_scr_parse_planner_budget("0", &v) || v != 0u) return 3;
    if (ds4_scr_parse_planner_budget("60000", &v) || v != 60000u) return 4;
    if (!ds4_scr_parse_planner_budget("-1", &v)) return 5;
    if (!ds4_scr_parse_planner_budget("60001", &v)) return 6;
    if (!ds4_scr_parse_planner_budget("5000x", &v)) return 7;
    if (!ds4_scr_parse_planner_budget("", &v)) return 8;
    puts("budget_parse_test: ok");
    return 0;
}
