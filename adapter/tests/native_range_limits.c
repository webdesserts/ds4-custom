#include "ds4.h"

typedef char scr_k32_constant_assert[(DS4_SCR_MULTI_MAX_RANGES == 32u) ? 1 : -1];
typedef char scr_k32_receipt_assert[
    (sizeof(((ds4_scr_multi_receipt *)0)->ranges) / sizeof(ds4_scr_range) == 32u) ? 1 : -1];
int main(void) {
    return 0;
}
