/* Model-free tests owned by the existing server test translation unit.
 * Expected SHA-1 values use independently specified LE byte preimages. */
static void test_trace_ids_vectors(void) {
    int ids[] = {1, 2};
    ds4_tokens t = {.v = ids, .len = 2};
    char hex[41];
    TEST_ASSERT(trace_ids_sha1(&t, hex) == NULL);
    TEST_ASSERT(!strcmp(hex, "8ebd9d0508b1a332d9c2a5412a69f3102f937bcc"));
    ids[0] = 2; ids[1] = 1;
    TEST_ASSERT(trace_ids_sha1(&t, hex) == NULL);
    TEST_ASSERT(!strcmp(hex, "969bcfff9a67df78d0222288124c893bd54d56b4"));
    ids[0] = 1; t.len = 1;
    TEST_ASSERT(trace_ids_sha1(&t, hex) == NULL);
    TEST_ASSERT(!strcmp(hex, "e1c63cc9c81df165dc23b440f3b18f6100c5f93f"));
    ids[0] = 0x12345678;
    TEST_ASSERT(trace_ids_sha1(&t, hex) == NULL);
    TEST_ASSERT(!strcmp(hex, "f5991fc81a4571af365e4b394c8bc806e93b3071"));
    t.len = 0; t.v = NULL;
    TEST_ASSERT(trace_ids_sha1(&t, hex) == NULL);
    TEST_ASSERT(!strcmp(hex, "05fe405753166f125559e7c9ac558654f107c7e9"));
    TEST_ASSERT(!strcmp(trace_ids_sha1(NULL, hex), "invalid-count"));
    t.len = -1;
    TEST_ASSERT(!strcmp(trace_ids_sha1(&t, hex), "invalid-count"));
    t.len = DS4_TRACE_IDS_LIMIT + 1;
    TEST_ASSERT(!strcmp(trace_ids_sha1(&t, hex), "oversize"));
    t.len = 1;
    TEST_ASSERT(!strcmp(trace_ids_sha1(&t, hex), "invalid-pointer"));
    t.v = ids; ids[0] = -1;
    TEST_ASSERT(!strcmp(trace_ids_sha1(&t, hex), "negative-id"));
    TEST_ASSERT(ids[0] == -1);
}

static void trace_test_read(FILE *fp, char *text, size_t size) {
    rewind(fp);
    size_t n = fread(text, 1, size - 1, fp);
    text[n] = '\0';
}

static void test_server_trace_ids(void) {
    test_trace_ids_vectors();
    server s = {0};
    job j = {0};
    int ids[] = {1, 2}, effective_ids[] = {2, 1};
    j.req.prompt.v = ids;
    j.req.prompt.len = 2;
    ds4_tokens effective = {.v = effective_ids, .len = 2};
    j.req.model = "diagnostic-model";
    j.req.temperature = 0.123456789f;
    j.req.top_p = 0.987654321f;
    j.req.min_p = 0.0123456789f;
    j.req.top_k = 17;
    j.req.max_tokens = 23;
    j.req.seed = UINT64_C(123456789012345);
    j.req.think_mode = DS4_THINK_HIGH;
    j.req.raw_body = "{\"fixture\":true}";
    j.req.prompt_text = "rendered fixture";
    /* Disabled trace must not even dereference the selected pointer. */
    TEST_ASSERT(trace_begin(&s, &j, 0, NULL, NULL, "none", 0, NULL) == 0);
    TEST_ASSERT(s.trace_seq == 0 && ids[0] == 1 && ids[1] == 2);
    s.trace = tmpfile();
    TEST_ASSERT(s.trace != NULL);
    pthread_mutex_init(&s.trace_mu, NULL);
    request before = j.req;
    TEST_ASSERT(trace_begin(&s, &j, 0, &effective, NULL, "none", 0, NULL) == 1);
    char text[8192];
    trace_test_read(s.trace, text, sizeof(text));
    TEST_ASSERT(strstr(text, "canonical_prompt_ids: count=2 format=sha1-le64count-le32ids-v1 sha1=8ebd9d0508b1a332d9c2a5412a69f3102f937bcc\n"));
    TEST_ASSERT(strstr(text, "effective_prompt_ids: count=2 format=sha1-le64count-le32ids-v1 sha1=969bcfff9a67df78d0222288124c893bd54d56b4\n"));
    TEST_ASSERT(!strstr(text, "same-vector="));
    float value;
    TEST_ASSERT(sscanf(strstr(text, "temperature:"), "temperature: %f", &value) == 1);
    TEST_ASSERT(value == j.req.temperature);
    TEST_ASSERT(sscanf(strstr(text, "top_p:"), "top_p: %f", &value) == 1);
    TEST_ASSERT(value == j.req.top_p);
    TEST_ASSERT(sscanf(strstr(text, "min_p:"), "min_p: %f", &value) == 1);
    TEST_ASSERT(value == j.req.min_p);
    TEST_ASSERT(strstr(text, "model: diagnostic-model\n"));
    TEST_ASSERT(strstr(text, "think_mode: high\n"));
    TEST_ASSERT(strstr(text, "seed: 123456789012345\n"));
    TEST_ASSERT(strstr(text, "max_tokens: 23\n"));
    TEST_ASSERT(strstr(text, "top_k: 17\n"));
    TEST_ASSERT(strstr(text, "{\"fixture\":true}\n"));
    TEST_ASSERT(strstr(text, "rendered fixture\n"));
    TEST_ASSERT(!memcmp(&before, &j.req, sizeof(before)));
    TEST_ASSERT(ids[0] == 1 && ids[1] == 2);
    TEST_ASSERT(effective_ids[0] == 2 && effective_ids[1] == 1);
    fclose(s.trace);
    s.trace = tmpfile();
    TEST_ASSERT(s.trace != NULL);
    TEST_ASSERT(trace_begin(&s, &j, 0, &j.req.prompt, NULL, "none", 0, NULL) == 2);
    trace_test_read(s.trace, text, sizeof(text));
    TEST_ASSERT(strstr(text, "effective_prompt_ids: same-vector=canonical_prompt_ids\n"));
    fclose(s.trace);
    /* Distinct vector objects with equal contents still get distinct records. */
    effective.v = ids;
    s.trace = tmpfile();
    TEST_ASSERT(s.trace != NULL);
    trace_begin(&s, &j, 0, &effective, NULL, "none", 0, NULL);
    trace_test_read(s.trace, text, sizeof(text));
    TEST_ASSERT(strstr(text, "effective_prompt_ids: count=2 format=sha1-le64count-le32ids-v1 sha1=8ebd9d0508b1a332d9c2a5412a69f3102f937bcc\n"));
    TEST_ASSERT(!strstr(text, "same-vector="));
    fclose(s.trace);
    s.trace = tmpfile();
    TEST_ASSERT(s.trace != NULL);
    effective.v = NULL; effective.len = DS4_TRACE_IDS_LIMIT + 1;
    j.req.prompt.v = NULL;
    TEST_ASSERT(trace_begin(&s, &j, 0, &effective, NULL, "none", 0, NULL) == 4);
    trace_test_read(s.trace, text, sizeof(text));
    TEST_ASSERT(strstr(text, "canonical_prompt_ids: count=2 format=sha1-le64count-le32ids-v1 unavailable=invalid-pointer\n"));
    TEST_ASSERT(strstr(text, "effective_prompt_ids: count=1048577 format=sha1-le64count-le32ids-v1 unavailable=oversize\n"));
    TEST_ASSERT(j.req.prompt.v == NULL && effective.v == NULL);
    TEST_ASSERT(j.req.prompt.len == 2 && effective.len == DS4_TRACE_IDS_LIMIT + 1);
    TEST_ASSERT(strstr(text, "--- generated text ---"));
    fclose(s.trace);
    s.trace = tmpfile();
    TEST_ASSERT(s.trace != NULL);
    TEST_ASSERT(trace_begin(&s, &j, 0, NULL, NULL, "none", 0, NULL) == 5);
    trace_test_read(s.trace, text, sizeof(text));
    TEST_ASSERT(strstr(text, "effective_prompt_tokens: -1\n"));
    TEST_ASSERT(strstr(text, "effective_prompt_ids: count=-1 format=sha1-le64count-le32ids-v1 unavailable=invalid-count\n"));
    fclose(s.trace);
    pthread_mutex_destroy(&s.trace_mu);
}
